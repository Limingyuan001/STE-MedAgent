import logging
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("TOOL_SERVER_CUDA_VISIBLE_DEVICES", "0")

import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
import torch

from medrax.llava.mm_utils import process_images
from medrax.llava.model.builder import load_pretrained_model
from medrax.biovil_t_image import BIOVIL_T_JOINT_FEATURE_SIZE, BIOVIL_T_MODEL_NAME
from medrax.biovil_t_image import BioViLTImageInferenceEngine
from medrax.tools.remote import encode_for_transport


ROOT = os.getenv("MEDRAX_ROOT", os.path.abspath(os.path.dirname(__file__)))
MODEL_DIR = os.getenv("TOOL_SERVER_MODEL_DIR", f"{ROOT}/benchmark/tool_models")
TOOL_DEVICE = os.getenv("TOOL_SERVER_DEVICE", "cuda")
HOST = os.getenv("TOOL_SERVER_MIMIC_HOST", os.getenv("TOOL_SERVER_HOST", "0.0.0.0"))
PORT = int(os.getenv("TOOL_SERVER_MIMIC_PORT", "8011"))
LOG_LEVEL = os.getenv("TOOL_SERVER_MIMIC_LOG_LEVEL", os.getenv("TOOL_SERVER_LOG_LEVEL", "info"))
LLAVA_MODEL_PATH = os.getenv(
    "TOOL_SERVER_MIMIC_LLAVA_MODEL",
    "microsoft/llava-med-v1.5-mistral-7b",
)
LLAVA_LOAD_IN_8BIT = os.getenv("TOOL_SERVER_MIMIC_LLAVA_LOAD_IN_8BIT", "1").lower() not in {
    "0",
    "false",
    "off",
}
DEFAULT_IMAGE_EMBEDDING_MODEL = os.getenv(
    "TOOL_SERVER_MIMIC_DEFAULT_EMBEDDING_MODEL",
    "llava-med-clip-patch-mean-1024",
)
SUPPORTED_IMAGE_EMBEDDING_MODELS = {
    "llava-med-clip-patch-mean-1024",
    BIOVIL_T_MODEL_NAME,
}


app = FastAPI(title="MedRAX MIMIC Tool Server")
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

tool_lock = Lock()
stats_lock = Lock()
tool_instances: Dict[str, Any] = {}
tools_ready = False
server_stats: Dict[str, int] = {
    "inflight_requests": 0,
    "active_requests": 0,
    "queued_requests": 0,
    "completed_requests": 0,
    "failed_requests": 0,
    "max_queued_requests": 0,
    "request_seq": 0,
}
per_tool_stats: Dict[str, Dict[str, int]] = {}


class ImageEmbeddingInput(BaseModel):
    image_path: str
    embedding_model: str = DEFAULT_IMAGE_EMBEDDING_MODEL


class CheXbertLabelsInput(BaseModel):
    texts: List[str]


class LlavaMedImageEmbeddingService:
    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        device: str,
        load_in_8bit: bool,
    ) -> None:
        import torch

        self.torch = torch
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_path,
            load_in_4bit=False,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        self.model.eval()
        self.vision_tower = self.model.get_vision_tower()

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        return image_tensor.unsqueeze(0).to(device=self.model.device, dtype=self.model.dtype)

    def embed_image(self, image_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image_tensor = self._load_image_tensor(image_path)
        with self.torch.inference_mode():
            patch_features = self.vision_tower(image_tensor)

        if isinstance(patch_features, list):
            patch_features = patch_features[0]
        if patch_features.ndim != 3:
            raise ValueError(f"Unexpected patch feature shape: {tuple(patch_features.shape)}")

        pooled = patch_features.mean(dim=1)[0]
        pooled = pooled / pooled.norm(p=2).clamp(min=1e-12)
        embedding = pooled.float().detach().cpu().tolist()

        feature_shape = list(patch_features.shape[1:])
        embedding_dim = int(len(embedding))
        payload = {
            "image_embedding": embedding,
            "image_embedding_model": "llava-med-clip-patch-mean-1024",
            "vision_tower_name": getattr(self.model.config, "mm_vision_tower", ""),
            "vision_select_layer": getattr(self.model.config, "mm_vision_select_layer", None),
            "vision_select_feature": getattr(self.model.config, "mm_vision_select_feature", None),
            "feature_shape": feature_shape,
            "embedding_dim": embedding_dim,
            "num_patches": int(feature_shape[0]) if feature_shape else None,
        }
        metadata = {
            "image_path": image_path,
            "analysis_status": "completed",
        }
        return payload, metadata

    def invoke(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.embed_image(str(args.get("image_path", "")))


class BioViLTImageEmbeddingService:
    def __init__(self, cache_dir: str, device: str) -> None:
        self.engine = BioViLTImageInferenceEngine(cache_dir=cache_dir, device=device)

    def embed_image(self, image_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        embedding = self.engine.get_projected_global_embedding(image_path)
        embedding_list = embedding.float().detach().cpu().tolist()
        payload = {
            "image_embedding": embedding_list,
            "image_embedding_model": BIOVIL_T_MODEL_NAME,
            "feature_shape": [BIOVIL_T_JOINT_FEATURE_SIZE],
            "embedding_dim": int(len(embedding_list)),
            "num_patches": None,
        }
        metadata = {
            "image_path": image_path,
            "analysis_status": "completed",
        }
        return payload, metadata

    def invoke(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.embed_image(str(args.get("image_path", "")))


class RoutedImageEmbeddingService:
    def __init__(
        self,
        cache_dir: str,
        device: str,
        llava_model_path: str,
        llava_load_in_8bit: bool,
    ) -> None:
        self.cache_dir = cache_dir
        self.device = device
        self.llava_model_path = llava_model_path
        self.llava_load_in_8bit = llava_load_in_8bit
        self._backends: Dict[str, Any] = {}

    def _get_backend(self, embedding_model: str) -> Any:
        if embedding_model not in SUPPORTED_IMAGE_EMBEDDING_MODELS:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_EMBEDDING_MODELS))
            raise ValueError(f"Unsupported embedding_model={embedding_model!r}. Supported: {supported}")
        backend = self._backends.get(embedding_model)
        if backend is not None:
            return backend

        if embedding_model == BIOVIL_T_MODEL_NAME:
            backend = BioViLTImageEmbeddingService(
                cache_dir=self.cache_dir,
                device=self.device,
            )
        else:
            backend = LlavaMedImageEmbeddingService(
                model_path=self.llava_model_path,
                cache_dir=self.cache_dir,
                device=self.device,
                load_in_8bit=self.llava_load_in_8bit,
            )
        self._backends[embedding_model] = backend
        return backend

    def invoke(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image_path = str(args.get("image_path", ""))
        embedding_model = str(args.get("embedding_model", DEFAULT_IMAGE_EMBEDDING_MODEL) or DEFAULT_IMAGE_EMBEDDING_MODEL)
        backend = self._get_backend(embedding_model)
        payload, metadata = backend.embed_image(image_path)
        metadata = {
            **metadata,
            "embedding_model": embedding_model,
        }
        return payload, metadata


class CheXbertLabelService:
    def __init__(self, device: str) -> None:
        try:
            from f1chexbert import F1CheXbert
        except ImportError as exc:
            raise RuntimeError(
                "f1chexbert is required for the CheXbert service. "
                "Install it in the active environment first."
            ) from exc

        self.model = F1CheXbert(device=device)
        self.device = str(self.model.device)
        self.target_names = list(self.model.target_names)
        self.target_names_5_index = [int(i) for i in self.model.target_names_5_index]

    def label_texts(self, texts: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        labels_14 = [list(map(int, self.model.get_label(str(text or "").strip()))) for text in texts]
        payload = {
            "labels_14": labels_14,
            "target_names_14": self.target_names,
            "target_names_5_index": self.target_names_5_index,
        }
        metadata = {
            "num_texts": len(texts),
            "device": self.device,
            "analysis_status": "completed",
        }
        return payload, metadata

    def invoke(self, args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        texts = args.get("texts", [])
        if not isinstance(texts, list):
            raise ValueError("Expected `texts` to be a list")
        return self.label_texts([str(text or "") for text in texts])


def ensure_tool_stats_locked(tool_key: str) -> Dict[str, int]:
    tool_stats = per_tool_stats.get(tool_key)
    if tool_stats is None:
        tool_stats = {
            "inflight_requests": 0,
            "active_requests": 0,
            "queued_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "max_queued_requests": 0,
        }
        per_tool_stats[tool_key] = tool_stats
    return tool_stats


def snapshot_stats() -> Dict[str, Any]:
    with stats_lock:
        return {
            **dict(server_stats),
            "per_tool_stats": {
                tool_key: dict(tool_stats)
                for tool_key, tool_stats in per_tool_stats.items()
            },
        }


def build_tool_instances() -> Dict[str, Any]:
    return {
        "llava_med_image_embedding": RoutedImageEmbeddingService(
            cache_dir=MODEL_DIR,
            device=TOOL_DEVICE,
            llava_model_path=LLAVA_MODEL_PATH,
            llava_load_in_8bit=LLAVA_LOAD_IN_8BIT,
        ),
        "chexbert_labels": CheXbertLabelService(
            device=os.getenv("TOOL_SERVER_MIMIC_CHEXBERT_DEVICE", TOOL_DEVICE),
        ),
    }


def run_tool(tool_key: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if not tools_ready or tool_key not in tool_instances:
        raise HTTPException(status_code=503, detail="tool server is not ready")

    tool = tool_instances[tool_key]
    lock_acquired = False
    was_queued = False
    request_id = 0
    start_time = time.monotonic()

    with stats_lock:
        server_stats["request_seq"] += 1
        request_id = server_stats["request_seq"]
        server_stats["inflight_requests"] += 1
        ensure_tool_stats_locked(tool_key)["inflight_requests"] += 1

    logger.info(
        "tool_request_start request_id=%s tool=%s inflight=%s active=%s queued=%s",
        request_id,
        tool_key,
        snapshot_stats()["inflight_requests"],
        snapshot_stats()["active_requests"],
        snapshot_stats()["queued_requests"],
    )

    try:
        lock_acquired = tool_lock.acquire(blocking=False)
        if not lock_acquired:
            with stats_lock:
                server_stats["queued_requests"] += 1
                server_stats["max_queued_requests"] = max(
                    server_stats["max_queued_requests"],
                    server_stats["queued_requests"],
                )
                tool_stats = ensure_tool_stats_locked(tool_key)
                tool_stats["queued_requests"] += 1
                tool_stats["max_queued_requests"] = max(
                    tool_stats["max_queued_requests"],
                    tool_stats["queued_requests"],
                )
            was_queued = True
            current = snapshot_stats()
            logger.info(
                "tool_request_queued request_id=%s tool=%s active=%s queued=%s tool_queued=%s",
                request_id,
                tool_key,
                current["active_requests"],
                current["queued_requests"],
                current["per_tool_stats"][tool_key]["queued_requests"],
            )
            tool_lock.acquire()
            lock_acquired = True

        with stats_lock:
            if was_queued:
                server_stats["queued_requests"] -= 1
                ensure_tool_stats_locked(tool_key)["queued_requests"] -= 1
            server_stats["active_requests"] += 1
            ensure_tool_stats_locked(tool_key)["active_requests"] += 1

        current = snapshot_stats()
        logger.info(
            "tool_request_execute request_id=%s tool=%s active=%s queued=%s tool_active=%s tool_queued=%s waited_sec=%.3f",
            request_id,
            tool_key,
            current["active_requests"],
            current["queued_requests"],
            current["per_tool_stats"][tool_key]["active_requests"],
            current["per_tool_stats"][tool_key]["queued_requests"],
            time.monotonic() - start_time,
        )

        try:
            result = tool.invoke(args)
        finally:
            with stats_lock:
                server_stats["active_requests"] -= 1
                ensure_tool_stats_locked(tool_key)["active_requests"] -= 1
            tool_lock.release()
            lock_acquired = False
    except Exception as exc:
        with stats_lock:
            server_stats["failed_requests"] += 1
            server_stats["inflight_requests"] -= 1
            tool_stats = ensure_tool_stats_locked(tool_key)
            tool_stats["failed_requests"] += 1
            tool_stats["inflight_requests"] -= 1
        if lock_acquired:
            with stats_lock:
                server_stats["active_requests"] = max(0, server_stats["active_requests"] - 1)
                ensure_tool_stats_locked(tool_key)["active_requests"] = max(
                    0,
                    ensure_tool_stats_locked(tool_key)["active_requests"] - 1,
                )
            tool_lock.release()
        current = snapshot_stats()
        logger.exception(
            "tool_request_failed request_id=%s tool=%s elapsed_sec=%.3f inflight=%s active=%s queued=%s tool_failed=%s",
            request_id,
            tool_key,
            time.monotonic() - start_time,
            current["inflight_requests"],
            current["active_requests"],
            current["queued_requests"],
            current["per_tool_stats"][tool_key]["failed_requests"],
        )
        raise HTTPException(status_code=500, detail=f"tool execution failed: {exc}") from exc
    else:
        with stats_lock:
            server_stats["completed_requests"] += 1
            server_stats["inflight_requests"] -= 1
            tool_stats = ensure_tool_stats_locked(tool_key)
            tool_stats["completed_requests"] += 1
            tool_stats["inflight_requests"] -= 1
        current = snapshot_stats()
        logger.info(
            "tool_request_done request_id=%s tool=%s elapsed_sec=%.3f inflight=%s active=%s queued=%s completed=%s failed=%s tool_completed=%s tool_failed=%s",
            request_id,
            tool_key,
            time.monotonic() - start_time,
            current["inflight_requests"],
            current["active_requests"],
            current["queued_requests"],
            current["completed_requests"],
            current["failed_requests"],
            current["per_tool_stats"][tool_key]["completed_requests"],
            current["per_tool_stats"][tool_key]["failed_requests"],
        )

    payload: Any
    metadata: Dict[str, Any]
    if isinstance(result, tuple) and len(result) == 2:
        payload, metadata = result
    else:
        payload, metadata = result, {}

    return {
        "payload": encode_for_transport(payload),
        "metadata": encode_for_transport(metadata),
    }


@app.on_event("startup")
def startup_event() -> None:
    global tool_instances, tools_ready
    tool_instances = build_tool_instances()
    tools_ready = True


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> Dict[str, Any]:
    return {
        "ready": tools_ready,
        "loaded_tools": sorted(tool_instances.keys()),
        "stats": snapshot_stats(),
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return {
        "ready": tools_ready,
        "loaded_tools": sorted(tool_instances.keys()),
        "stats": snapshot_stats(),
    }


@app.post("/tools/llava_med_image_embedding")
def llava_med_image_embedding(payload: ImageEmbeddingInput) -> Dict[str, Any]:
    return run_tool("llava_med_image_embedding", payload.dict())


@app.post("/tools/chexbert_labels")
def chexbert_labels(payload: CheXbertLabelsInput) -> Dict[str, Any]:
    return run_tool("chexbert_labels", payload.dict())


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)


if __name__ == "__main__":
    main()
