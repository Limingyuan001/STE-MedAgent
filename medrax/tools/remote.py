import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import httpx
from pydantic import BaseModel, PrivateAttr

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from .classification import ChestXRayClassifierTool
from .classification import ChestXRayInput as ClassificationInput
from .grounding import XRayPhraseGroundingTool
from .grounding import XRayPhraseGroundingInput
from .llava_med import LlavaMedTool
from .llava_med import LlavaMedInput
from .report_generation import ChestXRayReportGeneratorTool
from .report_generation import ChestXRayInput as ReportGenerationInput
from .segmentation import ChestXRaySegmentationTool
from .segmentation import ChestXRaySegmentationInput
from .xray_vqa import XRayVQATool
from .xray_vqa import XRayVQAToolInput


TRANSPORT_TYPE_KEY = "__medrax_transport_type__"

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in lightweight envs
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in lightweight envs
    torch = None


def encode_for_transport(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if np is not None and isinstance(value, np.generic):
        return encode_for_transport(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None and isinstance(value, np.ndarray):
        return encode_for_transport(value.tolist())
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return encode_for_transport(tensor.item())
        return encode_for_transport(tensor.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return {
            TRANSPORT_TYPE_KEY: "tuple",
            "items": [encode_for_transport(item) for item in value],
        }
    if isinstance(value, list):
        return [encode_for_transport(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode_for_transport(item) for key, item in value.items()}
    return str(value)


def decode_from_transport(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(TRANSPORT_TYPE_KEY) == "tuple":
            return tuple(decode_from_transport(item) for item in value.get("items", []))
        return {key: decode_from_transport(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_from_transport(item) for item in value]
    return value


class _RemoteToolBase(BaseTool):
    api_base_url: str = "http://127.0.0.1:8010"
    api_timeout: float = 300.0

    _client: httpx.Client = PrivateAttr()

    endpoint: str = ""

    def __init__(self, api_base_url: str = "http://127.0.0.1:8010", api_timeout: float = 300.0, **kwargs: Any):
        super().__init__()
        self.api_base_url = api_base_url.rstrip("/")
        self.api_timeout = float(api_timeout)
        timeout = httpx.Timeout(self.api_timeout, connect=min(10.0, self.api_timeout))
        self._client = httpx.Client(timeout=timeout)

    def _invoke_remote(self, payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        url = f"{self.api_base_url}{self.endpoint}"
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(
                f"Tool API request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Tool API request failed: {exc}") from exc

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Tool API response is not a JSON object")
        if "payload" not in body or "metadata" not in body:
            raise ValueError("Tool API response is missing payload/metadata")

        return (
            decode_from_transport(body["payload"]),
            decode_from_transport(body["metadata"]),
        )

    def _request_or_fallback(
        self,
        payload: Dict[str, Any],
        fallback_builder,
    ) -> Tuple[Any, Dict[str, Any]]:
        # API-layer failures should stop the benchmark sample rather than becoming
        # normal tool observations that can mislead the agent's final answer.
        return self._invoke_remote(payload)


class RemoteChestXRayReportGeneratorTool(_RemoteToolBase):
    name: str = "chest_xray_report_generator"
    description: str = (
        "A tool that analyzes chest X-ray images and generates comprehensive radiology reports "
        "containing both detailed findings and impression summaries. Input should be the path "
        "to a chest X-ray image file. Output is a structured report with both detailed "
        "observations and key clinical conclusions."
    )
    args_schema: Type[BaseModel] = ReportGenerationInput
    endpoint: str = "/tools/chest_xray_report_generator"

    def __init__(
        self,
        cache_dir: str = "/model-weights",
        device: Optional[str] = "cuda",
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(api_base_url=api_base_url, api_timeout=api_timeout, **kwargs)

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        def _fallback(exc: Exception) -> Tuple[str, Dict]:
            message = str(exc)
            return f"Error generating report: {message}", {
                "image_path": image_path,
                "analysis_status": "failed",
                "error": message,
            }

        return self._request_or_fallback({"image_path": image_path}, _fallback)

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        return self._run(image_path)


class RemoteChestXRayClassifierTool(_RemoteToolBase):
    name: str = "chest_xray_classifier"
    description: str = (
        "A tool that analyzes chest X-ray images and classifies them for 18 different pathologies. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their predicted probabilities (0 to 1). "
        "Pathologies include: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, "
        "Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ClassificationInput
    endpoint: str = "/tools/chest_xray_classifier"

    def __init__(
        self,
        model_name: str = "densenet121-res224-all",
        device: Optional[str] = "cuda",
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(api_base_url=api_base_url, api_timeout=api_timeout, **kwargs)

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        def _fallback(exc: Exception) -> Tuple[Dict[str, Any], Dict]:
            return {"error": str(exc)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

        return self._request_or_fallback({"image_path": image_path}, _fallback)

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        return self._run(image_path)


class RemoteChestXRaySegmentationTool(_RemoteToolBase):
    name: str = "chest_xray_segmentation"
    description: str = (
        "Segments chest X-ray images to specified anatomical structures. "
        "Available organs: Left/Right Clavicle (collar bones), Left/Right Scapula (shoulder blades), "
        "Left/Right Lung, Left/Right Hilus Pulmonis (lung roots), Heart, Aorta, "
        "Facies Diaphragmatica (diaphragm), Mediastinum (central cavity), Weasand (esophagus), "
        "and Spine. Returns segmentation visualization and comprehensive metrics. "
        "Let the user know the area is not accurate unless input has been DICOM."
    )
    args_schema: Type[BaseModel] = ChestXRaySegmentationInput
    endpoint: str = "/tools/chest_xray_segmentation"

    def __init__(
        self,
        device: Optional[str] = "cuda",
        temp_dir: Optional[str] = "temp",
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(api_base_url=api_base_url, api_timeout=api_timeout, **kwargs)

    def _run(
        self,
        image_path: str,
        organs: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        def _fallback(exc: Exception) -> Tuple[Dict[str, Any], Dict]:
            message = str(exc)
            return {"error": message}, {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_traceback": message,
            }

        return self._request_or_fallback(
            {"image_path": image_path, "organs": organs},
            _fallback,
        )

    async def _arun(
        self,
        image_path: str,
        organs: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        return self._run(image_path, organs)


class RemoteXRayPhraseGroundingTool(_RemoteToolBase):
    name: str = "xray_phrase_grounding"
    description: str = (
        "Locates and visualizes specific medical findings in chest X-ray images. "
        "Takes a chest X-ray image and medical phrase to locate (e.g., 'Pleural effusion', 'Cardiomegaly'). "
        "Returns bounding box coordinates in format [x_topleft, y_topleft, x_bottomright, y_bottomright] "
        "where each value is between 0-1 representing relative position in the image, "
        "a visualization of the finding's location, and confidence metadata. "
        "Example input: {'image_path': '/path/to/xray.png', 'phrase': 'Pleural effusion', 'max_new_tokens': 300}"
    )
    args_schema: Type[BaseModel] = XRayPhraseGroundingInput
    endpoint: str = "/tools/xray_phrase_grounding"

    def __init__(
        self,
        model_path: str = "microsoft/maira-2",
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: Optional[str] = "cuda",
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(api_base_url=api_base_url, api_timeout=api_timeout, **kwargs)

    def _run(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        def _fallback(exc: Exception) -> Tuple[Dict[str, Any], Dict]:
            message = str(exc)
            return {"error": message}, {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_details": message,
            }

        return self._request_or_fallback(
            {
                "image_path": image_path,
                "phrase": phrase,
                "max_new_tokens": max_new_tokens,
            },
            _fallback,
        )

    async def _arun(
        self,
        image_path: str,
        phrase: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        return self._run(image_path, phrase)


class RemoteXRayVQATool(_RemoteToolBase):
    name: str = "chest_xray_expert"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform multiple tasks including: visual question answering, report generation, "
        "abnormality detection, comparative analysis, anatomical description, "
        "and clinical interpretation. Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True
    endpoint: str = "/tools/chest_xray_expert"

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: Any = None,
        cache_dir: Optional[str] = None,
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_base_url=api_base_url, api_timeout=api_timeout, **kwargs)

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        def _fallback(exc: Exception) -> Tuple[Dict[str, Any], Dict]:
            message = str(exc)
            return {"error": message}, {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "error_details": message,
            }

        return self._request_or_fallback(
            {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
            _fallback,
        )

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        return self._run(image_paths, prompt, max_new_tokens)


class RemoteLlavaMedTool(_RemoteToolBase):
    name: str = "llava_med_qa"
    description: str = (
        "A tool that answers questions about biomedical images and general medical questions using LLaVA-Med. "
        "While it can process chest X-rays, it may not be as reliable for detailed chest X-ray analysis. "
        "Input should be a question and optionally a path to a medical image file."
    )
    args_schema: Type[BaseModel] = LlavaMedInput
    endpoint: str = "/tools/llava_med_qa"

    def __init__(
        self,
        model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
        cache_dir: str = "/model-weights",
        low_cpu_mem_usage: bool = True,
        torch_dtype: Any = None,
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(api_base_url=api_base_url, api_timeout=api_timeout, **kwargs)

    def _run(
        self,
        question: str,
        image_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        def _fallback(exc: Exception) -> Tuple[str, Dict]:
            message = str(exc)
            return f"Error generating answer: {message}", {
                "question": question,
                "image_path": image_path,
                "analysis_status": "failed",
            }

        return self._request_or_fallback(
            {
                "question": question,
                "image_path": image_path,
            },
            _fallback,
        )

    async def _arun(
        self,
        question: str,
        image_path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        return self._run(question, image_path)


class LlavaMedImageEmbeddingInput(BaseModel):
    image_path: str
    embedding_model: str = "llava-med-clip-patch-mean-1024"


class RemoteLlavaMedImageEmbeddingTool(_RemoteToolBase):
    name: str = "llava_med_image_embedding"
    description: str = (
        "Returns a LLaVA-Med vision-tower image embedding for a chest X-ray image. "
        "The embedding is a 1024-dimensional mean-pooled CLIP patch feature."
    )
    args_schema: Type[BaseModel] = LlavaMedImageEmbeddingInput
    endpoint: str = "/tools/llava_med_image_embedding"

    def __init__(
        self,
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(api_base_url=api_base_url, api_timeout=api_timeout, **kwargs)

    def _run(
        self,
        image_path: str,
        embedding_model: str = "llava-med-clip-patch-mean-1024",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        def _fallback(exc: Exception) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            message = str(exc)
            return {"error": message}, {
                "image_path": image_path,
                "embedding_model": embedding_model,
                "analysis_status": "failed",
                "error": message,
            }

        return self._request_or_fallback(
            {"image_path": image_path, "embedding_model": embedding_model},
            _fallback,
        )

    async def _arun(
        self,
        image_path: str,
        embedding_model: str = "llava-med-clip-patch-mean-1024",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._run(image_path, embedding_model)


class RemoteLlavaMedImageEmbeddingClient:
    def __init__(
        self,
        api_base_url: str = "http://127.0.0.1:8010",
        api_timeout: float = 300.0,
    ) -> None:
        self.tool = RemoteLlavaMedImageEmbeddingTool(
            api_base_url=api_base_url,
            api_timeout=api_timeout,
        )

    def embed_image(
        self,
        image_path: str,
        embedding_model: str = "llava-med-clip-patch-mean-1024",
    ) -> Dict[str, Any]:
        payload, metadata = self.tool._invoke_remote(
            {
                "image_path": image_path,
                "embedding_model": embedding_model,
            }
        )
        if not isinstance(payload, dict):
            raise ValueError("Embedding endpoint returned a non-dict payload")
        if metadata:
            payload.setdefault("metadata", metadata)
        return payload


def get_remote_tool_factories(api_base_url: str, api_timeout: float):
    return {
        "ChestXRayReportGeneratorTool": lambda: RemoteChestXRayReportGeneratorTool(
            api_base_url=api_base_url,
            api_timeout=api_timeout,
        ),
        "ChestXRayClassifierTool": lambda: RemoteChestXRayClassifierTool(
            api_base_url=api_base_url,
            api_timeout=api_timeout,
        ),
        "ChestXRaySegmentationTool": lambda: RemoteChestXRaySegmentationTool(
            api_base_url=api_base_url,
            api_timeout=api_timeout,
        ),
        "XRayPhraseGroundingTool": lambda: RemoteXRayPhraseGroundingTool(
            api_base_url=api_base_url,
            api_timeout=api_timeout,
        ),
        "XRayVQATool": lambda: RemoteXRayVQATool(
            api_base_url=api_base_url,
            api_timeout=api_timeout,
        ),
        "LlavaMedTool": lambda: RemoteLlavaMedTool(
            api_base_url=api_base_url,
            api_timeout=api_timeout,
        ),
    }


def get_remote_image_embedding_client(
    api_base_url: str,
    api_timeout: float,
) -> RemoteLlavaMedImageEmbeddingClient:
    return RemoteLlavaMedImageEmbeddingClient(
        api_base_url=api_base_url,
        api_timeout=api_timeout,
    )
