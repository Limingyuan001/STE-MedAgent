import logging
import os
import time
from threading import Lock
from typing import Any, Dict, Tuple

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("TOOL_SERVER_CUDA_VISIBLE_DEVICES", "0")

from fastapi import FastAPI, HTTPException
import uvicorn

from medrax.tools import (
    ChestXRayClassifierTool,
    ChestXRayReportGeneratorTool,
    ChestXRaySegmentationTool,
    LlavaMedTool,
    XRayPhraseGroundingTool,
    XRayVQATool,
)
from medrax.tools.classification import ChestXRayInput as ClassificationInput
from medrax.tools.grounding import XRayPhraseGroundingInput
from medrax.tools.llava_med import LlavaMedInput
from medrax.tools.remote import encode_for_transport
from medrax.tools.report_generation import ChestXRayInput as ReportGenerationInput
from medrax.tools.segmentation import ChestXRaySegmentationInput
from medrax.tools.xray_vqa import XRayVQAToolInput


ROOT = os.getenv("MEDRAX_ROOT", os.path.abspath(os.path.dirname(__file__)))
MODEL_DIR = os.getenv("TOOL_SERVER_MODEL_DIR", f"{ROOT}/benchmark/tool_models")
TEMP_DIR = os.getenv("TOOL_SERVER_TEMP_DIR", "temp")
TOOL_DEVICE = os.getenv("TOOL_SERVER_DEVICE", "cuda")
HOST = os.getenv("TOOL_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("TOOL_SERVER_PORT", "8010"))
LOG_LEVEL = os.getenv("TOOL_SERVER_LOG_LEVEL", "info")


app = FastAPI(title="MedRAX Neural Tool Server")
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
        "chest_xray_report_generator": ChestXRayReportGeneratorTool(
            cache_dir=MODEL_DIR,
            device=TOOL_DEVICE,
        ),
        "chest_xray_classifier": ChestXRayClassifierTool(device=TOOL_DEVICE),
        "chest_xray_segmentation": ChestXRaySegmentationTool(device=TOOL_DEVICE),
        "xray_phrase_grounding": XRayPhraseGroundingTool(
            cache_dir=MODEL_DIR,
            temp_dir=TEMP_DIR,
            device=TOOL_DEVICE,
            load_in_8bit=True,
        ),
        "chest_xray_expert": XRayVQATool(
            cache_dir=MODEL_DIR,
            device=TOOL_DEVICE,
        ),
        "llava_med_qa": LlavaMedTool(
            cache_dir=MODEL_DIR,
            device=TOOL_DEVICE,
            load_in_8bit=True,
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


@app.post("/tools/chest_xray_report_generator")
def chest_xray_report_generator(payload: ReportGenerationInput) -> Dict[str, Any]:
    return run_tool("chest_xray_report_generator", payload.dict())


@app.post("/tools/chest_xray_classifier")
def chest_xray_classifier(payload: ClassificationInput) -> Dict[str, Any]:
    return run_tool("chest_xray_classifier", payload.dict())


@app.post("/tools/chest_xray_segmentation")
def chest_xray_segmentation(payload: ChestXRaySegmentationInput) -> Dict[str, Any]:
    return run_tool("chest_xray_segmentation", payload.dict())


@app.post("/tools/xray_phrase_grounding")
def xray_phrase_grounding(payload: XRayPhraseGroundingInput) -> Dict[str, Any]:
    return run_tool("xray_phrase_grounding", payload.dict())


@app.post("/tools/chest_xray_expert")
def chest_xray_expert(payload: XRayVQAToolInput) -> Dict[str, Any]:
    return run_tool("chest_xray_expert", payload.dict())


@app.post("/tools/llava_med_qa")
def llava_med_qa(payload: LlavaMedInput) -> Dict[str, Any]:
    return run_tool("llava_med_qa", payload.dict())


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)


if __name__ == "__main__":
    main()
