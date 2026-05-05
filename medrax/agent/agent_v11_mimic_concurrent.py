import json
import os

from medrax.runtime_context import get_sample_context

from .agent_v11_mimic import AgentV11Mimic


class AgentV11MimicConcurrent(AgentV11Mimic):
    def _resolve_image_path(self, image_path: str) -> str:
        if not image_path:
            return image_path
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        sample_context = get_sample_context()
        image_paths = sample_context.get("image_paths", [])
        if isinstance(image_paths, list):
            for img_path in image_paths:
                if not isinstance(img_path, str) or not os.path.exists(img_path):
                    continue
                if os.path.basename(img_path).lower() in image_path.lower():
                    return img_path
            if image_paths and isinstance(image_paths[0], str) and os.path.exists(image_paths[0]):
                return image_paths[0]

        image_paths_json = os.getenv("MEDRAX_IMAGE_PATHS")
        if image_paths_json:
            try:
                legacy_paths = json.loads(image_paths_json)
            except Exception:
                legacy_paths = None
            if isinstance(legacy_paths, list):
                for img_path in legacy_paths:
                    if not isinstance(img_path, str) or not os.path.exists(img_path):
                        continue
                    if os.path.basename(img_path).lower() in image_path.lower():
                        return img_path
                if legacy_paths and isinstance(legacy_paths[0], str) and os.path.exists(legacy_paths[0]):
                    return legacy_paths[0]

        figures_dir = str(sample_context.get("figures_dir", "") or os.getenv("MEDRAX_FIGURES_DIR", ""))
        case_id = str(sample_context.get("case_id", "") or os.getenv("MEDRAX_CASE_ID", ""))
        if figures_dir and case_id:
            candidate = os.path.join(figures_dir, case_id, image_path)
            if os.path.exists(candidate):
                return candidate
        return image_path
