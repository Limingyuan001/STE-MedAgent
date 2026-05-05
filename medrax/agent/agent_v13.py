import re
from typing import Any, Dict, List, Optional

from .agent_v11 import AgentStateV11, AgentV11, DEFAULT_PROCESS_POLICY


ALLOWED_QUESTION_TYPES = [
    "Abnormality-Check",
    "Normality",
    "Localization-Side",
    "Localization-Lobe",
    "Severity",
    "Modality-Study",
    "System-Abnormality",
]

ALLOWED_PATHOLOGICAL_FINDINGS = [
    "Pulmonary Atelectasis",
    "Pleural Effusion",
    "Cardiomegaly",
    "Elevated Diaphragm",
    "Indwelling Catheter",
    "Surgical Instruments",
    "Hyperdistention",
    "Hypoinflation",
    "Lung Opacity",
    "Pneumonia",
    "Pneumothorax",
    "Mass",
    "Nodule",
    "Thoracic Scoliosis",
    "Vertebral Degeneration",
    "Thoracic Aortic Tortuosity",
    "Bronchovascular Markings",
    "Calcinosis",
    "Infiltration",
    "Skeletal Deformity",
]
ALLOWED_PATHOLOGICAL_FINDINGS_SET = set(ALLOWED_PATHOLOGICAL_FINDINGS)

PATHOLOGICAL_FINDING_ALIAS_TO_CANONICAL = {
    "atelectasis": "Pulmonary Atelectasis",
    "pulmonary atelectasis": "Pulmonary Atelectasis",
    "base pulmonary atelectasis": "Pulmonary Atelectasis",
    "lobe pulmonary atelectasis": "Pulmonary Atelectasis",
    "pleural effusion": "Pleural Effusion",
    "effusion": "Pleural Effusion",
    "small pleural effusion": "Pleural Effusion",
    "large pleural effusion": "Pleural Effusion",
    "base pleural effusion": "Pleural Effusion",
    "cardiomegaly": "Cardiomegaly",
    "mild cardiomegaly": "Cardiomegaly",
    "severe cardiomegaly": "Cardiomegaly",
    "elevated diaphragm": "Elevated Diaphragm",
    "elevated right diaphragm": "Elevated Diaphragm",
    "elevated left diaphragm": "Elevated Diaphragm",
    "indwelling catheter": "Indwelling Catheter",
    "thoracic indwelling catheter": "Indwelling Catheter",
    "right indwelling catheter": "Indwelling Catheter",
    "left indwelling catheter": "Indwelling Catheter",
    "surgical instruments": "Surgical Instruments",
    "surgical instruments in the lung": "Surgical Instruments",
    "surgical instruments in the abdomen": "Surgical Instruments",
    "surgical instruments in the breast": "Surgical Instruments",
    "hyperdistention": "Hyperdistention",
    "hyperdistention of the lung": "Hyperdistention",
    "hypoinflated lung": "Hypoinflation",
    "hypoinflation": "Hypoinflation",
    "lung opacity": "Lung Opacity",
    "opacity": "Lung Opacity",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "mass": "Mass",
    "nodule": "Nodule",
    "thoracic vertebrae scoliosis": "Thoracic Scoliosis",
    "lumbar vertebrae scoliosis": "Thoracic Scoliosis",
    "scoliosis": "Thoracic Scoliosis",
    "degeneration of thoracic vertebrae": "Vertebral Degeneration",
    "vertebral degeneration": "Vertebral Degeneration",
    "tortuous thoracic aorta": "Thoracic Aortic Tortuosity",
    "thoracic aortic tortuosity": "Thoracic Aortic Tortuosity",
    "aortic tortuosity": "Thoracic Aortic Tortuosity",
    "bronchovascular markings": "Bronchovascular Markings",
    "pulmonary infiltration": "Infiltration",
    "pleural infiltration": "Infiltration",
    "infiltration": "Infiltration",
    "calcinosis": "Calcinosis",
    "abdominal calcinosis": "Calcinosis",
    "aortic calcinosis": "Calcinosis",
    "thoracic aorta calcinosis": "Calcinosis",
    "clavicle deformity": "Skeletal Deformity",
    "rib deformity": "Skeletal Deformity",
    "humerus deformity": "Skeletal Deformity",
    "skeletal deformity": "Skeletal Deformity",
}

DEFAULT_EXTRACT_PROMPT = """You are a medical tagging assistant for CheXBench chest imaging MCQ questions.
Extract tags from the question stem and options as JSON with this exact schema:
{
  "question_type": ["..."],
  "symptoms": ["..."],
  "demographics": ["..."],
  "risk_factors": ["..."],
  "pathological_findings": ["..."],
  "denied_information": {
    "symptoms": ["..."],
    "pathological_findings": ["..."],
    "risk_factors": ["..."]
  }
}

question_type must be a subset of exactly these 7 labels (use these exact names):
- Abnormality-Check: yes/no questions asking whether any abnormality or a named disease/finding is present.
- Normality: questions asking whether the image or organ is normal/healthy.
- Localization-Side: side-specific questions such as right vs left.
- Localization-Lobe: lobe-specific questions such as upper vs lower lobe.
- Severity: questions contrasting mild vs severe or similar severity grades.
- Modality-Study: questions about modality, plane, or study type (e.g. X-Ray, chest study).
- System-Abnormality: questions about whether a broader anatomic system/region is abnormal, such as lung, pleura, skeletal system, mediastinum, or cardiovascular system.

Extraction rules:
1) Prefer CheXBench-native labels:
   - Fine-Grained Reasoning questions usually map to Localization-Side, Localization-Lobe, or Severity.
   - Visual Question Answering yes/no abnormality questions usually map to Abnormality-Check, Normality, or System-Abnormality.
   - Modality/study/plane questions map to Modality-Study.
2) pathological_findings must be selected only from this closed set:
   ["Pulmonary Atelectasis","Pleural Effusion","Cardiomegaly","Elevated Diaphragm","Indwelling Catheter","Surgical Instruments","Hyperdistention","Hypoinflation","Lung Opacity","Pneumonia","Pneumothorax","Mass","Nodule","Thoracic Scoliosis","Vertebral Degeneration","Thoracic Aortic Tortuosity","Bronchovascular Markings","Calcinosis","Infiltration","Skeletal Deformity"].
3) Extract pathological_findings from the question stem and options only. No image inference.
4) If a synonym or variant appears, map it to the closest canonical label from the closed set above.
5) Do not output non-pathology concepts such as "normal", "healthy", "X-Ray", "chest study", anatomic systems, or yes/no answers as pathological_findings.
6) Symptoms are subjective, patient-reported experiences such as pain, nausea, fatigue, dizziness, headache, cough, fever, chest pain, or dyspnea.
7) Demographics are human descriptors such as age, sex/gender, pregnancy status, ethnicity, occupation, education, or income, only when explicitly stated.
8) Risk factors are characteristics, conditions, exposures, or behaviors that increase disease risk, such as smoking, TB exposure, or immunosuppression.
9) denied_information must contain only explicitly negated items from the question stem for symptoms, pathological_findings, and risk_factors. Never infer negation.
10) denied_information.pathological_findings must also use the same canonical closed set.
11) Keep all lists concise and deduplicated. If no evidence is present, return an empty list.

Return JSON only. No markdown.
"""


class AgentV13(AgentV11):
    def __init__(
        self,
        model,
        tools,
        *,
        checkpointer: Any = None,
        system_prompt: str = "",
        extract_prompt: str = DEFAULT_EXTRACT_PROMPT,
        process_policy_prompt: str = DEFAULT_PROCESS_POLICY,
        tag_embedder=None,
        memory_retriever=None,
        retrieval_top_k: int = 3,
        similarity_threshold: float = 0.3,
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        max_tool_output_chars: int = 2500,
        max_reasoning_steps: int = 7,
    ):
        super().__init__(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
            extract_prompt=extract_prompt,
            process_policy_prompt=process_policy_prompt,
            tag_embedder=tag_embedder,
            memory_retriever=memory_retriever,
            retrieval_top_k=retrieval_top_k,
            similarity_threshold=similarity_threshold,
            log_tools=log_tools,
            log_dir=log_dir,
            max_tool_output_chars=max_tool_output_chars,
            max_reasoning_steps=max_reasoning_steps,
        )

    @staticmethod
    def _canonicalize_pathological_finding(term: str) -> Optional[str]:
        cleaned = re.sub(r"\s+", " ", re.sub(r"[_/\-]+", " ", str(term).strip().lower()))
        if not cleaned:
            return None

        direct = PATHOLOGICAL_FINDING_ALIAS_TO_CANONICAL.get(cleaned)
        if direct and direct in ALLOWED_PATHOLOGICAL_FINDINGS_SET:
            return direct

        alias_items = sorted(
            PATHOLOGICAL_FINDING_ALIAS_TO_CANONICAL.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        for alias, canonical in alias_items:
            if canonical not in ALLOWED_PATHOLOGICAL_FINDINGS_SET:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", cleaned):
                return canonical
        return None

    def _normalize_tags(self, data: Any, question_stem: str = "") -> Dict[str, Any]:
        normalized: Dict[str, Any] = {
            "question_type": [],
            "symptoms": [],
            "demographics": [],
            "risk_factors": [],
            "pathological_findings": [],
            "denied_information": {
                "symptoms": [],
                "pathological_findings": [],
                "risk_factors": [],
            },
        }
        if not isinstance(data, dict):
            return normalized

        allowed_type_map = {x.lower(): x for x in ALLOWED_QUESTION_TYPES}
        question_type = data.get("question_type", [])
        if isinstance(question_type, str):
            question_type = [question_type]
        if isinstance(question_type, list):
            cleaned_qt = []
            for item in question_type:
                text = str(item).strip()
                if not text:
                    continue
                mapped = allowed_type_map.get(text.lower())
                if mapped:
                    cleaned_qt.append(mapped)
            normalized["question_type"] = self._dedupe_keep_order(cleaned_qt, max_items=len(ALLOWED_QUESTION_TYPES))

        def _as_list(value: Any) -> List[str]:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return []
            cleaned_values = []
            for item in value:
                text = str(item).strip()
                lowered = text.lower()
                if not text or lowered in {"yes", "no"}:
                    continue
                cleaned_values.append(text)
            return cleaned_values

        auto_denied_symptoms: List[str] = []
        for symptom in _as_list(data.get("symptoms", [])):
            if self._term_is_explicitly_negated(symptom, question_stem):
                auto_denied_symptoms.append(symptom)
            else:
                normalized["symptoms"].append(symptom)
        normalized["symptoms"] = self._dedupe_keep_order(normalized["symptoms"], max_items=30)

        normalized["demographics"] = self._dedupe_keep_order(
            _as_list(data.get("demographics", [])),
            max_items=20,
        )

        auto_denied_risk_factors: List[str] = []
        for risk_factor in _as_list(data.get("risk_factors", [])):
            if self._term_is_explicitly_negated(risk_factor, question_stem):
                auto_denied_risk_factors.append(risk_factor)
            else:
                normalized["risk_factors"].append(risk_factor)
        normalized["risk_factors"] = self._dedupe_keep_order(normalized["risk_factors"], max_items=20)

        findings = data.get("pathological_findings", [])
        if isinstance(findings, str):
            findings = [findings]
        if isinstance(findings, list):
            canonical_findings: List[str] = []
            auto_denied_findings: List[str] = []
            for item in findings:
                text = str(item).strip()
                if not text or text.lower() in {"yes", "no"}:
                    continue
                canonical = self._canonicalize_pathological_finding(text)
                if canonical:
                    if self._term_is_explicitly_negated(text, question_stem) or self._term_is_explicitly_negated(
                        canonical, question_stem
                    ):
                        auto_denied_findings.append(canonical)
                    else:
                        canonical_findings.append(canonical)
            normalized["pathological_findings"] = self._dedupe_keep_order(
                canonical_findings,
                max_items=len(ALLOWED_PATHOLOGICAL_FINDINGS),
            )
        else:
            auto_denied_findings = []

        denied_information = data.get("denied_information", {})
        if not isinstance(denied_information, dict):
            denied_information = {}

        denied_symptoms = []
        for item in auto_denied_symptoms + _as_list(denied_information.get("symptoms", [])):
            if self._term_is_explicitly_negated(item, question_stem):
                denied_symptoms.append(item)
        normalized["denied_information"]["symptoms"] = self._dedupe_keep_order(
            denied_symptoms,
            max_items=30,
        )

        denied_risk_factors = []
        for item in auto_denied_risk_factors + _as_list(denied_information.get("risk_factors", [])):
            if self._term_is_explicitly_negated(item, question_stem):
                denied_risk_factors.append(item)
        normalized["denied_information"]["risk_factors"] = self._dedupe_keep_order(
            denied_risk_factors,
            max_items=20,
        )

        denied_pathological_findings: List[str] = []
        denied_findings_raw = _as_list(denied_information.get("pathological_findings", []))
        for item in auto_denied_findings + denied_findings_raw:
            canonical = self._canonicalize_pathological_finding(item)
            if not canonical:
                continue
            if self._term_is_explicitly_negated(item, question_stem) or self._term_is_explicitly_negated(
                canonical, question_stem
            ):
                denied_pathological_findings.append(canonical)
        denied_pathological_findings = self._dedupe_keep_order(
            denied_pathological_findings,
            max_items=len(ALLOWED_PATHOLOGICAL_FINDINGS),
        )
        normalized["denied_information"]["pathological_findings"] = denied_pathological_findings
        if denied_pathological_findings:
            denied_set = set(denied_pathological_findings)
            normalized["pathological_findings"] = [
                finding for finding in normalized["pathological_findings"] if finding not in denied_set
            ]
        return normalized


AgentStateV13 = AgentStateV11
