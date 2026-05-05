# STE-MedAgent: Scaling Tool-Use Experience for Medical Reasoning Agent

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-coming soon-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv (coming soon)">
</p>

> **TODO**
> - [ ] Add arXiv preprint link
> - [ ] Add paper link (conference proceedings)
> - [ ] Add GitHub repository link
> - [ ] Add HuggingFace model / dataset card

---

## Overview

STE-MedAgent is a **long-term experience-guided medical reasoning framework** that equips a ReAct-style agent with continuously accumulated tool-use experience. Rather than relying solely on static tool descriptions, the agent:

1. **Retrieves** tool-reliability records from semantically similar historical cases at inference time.
2. **Injects** the retrieved reference into the reasoning context for reliability-aware tool selection.
3. **Scores** each tool event post-inference via a label-free LLM-as-a-Judge module across five criteria.
4. **Writes** the scored experience back to long-term memory, enabling self-improvement without retraining.

Extensive experiments on **ChestAgentBench**, **CheXbench**, **MIMIC-CXR**, and **SLAKE-VQA** with Qwen3-VL 2B, Qwen3-VL 32B, and GPT-5.5 demonstrate consistent accuracy gains that grow monotonically with accumulated experience.

---

## Architecture

```
New Case (Image + Question)
        │
        ▼
  Tag Extraction  ──► BGE Embedding  ──► Memory Retrieval (cosine sim, τ=0.99, k=3)
        │                                        │
        │              Tool-Reliability Reference Γ(X)
        │                                        │
        ▼                                        ▼
   LLM Core (Qwen3-VL / GPT-5.5)  ◄─────────────┘
        │  ReAct Loop (max 7 steps)
        ▼
   Tool API Server ──► [Cls | Dgn | Rpt | Seg | Vqa | Grd]
        │
        ▼
   Final Answer  ──► LLM-as-a-Judge (5 criteria)  ──► Memory Update
```

### Six Specialized Tools

| Abbr. | Task | Implementation | Input | Output |
|-------|------|----------------|-------|--------|
| **Cls** | Classification | [TorchXRayVision (XRV)](https://github.com/mlmed/torchxrayvision) | image_path | 18 pathology scores |
| **Dgn** | Diagnosis | [CheXagent](https://github.com/Stanford-AIMI/CheXagent) | image_path + prompt | Findings / diagnosis |
| **Rpt** | Report Generation | [CheXpert Plus](https://github.com/Stanford-AIMI/CheXpert-Plus) | image_path | Findings + impression |
| **Seg** | Segmentation | [PSPNet](https://github.com/lianjizhe/ChestX-Det) | image_path + organ list | Organ mask |
| **Vqa** | VQA | [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) | question + image | Free-text answer |
| **Grd** | Phrase Grounding | [Maira-2](https://github.com/microsoft/maira) | image_path + phrase | BBox coordinates |

---

## Service Architecture

The framework relies on **three independent API services**. Each runs as a separate process, typically on different ports:

### Service 1 — LLM Core (32B, default)

Serves Qwen3-VL 32B via an OpenAI-compatible endpoint.

```bash
# Example using vLLM
vllm serve qwen3-vl-32b-instruct \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768
```

- **Base URL:** `http://127.0.0.1:8000/v1`
- **Model name:** `qwen3-vl-32b-instruct`
- **Recommended hardware:** 2× NVIDIA RTX Pro 6000 (96 GB VRAM each)

### Service 2 — LLM Core (2B, optional)

Serves Qwen3-VL 2B for lightweight experiments.

```bash
vllm serve qwen3-vl-2b-instruct-fp8 \
  --port 8001 \
  --max-model-len 32768
```

- **Base URL:** `http://127.0.0.1:8001/v1`
- **Model name:** `qwen3-vl-2b-instruct-fp8`

### Service 3 — Tool API Server

Hosts all six specialized neural tools behind a single REST endpoint.

```bash
python tool_server.py
# For MIMIC-CXR (BioViL-T image embedding):
python tool_server_mimic.py
```

- **Base URL:** `http://127.0.0.1:8010`
- Configure model weight paths in `.env` before starting (see **Configuration** below).

---

## Installation

### Requirements

- Python 3.11.14
- PyTorch 2.9.0+cu130 (CUDA 13.0)
- 2× GPU with ≥ 48 GB VRAM recommended (experiments used NVIDIA RTX Pro 6000 96 GB × 2)

### Install

```bash
git clone https://github.com/Limingyuan001/STE-MedAgent.git
cd STE-MedAgent
pip install -e .
```

### Environment Checklist

| Package | Purpose |
|---------|---------|
| `torch` ≥ 2.1 | Deep learning backend |
| `transformers` | CheXagent, LLaVA-Med, Maira-2 |
| `torchxrayvision` | Cls tool (XRV / DenseNet-121) |
| `sentence-transformers` | BGE-small-en-v1.5 embedding |
| `openai` ≥ 1.0 | LLM core API client |
| `langchain`, `langgraph` | ReAct agent loop |
| `fastapi`, `uvicorn` | Tool API server |
| `Pillow`, `pydicom` | Image / DICOM I/O |
| `numpy`, `scipy` | Numerical utilities |
| `tqdm` | Progress bars |

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```dotenv
# LLM core — 32B local (default)
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=qwen3-vl-32b-instruct
LLM_API_KEY=your_key_here

# LLM core — 2B local (optional)
LLM_2B_BASE_URL=http://127.0.0.1:8001/v1
LLM_2B_MODEL=qwen3-vl-2b-instruct-fp8
LLM_2B_API_KEY=your_key_here

# GPT-5.5 (optional)
OPENAI_API_KEY=your_openai_key_here

# Tool server
TOOL_API_BASE_URL=http://127.0.0.1:8010
```

---

## Datasets

Download the benchmark datasets and place them at the paths referenced in the experiment scripts:

| Dataset | Task | Download |
|---------|------|----------|
| **ChestAgentBench** | 2,500 six-choice CXR reasoning questions (7 categories) | [HuggingFace](https://huggingface.co/datasets/wanglab/chest-agent-bench) |
| **CheXbench** | Image-text VQA + fine-grained reasoning (618 questions) | [GitHub](https://github.com/Stanford-AIMI/CheXbench) |
| **MIMIC-CXR** | Radiology report generation (3,858 test images) | [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) |
| **SLAKE-VQA** | Medical VQA — 114 closed-ended English CXR samples | [Project page](https://www.med-vqa.com/slake/) · [GitHub](https://github.com/med-vqa/SLAKE) |

After downloading:

```
# ChestAgentBench — metadata already included at:
chestagentbench/metadata.jsonl

# CheXbench — point --data-path to your local copy:
/path/to/chexbench/chexbench_data.json

# MIMIC-CXR — follow M4CXR preprocessing:
experiments/M4CXR/MIMIC_CXR/run_prepare_m4cxr_mimic_subset.sh

# SLAKE-VQA — filtered subset included at:
experiments/M4CXR/SLAKE_VQA/slake_test_en_closed_xray_114.json
```

---

## Running Experiments

### ChestAgentBench — Test-time (ρ = 1.0)

```bash
bash experiments/run_v11v2_api_memory_93v10_API.sh
```

### ChestAgentBench — Pre-constructed 5-fold Cross-Validation

Pre-built memory files (`memory_84_1.jsonl` … `memory_84_5.jsonl`) are **not included** in the repository (large files). To reproduce, first run the test-time setting to accumulate memory, then use:

```bash
bash experiments/run_chestagentbench_preconstruct_5fold.sh
```

### ChestAgentBench — GPT-5.5

```bash
bash experiments/run_v11v2_api_memory_180_GPT-5.5.sh
```

### MIMIC-CXR (concurrent)

```bash
bash experiments/run_v11v2_api_MIMIC_MRG_134_all_32B.sh
```

### SLAKE-VQA

```bash
bash experiments/run_v12_api_SLAKE_VQA_32B.sh
```

### CheXbench

```bash
bash experiments/run_chexbenchv13_115-116.sh
```

---

## Memory

The long-term memory is a JSONL key-value store:

```
M = { (z_n, U_n) }
```

where `z_n ∈ ℝ^384` is a BGE-small-en-v1.5 embedding of structured semantic tags, and `U_n` is a set of scored tool events `(tool, args, score ∈ [−1, 1])`.

- **Retrieval:** cosine similarity, default `τ = 0.99`, `k = 3`
- **Scoring:** LLM-as-a-Judge across 5 criteria (clinical relevance, informational increment, diagnostic discriminability, citation fidelity, answer consistency)
- Memory files are written to `experiments/memory/` and are excluded from version control (`.gitignore`).

---

## Acknowledgements

This project builds directly on **[MedRAX](https://github.com/bowang-lab/MedRAX)**, which provides the tool library, agent loop, and ChestAgentBench benchmark that underpin our framework. We are grateful to the MedRAX authors for open-sourcing their work.

```bibtex
@article{fallahpour2025medrax,
  title={Medrax: Medical reasoning agent for chest x-ray},
  author={Fallahpour, Adibvafa and Ma, Jun and Munim, Alif and Lyu, Hongwei and Wang, Bo},
  journal={arXiv preprint arXiv:2502.02673},
  year={2025}
}
```
