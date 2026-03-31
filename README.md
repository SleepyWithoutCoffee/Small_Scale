# LLM Evaluation & Training Toolkit
[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b)](https://openreview.net/pdf?id=8xSU8Oscvg)

This is the official repository for the paper 'Pruning Long Chain-of-Thought in Large Reasoning Models via Small-Scale Preference Optimization' (ICLR'2026).

We provide a toolkit for offline inference evaluation and training of large language models, supporting vLLM / SGLang inference backends, multiple math/code/multiple-choice evaluation benchmarks, and DPO training based on LLaMA-Factory.

## Directory Structure

```
prune/
├── config/                          # Configuration files
│   ├── path.yaml                    # ★ Global path configuration (must be filled in)
│   ├── dataset/
│   │   ├── dataset_info.json        # Dataset metadata (dataset → task type mapping)
│   │   └── task_info.json           # Task prompt templates
│   ├── eval/
│   │   ├── vllm_offline.yaml        # vLLM inference config (with data-parallel shuffle)
│   │   ├── vllm_offline_bc.yaml     # vLLM inference config (basic version)
│   │   ├── sglang_offline.yaml      # SGLang inference config
│   │   ├── eval_protocol.yaml       # Evaluation protocol & LLM-as-Judge config
│   │   └── extra.yaml               # Extra evaluation parameters
│   └── train/
│       ├── deepspeed_z3.json        # DeepSpeed ZeRO-3 config
│       └── llama_factory/
│           └── dpo.yaml             # LLaMA-Factory DPO training config
├── data/test/                       # Test datasets (parquet format)
│   ├── math/                        # Math: math, math-500, aime24, aime25, amc23, gsm8k, ...
│   ├── code/                        # Code: lcb (LiveCodeBench)
│   └── mcq/                         # Multiple-choice: mmlu, gpqa-d, arc-c, winogrande
├── eval/
│   ├── generation/                  # Inference generation scripts
│   │   ├── vllm_offline.py          # vLLM multi-process data-parallel inference
│   │   ├── vllm_offline_random_dp.py# vLLM multi-process inference (with random shuffle)
│   │   ├── vllm_offline_naive.py    # vLLM single-process inference
│   │   ├── sglang_offline.py        # SGLang inference
│   │   └── sglang_logps.py          # SGLang log-probability computation
│   ├── judgers/                     # Evaluation judgers
│   │   ├── autojudger.py            # Auto-evaluation entry point
│   │   ├── math_judger.py           # Math task judger
│   │   ├── code_judger.py           # Code task judger
│   │   ├── mcq_judger.py            # Multiple-choice task judger
│   │   ├── base_judger.py           # Base judger class
│   │   └── compute_score.py         # Score computation
│   └── utils/                       # Evaluation utilities
│       ├── math/utils.py            # Math answer extraction & normalization
│       ├── lcb/                     # LiveCodeBench code evaluation utilities
│       └── llm_as_judge/            # LLM-as-Judge utilities
│           ├── api_router.py        # API router
│           └── api_utils/
│               └── openai_utils.py  # OpenAI API call utilities
├── example/                         # Example scripts
│   ├── test.sh                      # Inference evaluation example
│   ├── test_llm_as_judge.sh         # LLM-as-Judge evaluation example
│   └── train.sh                     # Training environment setup example
└── LLaMA-Factory/                   # LLaMA-Factory training framework (submodule)
```

## Prerequisites

### 1. Path Variables to Fill In

Before use, you must replace all `/path/to/` prefixes with your actual paths. The relevant files are as follows:

#### `config/path.yaml` (Core Path Configuration)

| Variable | Description | Example |
|----------|-------------|---------|
| `data_dir` | Root directory for test data | `/home/user/prune/data` |
| `model_dir` | Model storage directory | `/home/user/models` |
| `output_dir.models` | Training output model directory | `/home/user/output/model` |
| `output_dir.data` | Inference result output directory | `/home/user/output/data` |
| `output_dir.results` | Evaluation result summary directory | `/home/user/output/results` |
| `log_dir` | Root log directory | `/home/user/output/logs` |

#### `config/eval/eval_protocol.yaml` (Evaluation Protocol)

| Variable | Description |
|----------|-------------|
| `llm_as_judge.judge_model.api_key` | OpenAI API Key (required when LLM-as-Judge is enabled) |
| `llm_as_judge.judge_model.api_url` | OpenAI API URL |
| `result_log_dir` | Evaluation result log directory |

#### `config/eval/vllm_offline.yaml` / `sglang_offline.yaml` / `vllm_offline_bc.yaml`

| Variable | Description |
|----------|-------------|
| `generation_log_dir` | Inference log output directory |

#### `config/train/llama_factory/dpo.yaml` (Training Configuration)

| Variable | Description |
|----------|-------------|
| `model_name_or_path` | Base model path |
| `deepspeed` | DeepSpeed config file path |
| `output_dir` | Training output directory |

### 2. Environment Dependencies

- Python 3.10+
- [vLLM](https://github.com/vllm-project/vllm) (when using the vLLM inference backend)
- [SGLang](https://github.com/sgl-project/sglang) (when using the SGLang inference backend)
- `transformers`, `pandas`, `jinja2`, `tqdm`, `requests`
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (when using DPO training)

### 3. Model Preparation

Place model weights in the directory specified by `model_dir` in `config/path.yaml`. The directory name serves as the `model_name`. For example:

```
/home/user/models/
├── Qwen3-4B-Instruct-2507/
├── DeepSeek-R1-Distill-Qwen-1.5B/
└── ...
```

### 4. Data Preparation

Test data is built-in under `data/test/`, stored in parquet format. Supported datasets:

| Type | Dataset | Default Metric |
|------|---------|----------------|
| Math | math, math-500, gsm8k, aime24, aime25, amc23, olympiadbench, minervamath, limr, hle-math | acc / avg@16 |
| Code | lcb (LiveCodeBench) | pass@1 |
| Multiple-choice | mmlu, gpqa-d, arc-c, winogrande | acc |

## Usage Guide

### Inference Evaluation (Generation + Evaluation)

**One-step inference**: Use the inference script to generate outputs for the specified model and dataset. Results are automatically saved as parquet files.

```bash
python eval/generation/vllm_offline.py \
    --config config/eval/vllm_offline.yaml \
    --model_name <model_name> \
    --dataset_name <dataset_name>
```

**Parameter Description**:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--config` | Yes | Inference config file path |
| `--model_name` | Yes | Model name (corresponds to the directory name under `model_dir`) |
| `--dataset_name` | Yes | Dataset name (see the supported dataset list above) |

**Example** — Run inference with Qwen3-4B on AIME25:

```bash
python eval/generation/vllm_offline.py \
    --config config/eval/vllm_offline.yaml \
    --model_name Qwen3-4B-Instruct-2507 \
    --dataset_name aime25
```

**Available Inference Backends**:

| Script | Config File | Description |
|--------|-------------|-------------|
| `eval/generation/vllm_offline.py` | `config/eval/vllm_offline.yaml` | vLLM multi-process DP inference (recommended) |
| `eval/generation/vllm_offline_random_dp.py` | `config/eval/vllm_offline.yaml` | vLLM multi-process DP + random shuffle |
| `eval/generation/vllm_offline_naive.py` | `config/eval/vllm_offline_bc.yaml` | vLLM single-process inference |
| `eval/generation/sglang_offline.py` | `config/eval/sglang_offline.yaml` | SGLang inference |

### Judging

After inference is complete, automatically judge the generated results:

```bash
python eval/judgers/autojudger.py \
    --config config/eval/eval_protocol.yaml \
    --file_path <inference_result_parquet_file_path>
```

**Parameter Description**:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--config` | Yes | Evaluation protocol config file |
| `--file_path` | Yes | Path to the `.parquet` result file generated during the inference phase |

**Example** — Judge the AIME25 inference results:

```bash
python eval/judgers/autojudger.py \
    --config config/eval/eval_protocol.yaml \
    --file_path output/data/math/aime25/Qwen3-4B-Instruct-2507/03231558_1.parquet
```

The judger automatically selects the corresponding judger based on the dataset's task type (math / code / mcq), outputs the score, average/max/min generation length, and appends the results to the log file under `result_log_dir`.

### End-to-End Inference + Evaluation Pipeline

After the inference script finishes, it writes the output file path to `eval/temp/gen_id.txt`. You can use this mechanism to chain inference and evaluation:

```bash
# Step 1: Inference
python eval/generation/vllm_offline.py \
    --config config/eval/vllm_offline.yaml \
    --model_name Qwen3-4B-Instruct-2507 \
    --dataset_name aime25

# Step 2: Read the output path and run evaluation
python eval/judgers/autojudger.py \
    --config config/eval/eval_protocol.yaml \
    --file_path $(cat eval/temp/gen_id.txt)
```

### DPO Training

Use LLaMA-Factory for DPO training. First, modify the paths and training hyperparameters in `config/train/llama_factory/dpo.yaml`, then set up the environment and start training:

```bash
# Set up the environment (adjust CUDA_VISIBLE_DEVICES according to your environment)
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Start training with LLaMA-Factory (refer to the LLaMA-Factory documentation for specific commands)
llamafactory-cli train config/train/llama_factory/dpo.yaml
```

**Training Parameters in `dpo.yaml`**:

| Parameter | Description |
|-----------|-------------|
| `model_name_or_path` | Base model path |
| `pref_loss` | Preference learning loss function |
| `pref_beta` | Bradley-Terry loss weight |
| `learning_rate` | Learning rate |
| `lr_scheduler_type` | Learning rate scheduler type |
| `warmup_ratio` | warmup_ratio |
| `num_train_epochs` | Number of training epochs |
| `cutoff_len` | Maximum sequence truncation length |
| `template` | Conversation template |

## Inference Parameter Configuration

All inference config files share the following sampling parameter structure:

```yaml
sampling_params:
  temperature: 0.6
  top_p: 0.95
  max_tokens: 32768       # vLLM uses max_tokens, SGLang uses max_new_tokens
```

Other adjustable parameters (modify in the corresponding yaml file):

| Parameter | Description |
|-----------|-------------|
| `engine_args.tensor_parallel_size` | Tensor parallel size (vLLM) |
| `engine_args.gpu_memory_utilization` | GPU memory utilization (vLLM) |
| `engine_args.tp_size` / `dp_size` | Tensor/data parallel size (SGLang) |
| `dp_size` | Number of data-parallel processes |
| `num_gpus` | Number of GPUs to use |
| `num_epochs` | Number of inference repetitions per data sample |
