[中文](./README_zh.md) | [English](./README.md)


# 📋 Project Overview

This is the official repository for our paper "Towards Faithful and Controllable Personalization via Critique-Post-Edit Reinforcement Learning“. This project provides a complete pipeline for training and evaluating large language models using the **Critique-Post-Edit** method. It includes scripts and configurations for Supervised Fine-Tuning (SFT) and Reinforcement Learning (PPO), leveraging powerful open-source frameworks like `LLaMA-Factory` and `verl`. The evaluation is conducted using `AlpacaEval` to ensure fair and comprehensive assessment of model performance. Our released models, including `Personalized-Qwen2.5-7B-Instruct` and `Personalized-Qwen2.5-14B-Instruct`, demonstrate significant improvements over the baseline models.

## 📂 Code Structure

```
Critique-Post-Edit
├── LLaMA-Factory/          # SFT training code
│   ├── data/
│   │   ├── sft.json        # SFT training data
│   │   └── sft_grm.json    # GRM training data
│   └── examples/
├── verl/                   # RL training code
│   └── examples/ppo_trainer/
└── eval/                   # Evaluation code
    ├── data/
    │   └── RL_data         # RL training data
    └── alpaca_eval/
```

# 🚀 Quick Start

## 🧩 Environment Setup

Recommended approach: Use an existing environment that supports VERL and LLaMA-Factory, then add the environment variable pointing to our code path to run. AlpacaEval will be installed in the steps below.

```bash
# 1. Clone the project
git clone https://github.com/OPPO-PersonalAI/Critique-Post-Edit.git
cd Critique-Post-Edit

# 2. Set environment variables
export WANDB_API_KEY=your_wandb_api_key
export PYTHONPATH=$PWD/verl:$PYTHONPATH # The environment variable needs to point to our code path

export OPENAI_API_KEY="sk-your-key"
export OPENAI_API_BASE="your-url"
```

## 📦 Data Preparation

Download the following datasets from the specified locations:

*   **GRM Training Data**: `LLaMA-Factory/data/sft_grm.json`
*   **SFT Training Data**: `LLaMA-Factory/data/sft.json`
*   **RL Training Data**: `eval/data/RL_data`

Our models are available on Hugging Face:

*   [GRM-Qwen2.5-14B-Instruct](https://huggingface.co/PersonalAILab/GRM-Qwen2.5-14B-Instruct)
*   [Personalized-Qwen2.5-7B-Instruct](https://huggingface.co/PersonalAILab/Personalized-Qwen2.5-7B-Instruct)
*   [Personalized-Qwen2.5-14B-Instruct](https://huggingface.co/PersonalAILab/Personalized-Qwen2.5-14B-Instruct)

## 🧠 Model Training

The 7B model was trained on a single server with 8 A800 GPUs. For the 14B model, it is recommended to use two servers with 8 A800 GPUs each.
The reinforcement learning part additionally uses two A800 GPUs to deploy vLLM.

*   **SFT Training Time**: Completes in a few hours
*   **RL Training Time (7B)**: Approximately 3 days

### Train GRM

```bash
cd LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/grm_14b.yaml
# Output is saved to LLaMA-Factory/saves by default
```

### Deploy GRM (using vLLM)

```bash
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --trust-remote-code \
    --served-model-name grm_14B_320 \
    --model PersonalAILab/GRM-Qwen2.5-14B-Instruct \
    --gpu-memory-utilization 0.98 \
    --tensor-parallel-size 2 \
    --port 8001
```

Use `ifconfig` to find the IP address, then configure the GRM port in the training script's environment variables:

```bash
# === Required: Model and Data Paths ===
export BASE_MODEL_PATH="path/to/your/sft/model"
export GRM_API_BASE_URL="http://your-vllm-ip:8001/v1"
export MODEL_TAG="example:sft-qwen2.5-7b [or example:sft-qwen2.5-14b]"

# Verify the service
curl http://your-vllm-ip:8001/v1/models

# === Optional: Advanced Settings ===
export ROLLOUT=4
export VERL_ROOT="$(pwd)/verl"

export GRM_OPENAI_API_KEY="EMPTY"
export FEEDBACK_OPENAI_API_KEY="EMPTY"
export FEEDBACK_API_BASE_URL="${GRM_API_BASE_URL}"
export FEEDBACK_MODEL_NAME="${GRM_MODEL_NAME}"

# Main method: Critique-Post-Edit, with an edit ratio of 50%
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.5

# Different edit ratios
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.1
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.25
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.75

# Different strategies
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh reward 0.5
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh improve 0.5
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh improve 0.1

# Default parameters: learning_rate=1e-6, batch_size=128, epoch=2
# Vanilla PPO (based on GRM)
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_origin_grm.sh origin 0
```

Training output locations:

*   Model checkpoints: `verl/checkpoints/`
*   Rollout decoding results: `verl/output/`

## 📊 Model Evaluation

```bash
BASE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd eval/alpaca_eval
pip install -e .
cd ..

# Configure OpenAI API (GPT-4.1 for evaluation)
export OPENAI_API_KEY="sk-your-key"
export OPENAI_API_BASE="your url"

# Run evaluation
bash eval/auto_alpaca_eval_release.sh

# You can actually run one evaluation with 4 GPUs, so you can run two simultaneously.
CUDA_VISIBLE_DEVICES=4,5,6,7 bash eval/auto_alpaca_eval_release.sh
```

Evaluation output structure:

```
eval/
├── alpaca_eval_release/
│   └── leaderboard_300_release.csv    # Main metrics
├── alpaca_eval_results/
│   └── annotations.json               # Detailed comparison
└── generate_data_300/                 # Basically the same content as leaderboard_300_release.csv
```

Description of metrics in `leaderboard_300_release.csv`:

| Metric                      | Meaning                                     |
| ----------------------------- | ------------------------------------------- |
| **win_rate**                  | Win rate relative to the baseline model     |
| **avg_length**                | Average response length                     |
| **length_controlled_winrate** | Win rate after length control (fairer comparison) |

## 📈 Experimental Results

| Model                        | Method               | win-rate | length-controlled win-rate |
| ---------------------------- | -------------------- | -------- | -------------------------- |
| Qwen2.5-7B-Instruct          | Original             | 27.3     | 31.2                       |
| Qwen2.5-14B-Instruct         | Original             | 28.9     | 33.2                       |
| Personalized-Qwen2.5-7B-Instruct  | Critique-Post-Edit   | 64.6     | 63.7                       |
| Personalized-Qwen2.5-14B-Instruct | Critique-Post-Edit   | 74.3     | 75.5                       |

## 📜 License

This project is licensed under the Apache License Version 2.0.

## Acknowledgements

This project is based on the following open-source projects:

*   [VERL](https://github.com/volcengine/verl) — An RLHF framework open-sourced by ByteDance
*   [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — An efficient LLM training toolkit
*   [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) — A model evaluation tool
