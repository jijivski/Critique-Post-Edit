[中文](./README_zh.md) | [English](./README.md)

# 📋 项目概述

这是我们论文 "Towards Faithful and Controllable Personalization via Critique-Post-Edit Reinforcement Learning“ 的官方库。 本项目提供了一套完整的，使用 **Critique-Post-Edit** 方法来训练和评估大语言模型的流程。我们利用了 `LLaMA-Factory` 和 `verl` 等优秀的开源框架，涵盖了监督微调（SFT）和强化学习（PPO）的训练脚本与配置。项目使用 `AlpacaEval` 进行模型性能的全面评估。我们开源的 `Personalized-Qwen2.5-7B-Instruct` 和 `Personalized-Qwen2.5-14B-Instruct` 等模型相较于基线模型有显著的性能提升。

## 📂 代码结构

```
Critique-Post-Edit
├── LLaMA-Factory/          # SFT 训练代码
│   ├── data/
│   │   ├── sft.json        # GRM 训练数据
│   │   └── sft_grm.json    # SFT 训练数据
│   └── examples/
├── verl/                   # RL 训练代码
│   └── examples/ppo_trainer/
└── eval/                   # 评估代码
    ├── data/
    │   └── RL_data         # RL 训练数据
    └── alpaca_eval/
```

# 🚀 快速开始

## 🧩 环境设置

推荐方式：使用支持 VERL, LLaMA-Factory 的现有环境配置，然后加上环境变量指向我们的代码路径即可运行。 AlpacaEval会在下面的步骤中安装。

```bash
# 1. 克隆项目
git clone https://github.com/OPPO-PersonalAI/Critique-Post-Edit.git
cd Critique-Post-Edit

# 2. 设置环境变量
export WANDB_API_KEY=your_wandb_api_key
export PYTHONPATH=$PWD/verl:$PYTHONPATH # 环境变量需要指向我们的代码路径

export OPENAI_API_KEY="sk-your-key"
export OPENAI_API_BASE="your-url"
```

## 📦 数据准备

从指定位置下载以下数据集：

*   **GRM 训练数据**: `LLaMA-Factory/data/sft.json`
*   **SFT 训练数据**: `LLaMA-Factory/data/sft_grm.json`
*   **RL 训练数据**: `eval/data/RL_data`

我们的模型在huggingface上:

*   [GRM-Qwen2.5-14B-Instruct](https://huggingface.co/PersonalAILab/GRM-Qwen2.5-14B-Instruct)
*   [Personalized-Qwen2.5-7B-Instruct](https://huggingface.co/PersonalAILab/Personalized-Qwen2.5-7B-Instruct)
*   [Personalized-Qwen2.5-14B-Instruct](https://huggingface.co/PersonalAILab/Personalized-Qwen2.5-14B-Instruct)

## 🧠 模型训练

7B训练在一台 8 卡 A800 上完成。 14B训练建议使用在2台 8 卡 A800。
强化学习部分额外使用两张A800 GPU 部署 vLLM。

*   **SFT 训练时间**：几小时即可完成
*   **RL 训练时间 7B**：约 3 天

### 训练 GRM

```bash
cd LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/grm_14b.yaml
# 输出默认保存在 LLaMA-Factory/saves
```

### 部署 GRM（使用 vLLM）

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

使用 `ifconfig` 查询 IP，然后将 GRM 的端口配置到训练脚本的环境变量中：

```bash
# === Required: Model and Data Paths ===
export BASE_MODEL_PATH="path/to/your/sft/model"
export GRM_API_BASE_URL="http://your-vllm-ip:8001/v1"
export MODEL_TAG="example:sft-qwen2.5-7b [or example:sft-qwen2.5-14b]"

# 验证服务
curl http://your-vllm-ip:8001/v1/models

# === Optional: Advanced Settings ===
export ROLLOUT=4
export VERL_ROOT="$(pwd)/verl"

export GRM_OPENAI_API_KEY="EMPTY"
export FEEDBACK_OPENAI_API_KEY="EMPTY"
export FEEDBACK_API_BASE_URL="${GRM_API_BASE_URL}"
export FEEDBACK_MODEL_NAME="${GRM_MODEL_NAME}"

# 主方法：Critique-Post-Edit，编辑比例 50%
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.5

# 不同编辑比例
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.1
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.25
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh random 0.75

# 不同策略
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh reward 0.5
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh improve 0.5
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_critique_edit_strategy_X_ratio.sh improve 0.1

# 默认参数：learning_rate=1e-6, batch_size=128, epoch=2
# Vanilla PPO（基于 GRM）
bash verl/examples/ppo_trainer/persona_prms/release_template/train_ppo_origin_grm.sh origin 0
```

训练输出位置：

*   模型检查点: `verl/checkpoints/`
*   Rollout 解码结果: `verl/output/`

## 📊 模型评估

```bash
BASE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd eval/alpaca_eval
pip install -e .
cd ..

# 配置 OpenAI API（GPT-4.1 用于评测）
export OPENAI_API_KEY="sk-your-key"
export OPENAI_API_BASE="your url"

# 运行评估
bash eval/auto_alpaca_eval_release.sh

# 实际上用 4 张卡就可以跑一个评估，所以你可以同时跑两个。
CUDA_VISIBLE_DEVICES=4,5,6,7 bash eval/auto_alpaca_eval_release.sh
```

评估输出结构：

```
eval/
├── alpaca_eval_release/
│   └── leaderboard_300_release.csv    # 主要指标
├── alpaca_eval_results/
│   └── annotations.json               # 详细对比
└── generate_data_300/                 # 基本上和 leaderboard_300_release.cvs 内容一样
```

`leaderboard_300_release.csv` 指标说明：

| 指标                            | 含义               |
| ----------------------------- | ---------------- |
| **win_rate**                  | 相对基线模型的胜率        |
| **avg_length**                | 平均回复长度           |
| **length_controlled_winrate** | 长度控制后的胜率（更公平的对比） |

## 📈 实验结果

| 模型      | 方法  | win-rate | length-controlled win-rate |
| ------- | --- | -------- | -------------------------- |
| Qwen2.5-7B-Instruct  | Original | 27.3      | 31.2                        |
| Qwen2.5-14B-Instruct | Original | 28.9      | 33.2                        |
| Personalized-Qwen2.5-7B-Instruct  | Critique-Post-Edit | 64.6      |   63.7                      |
| Personalized-Qwen2.5-14B-Instruct | Critique-Post-Edit | 74.3      |   75.5                      |

## 📜 许可证

本项基于 Apache License Version 2.0 许可。

## 致谢

本项目基于以下开源项目：

*   [VERL](https://github.com/volcengine/verl) — ByteDance 开源的 RLHF 框架
*   [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 高效的 LLM 训练工具
*   [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) — 模型评估工具
