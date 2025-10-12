# verl/examples/release_template/train_ppo_origin_grm.sh
#!/bin/bash
set -e

# === Parse parameters ===
STRATEGY=${1:-"origin"}  # origin
RATIO=${2:-"0.0"}        # edit ratio: 0.0

# Validate parameters
if [[ ! "$STRATEGY" =~ ^(origin)$ ]]; then
    echo "Error: strategy must be 'origin'"
    echo "Usage: $0 <strategy> <ratio>"
    echo "  strategy: origin (original rollout, without feedback, need grm only)"
    echo "  ratio: edit ratio between 0.0 (e.g., 0.0)"
    exit 1
fi

# # Validate ratio is a number between 0 and 1
# if ! [[ "$RATIO" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$RATIO < 0" | bc -l) )) || (( $(echo "$RATIO > 1" | bc -l) )); then
#     echo "Error: ratio must be a number between 0.0 and 1.0"
#     exit 1
# fi

# === Auto-detect project root ===
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../../../..")
VERL_ROOT=$(realpath "$SCRIPT_DIR/../../../..")

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# === Load environment configuration (optional) ===
CONFIG_FILE="${VERL_ROOT}/config/env.sh"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "No config file found at $CONFIG_FILE, using environment variables"
fi



export TRAIN_DATA="${PROJECT_ROOT}/eval/data/RL_data/train.parquet" 
export VAL_DATA="${PROJECT_ROOT}/eval/data/RL_data/test.parquet"

# === Check required environment variables ===
REQUIRED_VARS=(
    "BASE_MODEL_PATH"
    # "TRAIN_DATA"
    "GRM_API_BASE_URL"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "Error: Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please set these variables before running the script. See README for details."
    exit 1
fi

# === Set defaults for optional variables ===
# export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${BASE_MODEL_PATH}}"
export MODEL_TAG="${MODEL_TAG:-sft-qwen2.5-7b}"
export ROLLOUT="${ROLLOUT:-4}"
export GRM_MODEL_NAME="${GRM_MODEL_NAME:-grm_14B_320}"
export GRM_OPENAI_API_KEY="${GRM_OPENAI_API_KEY:-EMPTY}"


# original rollout, without feedback, need grm only

# export FEEDBACK_OPENAI_API_KEY="${FEEDBACK_OPENAI_API_KEY:-EMPTY}"
# export FEEDBACK_API_BASE_URL="${FEEDBACK_API_BASE_URL:-${GRM_API_BASE_URL}}"
# export FEEDBACK_MODEL_NAME="${FEEDBACK_MODEL_NAME:-${GRM_MODEL_NAME}}"

# === Strategy mapping ===
case $STRATEGY in
    origin)
        SELECT_FUN="origin"
        ;;
esac

# === Set training parameters ===
export random_edit_prob=$RATIO
export edit_sample_prob=$RATIO
export select_fun=$SELECT_FUN

# === Experiment naming ===
DATE=$(date +%Y%m%d)
export exp_name="${MODEL_TAG}_rollout${ROLLOUT}_ratio${RATIO}_${STRATEGY}_${DATE}"

# Change to VERL root directory
cd "$VERL_ROOT"

echo "=========================================="
echo "Critique-Post-Edit PPO Training"
echo "=========================================="
echo "Strategy: $STRATEGY"
echo "Edit ratio: $RATIO"
echo "Base model: $BASE_MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "GRM API: $GRM_API_BASE_URL"
echo "Experiment name: $exp_name"
echo "Output directory: ${VERL_ROOT}/output/${exp_name}"
echo "Select function(another name of strategy): $SELECT_FUN"
echo "=========================================="

# === Start training ===
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.edit_config.enable_edit=False \
    actor_rollout_ref.rollout.feedback_config.enable_feedback=False \
    actor_rollout_ref.rollout.select_config.enable_select=False \
    actor_rollout_ref.rollout.select_config.select_fun=${select_fun} \
    actor_rollout_ref.rollout.select_config.edit_sample_prob=${edit_sample_prob} \
    actor_rollout_ref.rollout.select_config.random_edit_prob=${random_edit_prob} \
    actor_rollout_ref.actor.use_off_policy_loss=False \
    actor_rollout_ref.actor.off_policy_max_clip=-1 \
    actor_rollout_ref.actor.off_policy_min_clip=-1 \
    actor_rollout_ref.model.path=${BASE_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.rollout.save_path="${VERL_ROOT}/output/${exp_name}" \
    actor_rollout_ref.rollout.save_data=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=80000 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.use_dynamic_bsz=True \
    reward_model.forward_max_token_len_per_gpu=80000 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='critique_post_edit' \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    reward_model.reward_manager="batch" \
    custom_reward_function.path=$VERL_ROOT/verl/utils/reward_score/grm.py \
    custom_reward_function.name="compute_score_grm_batch" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.n_gpus_per_node=8 \
    trainer.balance_batch=True \
    trainer.total_epochs=2


    # reward_model.model.path=$REWARD_MODEL_PATH \


echo "=========================================="
echo "Training completed!"
echo "Output directory: ${VERL_ROOT}/output/${exp_name}"
echo "Checkpoints saved to: ${VERL_ROOT}/checkpoints/"
echo "=========================================="