#!/bin/bash

export PYTHONPATH=/mnt/data/chenghao/rollout_rephrase/verl:$PYTHONPATH

EVAL_DIR="/mnt/data/meiling/code/PRM/eval"
INPUT_FOLDER="/mnt/data/meiling/code/RLHF/RLHF-Reward-Modeling-main/rm_benchmark_4o_5"
PERSONA_FOLDER="/mnt/data/meiling/code/Persona_QA/persona_all"
OUTPUT_FOLDER="/mnt/data/meiling/code/PRM/eval/generate_data"
# COMPARISON_RESULTS_DIR="/mnt/data/meiling/code/PRM/eval/comparison_results/ml/vs_origin/logprobs"
COMPARISON_RESULTS_DIR="/mnt/data/meiling/code/PRM/eval/comparison_results/ml/critique_tie"
# COMPARISON_RESULTS_DIR="/mnt/data/meiling/code/PRM/eval/comparison_results"

# BASELINE_FILE="/mnt/data/meiling/code/PRM/eval/generate_data/easy_gpt-4o.jsonl"
# BASELINE_FILE="/mnt/data/meiling/code/PRM/eval/generate_data/easy_DeepSeek-R1-Distill-Qwen-7B.jsonl"
BASELINE_FILE="/mnt/data/meiling/code/PRM/eval/generate_data/easy_sft_qwen2-7b_persona_prm_origin_ppo_BSZ128_mini_0614_step_80.jsonl"

# BASE_MODEL_PATH="/mnt/data/model/open_source_model/qwen2.5/Qwen2.5-7B-Instruct"
BASE_MODEL_PATH="/mnt/old_data/end_side/code/Weaver/output2/save_qa/qa_sft_1_qwen_7b_instruct_attndrop0.1_wd0.1_neft5_3e-5_128_epochs2_wp0.1/checkpoint-1200"
# BASE_MODEL_PATH="/mnt/data/model/open_source_model/qwen2.5/qwen2.5-7b"

# export TRAIN_SAVE_FOLDERS=(/mnt/data/chenghao/rollout_rephrase/verl/checkpoints/persona/sft_qwen2-7b_persona_prm_grpo_1_only_edit_rollout_5_BSZ128_0616)
export MODEL_NAMES=(
    sft_qwen2-7b_persona_prm_rollout7_0.1_sample_edit_ppo_BSZ128_0622
    sft_qwen2-7b_persona_prm_rollout7_0.5_sample_edit_ppo_BSZ128_0622
) 

# export TRAIN_SAVE_FOLDERS=(/mnt/data/chenghao/rollout_rephrase/verl/checkpoints/persona/qwen2-7b_persona_prm_0.1_select_best_ppo_BSZ128_mini_0613)
# export MODEL_NAME="qwen2-7b_persona_prm_0.1_select_best_ppo_BSZ128_mini_0613"

export STEP_VALUES=(40)

# for train_save_folder in "${TRAIN_SAVE_FOLDERS[@]}"; do
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    train_save_folder="/mnt/data/chenghao/rollout_rephrase/verl/checkpoints/persona/${MODEL_NAME}"
    for step in "${STEP_VALUES[@]}"; do
        BASELINE_FILE="/mnt/data/meiling/code/PRM/eval/generate_data/easy_sft_qwen2-7b_persona_prm_origin_ppo_BSZ128_mini_0614_step_${step}.jsonl"
        # BASELINE_FILE="/mnt/data/meiling/code/PRM/eval/generate_data/easy_sft_qwen2-7b_persona_prm_origin_rollout5_ppo_BSZ1288_mini_0618_step_${step}.jsonl"

        echo "处理文件夹: ${train_save_folder}, 步数: ${step}"
        
        # cd /mnt/data/chenghao/rollout_rephrase/verl/
        # python scripts/model_merger.py merge \
        #     --backend fsdp \
        #     --tie-word-embedding \
        #     --hf_model_path ${BASE_MODEL_PATH} \
        #     --local_dir "${train_save_folder}/global_step_${step}/actor/" \
        #     --target_dir "${train_save_folder}/global_step_${step}/huggingface"

        # python "${EVAL_DIR}/generate_response.py" \
        #     --model_name "${train_save_folder}/global_step_${step}/huggingface" \
        #     --input_folder "${INPUT_FOLDER}" \
        #     --persona_folder "${PERSONA_FOLDER}" \
        #     --output_folder "${OUTPUT_FOLDER}" \
        #     --tensor_parallel_size 4 \
        #     --custom_name "${MODEL_NAME}_step_${step}"

        python "${EVAL_DIR}/win_rate_critique_tie.py" \
            --file_a "${OUTPUT_FOLDER}/easy_${MODEL_NAME}_step_${step}.jsonl" \
            --file_b "${BASELINE_FILE}" \
            --max_questions 100 \
            --output_dir "${COMPARISON_RESULTS_DIR}"

        if [ $? -eq 0 ]; then
            echo "步数 ${step} 处理成功"
        else
            echo "步数 ${step} 处理失败"
        fi
    done
done

echo "所有任务完成"


