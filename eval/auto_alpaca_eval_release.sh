#!/bin/bash
# eval/evaluate_models.sh
# Evaluation script for Critique-Post-Edit trained models

set -e

# === Auto-detect paths ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
EVAL_DIR="$SCRIPT_DIR"
VERL_ROOT="$PROJECT_ROOT/verl"

echo "=========================================="
echo "AlpacaEval Evaluation Setup"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "VERL root: $VERL_ROOT"
echo "=========================================="

# === Environment setup ===
export PYTHONPATH="${VERL_ROOT}:$PYTHONPATH"
export PYTHONPATH="${EVAL_DIR}/alpaca_eval/src:$PYTHONPATH"

# === Configuration: User needs to set these ===
# OpenAI API for AlpacaEval (gpt-4 as judge)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Please set: export OPENAI_API_KEY='your-key'"
    exit 1
fi

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"

# Base model path (SFT baseline model)
BASE_MODEL_PATH="path/to/your/sft/model" 
# you can use instruct model like Qwen2.5-7B-Instruct or Qwen2.5-14B-Instruct, since this is for merging checkpoint

# Evaluation data paths
INPUT_FOLDER="${INPUT_FOLDER:-${PROJECT_ROOT}/eval/data/rm_benchmark_all}"
PERSONA_FOLDER="${PERSONA_FOLDER:-${PROJECT_ROOT}/eval/data/persona_all}"

# Output paths
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${EVAL_DIR}/generate_data_300}"


# Baseline reference (e.g., GPT-4o)
BASELINE_JSONL="${EVAL_DIR}/generate_data_300/mix_gpt-4o.jsonl"
BASELINE_JSON="${EVAL_DIR}/generate_data_300/mix_gpt-4o.json"

mkdir -p "$OUTPUT_FOLDER"

# === Models to evaluate ===
MODEL_NAMES=(
    # Add your trained model names here
    sft-qwen2.5-7b_rollout4_ratio0.5_random_20251005
】

# Check if models are specified
if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
    echo "Warning: No models specified in MODEL_NAMES array"
    echo "Please edit this script and add model names to evaluate"
    echo "Example: MODEL_NAMES=(model_llama_7b_rollout4_0.1_improve_20250801)"
    exit 1
fi

# Checkpoints to evaluate (training steps)
# STEP_VALUES=(20 40 60 80 100 120 140 160 180 200)
STEP_VALUES=(60 80)



echo "=========================================="
echo "Will evaluate ${#MODEL_NAMES[@]} model(s) at ${#STEP_VALUES[@]} checkpoint(s)"
echo "=========================================="

# === Main evaluation loop ===
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    TRAIN_SAVE_FOLDER="${VERL_ROOT}/checkpoints/critique_post_edit/${MODEL_NAME}"

    if [ ! -d "$TRAIN_SAVE_FOLDER" ]; then
        echo "Warning: Model folder not found: $TRAIN_SAVE_FOLDER"
        echo "Skipping..."
        continue
    fi
    
    echo "Evaluating model: $MODEL_NAME"
    
    for step in "${STEP_VALUES[@]}"; do
        CHECKPOINT_DIR="${TRAIN_SAVE_FOLDER}/global_step_${step}"


        echo "  Processing checkpoint: step ${step}"
        HF_DIR="${CHECKPOINT_DIR}/huggingface"
        ACTOR_DIR="${CHECKPOINT_DIR}/actor"
        INPUT_JSONL="${OUTPUT_FOLDER}/mix_${MODEL_NAME}_step_${step}.jsonl"
        OUTPUT_JSON="${OUTPUT_FOLDER}/mix_${MODEL_NAME}_step_${step}.json"
        
        # === Step 1: Merge FSDP model to HuggingFace format ===
        echo "    [1/4] Merging model..."
        if [ -d "$HF_DIR" ]; then
            echo "    HuggingFace format already exists: $HF_DIR"
        else
            if [ ! -d "$ACTOR_DIR" ]; then
                echo "    Skip: neither HuggingFace dir nor actor weights found for step ${step}"
                continue
            fi
            cd "$VERL_ROOT"
            python scripts/model_merger.py merge \
                --backend fsdp \
                --tie-word-embedding \
                --hf_model_path "${BASE_MODEL_PATH}" \
                --local_dir "${ACTOR_DIR}/" \
                --target_dir "${HF_DIR}"
        fi

        # === Step 2: Generate responses ===
        echo "    [2/4] Generating responses..."
        cd "$EVAL_DIR"
        
        # Check if generation script exists
        if [ ! -f "${EVAL_DIR}/generate_response.py" ]; then
            echo "    Warning: generate_response.py not found, skipping generation"
            echo "    You need to implement response generation for your evaluation data"
            continue
        fi
        
        
        python "${EVAL_DIR}/generate_response.py" \
            --model_name "${CHECKPOINT_DIR}/huggingface" \
            --input_folder "${INPUT_FOLDER}" \
            --persona_folder "${PERSONA_FOLDER}" \
            --output_folder "${OUTPUT_FOLDER}" \
            --tensor_parallel_size 4 \
            --custom_name "${MODEL_NAME}_step_${step}" \
            --length_prompt "None"
        
        # === Step 3: Convert format for AlpacaEval ===
        echo "    [3/4] Converting format..."

        if [ ! -f "$INPUT_JSONL" ]; then
            echo "    Error: Generated file not found: $INPUT_JSONL"
            continue
        fi
        
        # Check if converter script exists
        if [ ! -f "${EVAL_DIR}/persona_jsonl2alpaca_json_parser.py" ]; then
            echo "    Warning: Format converter not found"
            echo "    You need to convert $INPUT_JSONL to AlpacaEval format manually"
            continue
        fi
        
        python "${EVAL_DIR}/persona_jsonl2alpaca_json_parser.py" \
            --input "${INPUT_JSONL}" \
            --output "${OUTPUT_JSON}" \
            --dataset persona_eval
        
        # === Step 4: Run AlpacaEval ===
        echo "    [4/4] Running AlpacaEval..."
        
        # Prepare baseline if needed
        if [ ! -f "${BASELINE_JSON}" ] && [ -f "${BASELINE_JSONL}" ]; then
            echo "    Converting baseline file..."
            python "${EVAL_DIR}/persona_jsonl2alpaca_json_parser.py" \
                --input "${BASELINE_JSONL}" \
                --output "${BASELINE_JSON}" \
                --dataset persona_eval
        fi
        
        if [ ! -f "${BASELINE_JSON}" ]; then
            echo "    Warning: Baseline file not found: ${BASELINE_JSON}"
            echo "    Skipping AlpacaEval..."
            continue
        fi
        
        alpaca_eval evaluate \
            --model_outputs "${OUTPUT_JSON}" \
            --annotators_config "${EVAL_DIR}/alpaca_eval/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_1" \
            --output_path "${EVAL_DIR}/alpaca_eval_release" \
            --precomputed_leaderboard "${EVAL_DIR}/alpaca_eval_release/leaderboard_300_release.csv" \
            --is_overwrite_leaderboard \
            --metric_kwargs "{'glm_name':'length_controlled_minimal'}" \
            --reference_outputs "${BASELINE_JSON}"
        
        echo "    ✓ Checkpoint step_${step} completed"
    done
    
    echo "✓ Model $MODEL_NAME evaluation completed"
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: ${EVAL_DIR}/alpaca_eval_results"
echo "=========================================="