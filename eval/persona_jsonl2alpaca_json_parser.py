'''
keys are 
    "question":  "persona": "response": 
but I need :
    output instruction [dataset, generator]

看看ml如何拼接的 包括prompt persona进入instruction



你的任务是基于给定的用户画像和问题，判断哪个AI回答更符合该用户的需求和偏好。

【用户画像】
{persona}

【问题】
{question}

对照的是这个代码
/mnt/data/meiling/code/PRM/eval/win_rate_logprob.py

output_path=/mnt/data/meiling/code/PRM/eval/generate_data/easy_sft_qwen2-7b_persona_prm_origin_grpo_BSZ128_rollout5_KL_0003_0617_step_80.jsonl


有一个switch的工作， 然后切换过来，但是需要prompt，因为会有baseline的响应



---

usage
cd /mnt/workspace/chenghao/rollout_rephrase/src_jijivski/

input_jsonl_path=/mnt/data/meiling/code/PRM/eval/generate_data/easy_sft_qwen2-7b_persona_prm_origin_grpo_BSZ128_rollout5_KL_0003_0617_step_80.jsonl
# basename_input=$("basename $input_jsonl_path")
basename_input=$(basename -s .jsonl "$input_jsonl_path")
output_json_path=/mnt/data/chenghao/rollout_rephrase/src_jijivski/logs/alpaca_custom/${basename_input}
python evals/alpaca_custom/persona_jsonl2alpaca_json_parser.py --input ${input_jsonl_path} --output ${output_json_path} --dataset persona_eval

/mnt/data/chenghao/rollout_rephrase/src_jijivski/evals/alpaca_custom/persona_jsonl2alpaca_json_parser.py

'''


import json
import argparse
import os

def convert_jsonl_to_json(input_file):
    # 从文件名提取generator信息
    generator = os.path.basename(input_file)
    
    result = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 提取需要的字段
            question = data.get("question", "")
            persona = data.get("persona", "")
            response = data.get("response", "")
            
            # 构建instruction
            instruction = f"""请基于以下三个指标对两个回答进行评估和排名：Helpfulness帮助性、Personalization个性化、Natural自然度。

比较以下两个回答，从三个维度进行评估，最终选择更优的回答。以下是评估维度及详细标准（请严格按照标准执行比较评估）：

**1. Helpfulness 帮助性比较**
【优选标准】
- 选择信息更准确全面、深度更适宜的回答
- 选择更好解决用户问题的回答
- 选择实用性更强的回答

【不能选择的回答】
- 表面回答，没有实质帮助
- 回答偏离问题重点
- 信息有明显遗漏、不准确或严重错误
- 完全答非所问的回答


**2. Personalization 个性化比较**
【优选标准】
- 选择个性化元素自然融入且显著提升回答质量的
- 选择适度体现persona特征，个性化元素相关且有用的
- 选择个性化运用恰当不突兀的

【不能选择的回答】
- 强行套用persona信息，与问题关联度低
- 为了体现个性化而偏离问题重点
- 个性化元素显得生硬或刻意
- 强行提及persona中的工具、习惯但与问题无关
- 加入不相关的比喻或个人风格描述
- 为体现个性化而偏离问题本意

**3. Natural 自然度比较**
【优选标准】
- 选择完全自然的对话，无刻意痕迹的回答
- 选择语言自然流畅，不冗长啰嗦的回答
- 选择没有程式化表达的回答

【不能选择的回答】
- 开头显式提及用户名字，让用户感到不适和被监控感的回答(如：李文，基于您...)
- 典型的自我总结式表达（如：（注：回答中自然融入了用户的职业背景、旅行偏好、工具使用习惯以及可持续发展理念，同时提供了可立即实践的调试方法论，符合用户既追求专业效率又注重思维质量的特点。））
- 明显为了展示能力而添加的meta评论
- 出现事实性错误或幻觉内容
- 过度使用括号解释或标注
- 严重冗长啰嗦，大量无关废话的回答

以下是【用户配置】
{persona}

以下是【问题】
{question}

注意，只要满足了不能选择回答中的任意一点都不要选择该回答！在2个回答差不多的情况下，优先选择更短的。
"""


            
            # 构建新的字典
            new_item = {
                "dataset": args.dataset,
                "instruction": instruction,
                "output": response,
                "generator": generator
            }
            
            result.append(new_item)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSONL to JSON format')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--dataset', type=str, default='persona_eval', help='Dataset name')
    
    args = parser.parse_args()
    
    converted_data = convert_jsonl_to_json(args.input)
    # generator = os.path.basename(args.input)

    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion completed. Saved to {args.output}")


