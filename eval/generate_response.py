# generate_response.py
import json
import os
import re
import time
import logging
from pathlib import Path
import vllm
from transformers import AutoTokenizer
import ast
# length_prompt = os.environ.get("Length_prompt", "None")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMPERATURE = 0.9

def extract_user_config_from_text(text):
    pattern = r"当前用户配置：\s*(.*?)\s*\n\n根据用户配置生成此问题的1个完全个性化AI助手回答。"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        config_str = match.group(1).strip()
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            print(f"解析用户配置JSON时出错: {e}")
    return None

def load_persona_from_file(user_id, persona_folder):
    path = os.path.join(persona_folder, f"{user_id}.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载user_id {user_id}的persona出错: {e}")
    return None

def prepare_prompt(question, persona_config,length_prompt):
    if length_prompt=="None":
        suffix="生成1个个性化回答："
    elif length_prompt=="Longer":
        suffix="请给出一个详细的回答："
    elif length_prompt=="Shorter":
        suffix="请给出一个简短的回答："

    rtn= f"""你需要模拟AI助手对用户问题的回答。请根据提供的用户性格配置，以AI助手的身份回答这位特定用户的问题。
回答应该考虑用户的性格特点、价值观、兴趣爱好和使用习惯，提供最适合该用户的回答。

重要规则：
1. 回答应该符合AI助手的语气和风格，专业、友好且有帮助
2. 考虑用户的偏好和特点，提供个性化的回答
3. 回答应该能够满足用户的需求，不要太显示的展示出字段信息，要根据已有的Persona Config来推理用户可能想要的个性化回答，而不是简单的结合字段信息。
4. 回答应该直接解决用户的问题，同时体现对用户偏好的了解

注意，输出的回答不要过长，不要提及太具体的字段信息！
注意，输出的回答要基于事实，不要强行和字段关联，要自然衔接，不要强行关联！

当前用户配置：
{json.dumps(persona_config, ensure_ascii=False, indent=2)}

根据用户配置生成此问题的1个完全个性化AI助手回答。回答应反映对配置中定义的用户特征、偏好和习惯的理解。需要结合多个字段信息来进行个性化的回答。不要显示提及人名！

以下是用户问题：
{question}

"""+suffix
    print(rtn)
    return rtn

def extract_model_info(model_path):
    path_parts = model_path.rstrip('/').split('/')
    
    # 查找包含 global_step_ 的部分
    model_name = None
    step_info = None
    
    for i, part in enumerate(path_parts):
        if 'global_step_' in part:
            # 提取步数
            step_num = part.split('_')[-1]
            step_info = f"step_{step_num}"
            
            # 提取模型名称（global_step的前一个目录）
            if i > 0:
                model_name = path_parts[i-1].split('/')[-1]
            break
    
    if model_name and step_info:
        return f"{model_name}_{step_info}"
    else:
        # 如果无法解析，使用默认方式
        return model_path.split('/')[-3] if len(model_path.split('/')) >= 3 else 'unknown_model'

def generate_responses_batch(model_name, data_list, persona_folder, top_p=0.9, temperature=0.8, tensor_parallel_size=4, max_length=4096,length_prompt=None):

    instructions = []
    valid_indices = []
    
    for i, data in enumerate(data_list):
        try:
            question = data['question']
            if 'user_id' in data and data['user_id']:
                user_id = data['user_id']
                persona = load_persona_from_file(user_id, persona_folder) or extract_user_config_from_text(data.get('prompt', '')) or {}
            else:
                persona = extract_user_config_from_text(data.get('prompt', '')) or {}
            
            prompt = prepare_prompt(question, persona,length_prompt)
            instructions.append(prompt)
            valid_indices.append(i)
            
        except Exception as e:
            logger.error(f"准备第{i+1}行数据时出错: {e}")
    
    if not instructions:
        return []
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    formatted_instructions = []
    for instruction in instructions:
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_instructions.append(formatted_prompt)
        except Exception as e:
            logger.warning(f"Chat template失败，使用原始prompt: {e}")
            formatted_instructions.append(instruction)
    
    model_args = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "max_model_len": max_length,
        'max_num_seqs': 40, 
        "trust_remote_code": True,
    }
    
    gen_kwargs = {
        "temperature": temperature,
        "max_tokens": 4096,  
        "top_p": top_p,
        'skip_special_tokens': True
    }
    
    params = vllm.SamplingParams(**gen_kwargs)
    
    try:
        client = vllm.LLM(**model_args)
        
        # 生成回答
        outputs = client.generate(formatted_instructions, params)
        # responses = [output.outputs[0].text.strip() for output in outputs]
        responses = [tokenizer.decode(output.outputs[0].token_ids,skip_special_tokens=True) for output in outputs]
        
    except Exception as e:
        logger.error(f"vLLM生成失败: {e}")

        return []
    
    results = []
    for idx, response in zip(valid_indices, responses):
        data = data_list[idx]
        question = data['question']
        
        if 'user_id' in data and data['user_id']:
            user_id = data['user_id']
            persona = load_persona_from_file(user_id, persona_folder) or extract_user_config_from_text(data.get('prompt', '')) or {}
        else:
            persona = extract_user_config_from_text(data.get('prompt', '')) or {}
        
        results.append({
            "question": question,
            "persona": json.dumps(persona, ensure_ascii=False),
            "response": response
        })
    
    # 清理内存
    try:
        del client
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    
    return results


def generate(folder_path, persona_folder, output_folder, model_name, tensor_parallel_size=4, custom_name=None, length_prompt=None):
    os.makedirs(output_folder, exist_ok=True)
    file_names = ['specific_easy.jsonl','specific_mid.jsonl','specific_hard.jsonl','general_easy.jsonl','general_mid.jsonl','general_hard.jsonl']
    # file_names = ['hard.jsonl']
    # file_names = ['easy.jsonl']


    # 确定输出文件名
    if custom_name:
        model_basename = custom_name
        print(f"使用自定义文件名: {custom_name}")
    else: 
        model_basename = extract_model_info(model_name)
        print(f"自动提取的文件名: {model_basename}")
    
    # 统一的输出路径
    output_path = Path(output_folder) / f"mix_{model_basename}.jsonl"
    
    all_results = []  # 存储所有文件的结果
    total_processed = 0
    
    for file_name in file_names:
        input_path = Path(folder_path) / file_name
        if not input_path.exists():
            print(f"文件不存在: {input_path}")
            continue
        
        print(f"开始处理 {file_name}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
            lines = lines[:50] 
            
        print(f"从 {file_name} 中读取 {len(lines)} 个问题")
        
        # 生成响应
        results = generate_responses_batch(model_name, lines, persona_folder, tensor_parallel_size=tensor_parallel_size, length_prompt=length_prompt)
        
        if not results:
            print(f"处理 {file_name} 失败，没有结果返回")
            continue
        
        # 将结果添加到总结果中
        valid_results = [res for res in results if res and res.get('response')]
        all_results.extend(valid_results)
        total_processed += len(valid_results)
        
        print(f"{file_name} 处理完成，成功处理: {len(valid_results)}/{len(results)} 个问题")
    
    # save all results to one file
    if all_results:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for res in all_results:
                out_f.write(json.dumps(res, ensure_ascii=False) + '\n')
        
        print(f"所有文件处理完成，结果保存到: {output_path}")
        print(f"总计成功处理: {total_processed} 个问题，共 {len(all_results)} 条结果")
        return str(output_path)
    else:
        print("没有成功处理任何问题")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成个性化回答')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称或路径')
    parser.add_argument('--input_folder', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--persona_folder', type=str, required=True, help='用户persona文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='张量并行大小')
    parser.add_argument('--custom_name', type=str, default=None, help='自定义输出文件名（不含easy_前缀和.jsonl后缀）')
    parser.add_argument('--length_prompt', type=str, default=None)
    args = parser.parse_args()
    
    generate(args.input_folder, args.persona_folder, args.output_folder, args.model_name, args.tensor_parallel_size, args.custom_name, args.length_prompt)


"""
--------base-----------
python eval/generate_response.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --input_folder "./eval/data/rm_benchmark_all" \
    --persona_folder "./eval/data/persona_all" \
    --output_folder "./eval/generate_data_300" \
    --tensor_parallel_size 4 \
    --custom_name "Qwen2.5-7B-Instruct" \
    --length_prompt "None"

"""