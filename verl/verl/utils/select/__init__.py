
# not used in this release, but maybe someone could be interested
# def only_edit()

import json

import torch

from verl import DataProto


def compute_pairwise_distances(responses,texts:list, method="cosine"):
    """
    计算token序列之间的成对距离，先将token转换为字符串
    
    参数:
    responses: 形状为[batch_size, seq_len]的token张量
    tokenizer: HuggingFace tokenizer对象
    method: 距离计算方法，可选"cosine"或"edit"
    
    返回:
    distances: 形状为[batch_size, batch_size]的距离矩阵
    """
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_distances
    
    batch_size = responses.shape[0]
    
    distances = torch.zeros((batch_size, batch_size), device=responses.device)
    
    if method == "cosine":
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 使用TF-IDF向量化文本
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # 计算余弦距离
        cosine_dists = cosine_distances(tfidf_matrix)
        distances = torch.tensor(cosine_dists, device=responses.device)
    
    elif method == "edit":
        import Levenshtein
        
        for i in range(batch_size):
            for j in range(i, batch_size):
                # 计算归一化编辑距离
                max_len = max(len(texts[i]), len(texts[j]))
                if max_len == 0:
                    dist = 0.0
                else:
                    dist = Levenshtein.distance(texts[i], texts[j]) / max_len
                
                distances[i, j] = dist
                distances[j, i] = dist
    
    return distances

def save_selected_text(texts, avg_distances, selected_idxs, discarded_idxs, config):
    import datetime
    import os

    # 创建输出目录（如果不存在）
    output_dir = config.get("output_dir", "diversity_selection_results")#TODO
    
    # 创建带有时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    
    distance_folder = os.path.join(output_dir, 'max_dis_selection')
    os.makedirs(distance_folder,exist_ok=True)
    json_file = os.path.join(distance_folder, f"diversity_selection_{timestamp}.json")

    
    # 准备JSON数据
    results = []
    for i, idx in enumerate(selected_idxs):
        status = "SELECTED"
        results.append({
            "original_index": int(idx),
            "diversity_score": float(avg_distances[idx]),
            "status": status,
            "text": texts[idx]
        })
    for i, idx in enumerate(discarded_idxs):
        status = "DISCARDED"
        results.append({
            "original_index": int(idx),
            "diversity_score": float(avg_distances[idx]),
            "status": status,
            "text": texts[idx]
        })
    
    # 保存为JSON文件
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def _select_latter_half(data: DataProto) -> DataProto:  
    """  
    筛选DataProto，保留后面的一半数据（使用索引语法）  
      
    Args:  
        data (DataProto): 输入的DataProto对象  
          
    Returns:  
        DataProto: 包含后一半数据的新DataProto对象  
    """  
    total_length = len(data)  
    start_idx = total_length // 2  
    rtn=data[start_idx:]
    # breakpoint()
    return rtn



def _select_max_distance(final_data: DataProto, config:dict) -> DataProto:
    # 基于response部分计算距离
    # breakpoint()
    tokenizer=config.get("tokenizer",None)
    responses = final_data.batch["responses"]   
    keep_ratio = float(config.get("keep", 0.5))
    keep_size = int(len(final_data) * keep_ratio)


    # 如果提供了tokenizer，先将token转换为字符串
    if tokenizer:
        texts = []
        for response in responses:
            # 移除padding token
            valid_tokens = response[response != tokenizer.pad_token_id]
            text = tokenizer.decode(valid_tokens)
            texts.append(text)

    # 计算各response之间的编辑距离或余弦距离
    distances = compute_pairwise_distances(responses,texts=texts,method='cosine')
    

    
    # 计算每个样本与其他所有样本的平均距离
    avg_distances = []
    for i in range(len(distances)):
        # 排除自身距离（通常为0）
        other_distances = [distances[i][j] for j in range(len(distances)) if i != j]
        avg_distances.append(sum(other_distances) / len(other_distances))
    
    # 选择平均距离最大的keep_size个样本
    sorted_idxs = sorted(range(len(avg_distances)), key=lambda i: avg_distances[i], reverse=True)
    
    selected_idxs = sorted_idxs[:keep_size]
    discarded_idxs = sorted_idxs[keep_size:]
    
    # breakpoint()
    save_selected_text(texts, avg_distances, selected_idxs, discarded_idxs, config)


    print(f"Selected {len(selected_idxs)} responses from {len(final_data)} responses (keep ratio: {keep_ratio:.2f})")
    return final_data.select_idxs(selected_idxs)



def _select_max_min_distances(distances, keep_size):
    n = len(distances)
    # 从任意点开始，这里选择第一个
    selected = [0]
    
    while len(selected) < keep_size:
        max_min_dist = -1
        max_idx = -1
        
        # 对于每个未选择的点
        for i in range(n):
            if i not in selected:
                # 计算到已选择点集合的最小距离
                min_dist = min(distances[i][j] for j in selected)
                
                # 更新最大最小距离
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    max_idx = i
        
        if max_idx != -1:
            selected.append(max_idx)
    
    return selected





def calculate_ngram_repetition_rate(text, n=2):
    """
    calculate the repetition rate of n-gram
    calculate the repetition rate: 1 - (unique n-gram number / total n-gram number)
    """
    import re
    from collections import Counter

    if isinstance(n,str):
        n = int(n)

    # 英文文本预处理 或者这里处理的是tokens 组成的 类似英文的东西：
    # '103942 3837 100339 99900 1773 56568'
    def preprocess(text):
        # 转换为小写并移除标点符号
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens


    # 中文文本预处理 - 字符级别的分词
    # def preprocess(text):
    #     # 移除标点符号和空白
    #     text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    #     # 对于中文，我们可以直接将每个字符作为一个token
    #     tokens = list(text)
    #     return tokens
    # 生成n-gram
    def get_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    # 预处理文本
    tokens = preprocess(text)
    
    # 生成n-gram
    ngrams = get_ngrams(tokens, n)
    
    # 计算n-gram频率
    counter = Counter(ngrams)
    # return counter.values
    unique_ngrams = len(counter)
    total_ngrams = len(ngrams)
    
    repetition_rate = 1.0 - (unique_ngrams / (total_ngrams+1e-6))

    return repetition_rate

def filter_batch_by_ngram_maxlen(batch: DataProto, traj_bsz: int, max_len: int,   
                           ngram_thresholds: dict = None) -> DataProto:  
    """  
    根据长度和n-gram重复率过滤批次  
      
    Args:  
        batch: 输入批次  
        traj_bsz: 目标批次大小  
        max_len: 最大响应长度  
        ngram_thresholds: n-gram重复率阈值，如 {2: 0.2, 3: 0.12}  
      
    Returns:  
        过滤后的批次  
    """  
      
    # 获取响应文本和长度  
    responses = batch.batch["responses"]  # shape: [batch_size, seq_len]  
    attention_mask = batch.batch["attention_mask"]  
      
    # 计算响应长度（只计算response部分）  
    response_lengths = attention_mask[:, -responses.shape[-1]:].sum(dim=-1)  
      
    # 1. 长度过滤：去掉达到最大长度的序列  
    length_filter = response_lengths < max_len  
      

    try:
        total = len(response_lengths)
        kept = length_filter.sum().item()
        filtered = total - kept
        print(f"总数: {total}，长度过滤掉了 {filtered} 条，剩余 {kept} 条")
    except:
        print('error verl/verl/utils/select/__init__.py L275')

    # 2. n-gram重复率过滤  
    quality_filter = torch.ones(len(batch), dtype=torch.bool)  
      



    # 需要将token ids转换为文本进行n-gram分析  
    # 没有tokenizer可用  
    for i in range(len(batch)):  
        if not length_filter[i]:  
            continue  
              
        # 获取响应部分的token ids  
        response_tokens = responses[i]  
        response_mask = attention_mask[i, -responses.shape[-1]:]  
        valid_tokens = response_tokens[response_mask.bool()]  
          
        # text = tokenizer.decode(valid_tokens, skip_special_tokens=True)  
        # 或者直接使用token ids作为文本  
        text = " ".join([str(token.item()) for token in valid_tokens])  
          
        # 检查n-gram重复率  
        passed_ngram_filter = True  
        for n, threshold in ngram_thresholds.items():  

            # breakpoint()

            repetition_rate = calculate_ngram_repetition_rate(text, n=n)  
            if repetition_rate > threshold:  
                print(f"n={n}, repetition_rate={repetition_rate:.2f}, threshold={threshold:.2f}")
                passed_ngram_filter = False  
                break  
          

        quality_filter[i] = passed_ngram_filter  
    # breakpoint()
      
    
    # 综合过滤条件  
    final_filter = length_filter & quality_filter  
    filtered_indices = torch.where(final_filter)[0]  
      

    
    # 应用过滤  
    if len(filtered_indices) == 0:  
        print("警告: 所有样本都被过滤掉了，保留原始批次")  
        return batch[:traj_bsz]  
    elif len(filtered_indices) < traj_bsz:  
        print(f"警告: 过滤后样本数量({len(filtered_indices)}) < 目标批次大小({traj_bsz})，复制部分过滤后的样本")  
        # 计算需要额外复制的样本数量
        needed_samples = traj_bsz - len(filtered_indices)
        
        # 将filtered_indices转换为列表
        filtered_list = filtered_indices.tolist()
        
        # 循环复制样本直到达到目标批次大小
        additional_indices = []
        while len(additional_indices) < needed_samples:
            # 从过滤后的样本中循环选择
            for idx in filtered_list:
                if len(additional_indices) < needed_samples:
                    additional_indices.append(idx)
                else:
                    break
        
        # 合并原始过滤后的索引和额外复制的索引
        final_indices = filtered_list + additional_indices
        
        if len(batch[final_indices])!=traj_bsz:
            breakpoint()

            
        return batch[final_indices]
    else:  
        print(f'have enough to filter: {len(filtered_indices)}-{traj_bsz}={len(filtered_indices)-traj_bsz}')

        # breakpoint()
        # save to tensor
        # 如果过滤后仍然过多，取前traj_bsz个  
        selected_indices = filtered_indices[:traj_bsz]  
        return batch[selected_indices.tolist()]

# do not have text in non_text data
# # ---
#     # if self.save_data and self.save_path:
    
#     import datetime
#     import json
#     import os
    
#     # prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["prompts"]]
#     prompts_text = [text for text in batch.non_tensor_batch["prompts"]]
#     # responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
    
#     save_data = []
#     for i in range(len(prompts_text)):
#         data_entry = {
#             "prompt": prompts_text[i],
#             "response": responses_text[i],
#             "selected": i >= total_length // 2
#         }
#         save_data.append(data_entry)
            
#     non_tensor_batch={'data_source': 'persona', 'ability': 'alignment', 'reward_model': {'ground_truth': '推荐尝试拉威尔的《水之嬉戏》——流动的钢琴音色与光影变幻形成奇妙共振，适合捕捉颜料渐变时的微妙情绪。德彪西的《前奏曲》第二册中《枯叶》篇章，其朦胧和弦能唤醒你在皖南观察到的斑驳漆器质感。若需要持续张力，梅西安《图伦加利拉交响曲》中鸟鸣动机的重复与变奏，或许能呼应你速写本上那些突然刹停记录的瞬间。这些作品在时间维度上的层次感，或许比商业榜单上的热门曲目更契合你的创作呼吸节奏。', 'style': 'model'}, 'extra_info': {'index': 759, 'split': 'test'}, 'index': 759, 'uid': '25befaad-d356-4002-b850-b69fa7591e04', 'raw_prompt': array([{'content': '\n你需要模拟AI助手对用户问题的回答。请根据提供的用户性格配置，以AI助手的身份回答这位特定用户的问题。\n回答应该考虑用户的性格特点、价值观、兴趣爱好和使用习惯，提供最适合该用户的回答。\n\n重要规则：\n1. 回答应该符合AI助手的语气和风格，专业、友好且有帮助\n2. 考虑用户的偏好和特点，提供个性化的回答\n3. 回答应该能够满足用户的需求，不要太显示的展示出字段信息，要根据已有的Persona Config来推理用户可能想要的个性化回答，而不是简单的结合字段信息。\n4. 回答应该直接解决用户的问题，同时体现对用户偏好的了解\n\n注意，输出的回答不要过长，不要提及太具体的字段信息！\n注意，输出的回答要基于事实，不要强行和字段关联，要自然衔接，不要强行关联！\n\n当前用户配置：\n{\n  "Name": "李刚",\n  "Demographics": {\n    "Age": 22,\n    "Gender": "女",\n    "Nationality": "中国",\n    "Language": [\n      "中文",\n      "英文"\n    ],\n    "Career_Information": "艺术学院学生，擅长油画和雕塑"\n  },\n  "Personality": {\n    "Extraversion_or_Introversion": "I",\n    "Sensing_or_Intuition": "N",\n    "Thinking_or_Feeling": "F",\n    "Judging_or_Perceiving": "J",\n    "Values_and_Interests": [\n      "喜欢探索情感深度的艺术作品",\n      "热爱大自然，常去户外写生",\n      "对心理学有浓厚兴趣",\n      "享受安静的独处时光",\n      "收藏古典音乐唱片",\n      "抵触过度商业化的艺术创作",\n      "坚信自然景观比城市建筑更能激发人性真实"\n    ]\n  },\n  "Preference": {\n    "Local_Life": {\n      "Food_Drinks": "偏爱学校附近的手工咖啡馆，总坐在靠窗位置观察行人神态；素食主义者，但对摆盘美感有近乎偏执的要求，会为造型独特的甜点破例。",\n      "Entertainment_Shopping": "每周三固定逛老城区艺术材料市场，熟悉每个摊位的纹理纸供应商；在二手书店收集上世纪欧洲画册，认为泛黄的纸张自带时间维度。",\n      "Daily_Outings": "骑改装过的凤凰牌自行车通勤，车筐里永远放着速写本，遇到有趣的光影变化会突然刹停记录"\n    },\n    "Travel": {\n      "Business_Trips": "参与跨省艺术联展时会刻意选择夜间绿皮火车，认为铁轨与枕木的规律震动能催生创作欲",\n      "Leisure_Travel": "专程探访偏远古镇的祠堂壁画，曾在皖南山区连续七天跟踪观察某个老漆匠的作息"\n    },\n    "Social": "在艺术展签售会上会主动与创作者探讨表现主义技法，但拒绝交换联系方式；运营着只有37个粉丝的豆瓣小号，专门记录地铁乘客的微表情分析",\n    "Productivity": {\n      "Study_Work": "用矿物颜料自制色卡索引系统，拒绝使用数字调色工具；在画室东南角设立「灵感祭坛」，陈列着松果、锈铁片和祖父的怀表",\n      "Fitness": "每天黎明前在美术学院后山独自行走，声称这个时段的露水气息有助于校准色彩感知"\n    },\n    "Content_Interests": [\n      "关注冷门后现代戏剧的舞台设计",\n      "系统研究荣格分析心理学与超现实主义的关系",\n      "收集各国艺术疗愈中心的建筑平面图"\n    ]\n  }\n}\n\n根据用户配置生成此问题的1个完全个性化AI助手回答。回答应反映对配置中定义的用户特征、偏好和习惯的理解。需要结合多个字段信息来进行个性化的回答。\n\n以下是用户问题：\n有什么推荐的经典音乐可以搭配我的油画创作？\n\n生成1个个性化回答：\n', 'role': 'user'}],

#     save_file = os.path.join(self.save_path, f"{self.select_fun}_responses_{self.save_counter}.json")
#         self.save_counter += 1
#         with open(save_file, 'w', encoding='utf-8') as f:
#             json.dump(save_data, f, ensure_ascii=False, indent=2)

# batch[0]
# 如果达到最长的长度， 那么筛掉， 接下来calculate_ngram_repetition_rate, 按照 2gram 0.2， 3gram 0.12 的规则去进行过滤
# 如果筛选后小于bsz， 那么打印出警告， 反正凑个数继续跑吧
# 如果筛选后大于bsz， 那么直接丢掉后面的？


def check_position_ids_and_mask(final_prompts, final_responses, final_position_ids, final_attention_mask, 
                              original_batch_size, batch_size, select_fun, new_non_tensor_batch):
    """
    检查 position_ids 和 attention_mask 的正确性
    
    Args:
        final_prompts: 最终的提示张量
        final_responses: 最终的响应张量
        final_position_ids: 最终的 position_ids 张量
        final_attention_mask: 最终的 attention_mask 张量
        original_batch_size: 原始批次大小
        batch_size: 当前批次大小
        select_fun: 选择模式
        new_non_tensor_batch: 非张量批次数据
    """
    # 1. 检查维度
    print("=== 维度检查 ===")
    print(f"final_prompts shape: {final_prompts.shape}")
    print(f"final_responses shape: {final_responses.shape}")
    print(f"final_position_ids shape: {final_position_ids.shape}")
    print(f"final_attention_mask shape: {final_attention_mask.shape}")
    
    # 2. 检查 position_ids 的连续性
    print("=== position_ids 检查 ===")
    response_length = final_responses.size(1)
    prompt_length = final_prompts.size(1)
    
    # 检查 prompt 部分的 position_ids
    prompt_position_ids = final_position_ids[..., :prompt_length]
    print(f"prompt position_ids 范围: {prompt_position_ids.min()} - {prompt_position_ids.max()}")
    
    # 检查 response 部分的 position_ids
    response_position_ids = final_position_ids[..., prompt_length:]
    expected_delta = torch.arange(1, response_length + 1, device=final_position_ids.device)
    if final_position_ids.dim() == 3:  # qwen2vl mrope
        expected_delta = expected_delta.view(1, 1, -1).expand(batch_size, 3, -1)
    else:
        expected_delta = expected_delta.unsqueeze(0).expand(batch_size, -1)
    
    # 验证 response 部分的 position_ids 是否正确递增
    last_prompt_position = final_position_ids[..., prompt_length-1:prompt_length]
    expected_response_positions = last_prompt_position + expected_delta
    position_diff = torch.abs(response_position_ids - expected_response_positions)
    print(f"position_ids 最大偏差: {position_diff.max()}")
    
    # 3. 检查 attention_mask
    print("=== attention_mask 检查 ===")
    unique_values = torch.unique(final_attention_mask)
    print(f"attention_mask 唯一值: {unique_values}")
    
    # 检查 prompt 和 response 部分的 attention_mask
    prompt_mask = final_attention_mask[:, :prompt_length]
    response_mask = final_attention_mask[:, prompt_length:]
    print(f"prompt mask 中 1 的比例: {prompt_mask.float().mean():.2f}")
    print(f"response mask 中 1 的比例: {response_mask.float().mean():.2f}")
    
    # 4. 检查序列长度
    print("=== 序列长度检查 ===")
    print(f"prompt 长度: {prompt_length}")
    print(f"response 长度: {response_length}")
    
    # 5. 检查设备一致性
    print("=== 设备一致性检查 ===")
    print(f"final_prompts device: {final_prompts.device}")
    print(f"final_responses device: {final_responses.device}")
    print(f"final_position_ids device: {final_position_ids.device}")
    print(f"final_attention_mask device: {final_attention_mask.device}")
    
    # 6. 检查不同选择模式下的维度变化
    print("=== 选择模式维度检查 ===")
    print(f"原始 batch_size: {original_batch_size}")
    print(f"当前 batch_size: {batch_size}")
    print(f"选择模式: {select_fun}")
    
    # 7. 添加断言检查
    assert torch.all((final_attention_mask == 0) | (final_attention_mask == 1)), \
        "attention_mask 包含非 0/1 值"
    
    assert position_diff.max() < 1e-5, \
        f"position_ids 不连续，最大偏差: {position_diff.max()}"
    
    # 8. 检查特殊位置
    print("=== 特殊位置检查 ===")
    print(f"序列开始 position_ids: {final_position_ids[..., 0]}")
    print(f"序列开始 attention_mask: {final_attention_mask[:, 0]}")
    print(f"序列结束 position_ids: {final_position_ids[..., -1]}")
    print(f"序列结束 attention_mask: {final_attention_mask[:, -1]}")
    
    # 9. 检查拼接操作
    if select_fun in ['only_edit', 'random', 'select_best']:
        assert final_position_ids.shape[0] == original_batch_size, \
            f"only_edit/random/select_best 模式下 batch_size 错误: {final_position_ids.shape[0]} != {original_batch_size}"
    elif select_fun == 'max_distance':
        assert final_position_ids.shape[0] == original_batch_size, \
            f"max_distance 模式下 batch_size 错误: {final_position_ids.shape[0]} != {original_batch_size}"
    else:
        assert final_position_ids.shape[0] == original_batch_size * 2, \
            f"其他模式下 batch_size 错误: {final_position_ids.shape[0]} != {original_batch_size * 2}"
    
    # 10. 检查 non_tensor_batch 的维度
    print("=== non_tensor_batch 检查 ===")
    for key, value in new_non_tensor_batch.items():
        if isinstance(value, np.ndarray):
            print(f"{key} shape: {value.shape}")
            print(f"{key} length: {len(value)}")
    
    return True
