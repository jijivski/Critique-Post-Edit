# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
# 假设你的原始输入是 content_dict（dict），包含 'content' 字段
import re
from collections import defaultdict
import os
import datetime

import torch

from verl import DataProto


def extract_user_config_and_question(content_dict):
    """
    extract user config and question from content
    """
    content = content_dict['content']
    # 1. extract user config
    config_match = re.search(
        r'当前用户配置：\s*([\{\[].*?[\}\]])\s*根据用户配置',
        content, re.DOTALL)
    if not config_match:
        print("无法提取用户配置。原始内容：", content)
        raise ValueError('no user config extracted')
    user_config_str = config_match.group(1)
    # 2. json normalize (if format is ok)
    try:
        user_config = json.loads(user_config_str)
        user_config_str = json.dumps(user_config, ensure_ascii=False, indent=2)
    except Exception as e:
        # breakpoint()
        assert False, f"user config json cannot be parsed: {e}"
        print("warning: user config json cannot be parsed, keep original string.", e)

    pattern1 = r"以下是用户问题：\s*(.*?)\s*\n\n生成1个个性化回答"  # V7
    pattern2 = r"以下是用户问题：\s*(.*?)\s*\n\n生成1个完全个性化回答"  # V4

    q_match = re.search(pattern1, content, re.DOTALL)
    if not q_match:
        q_match = re.search(pattern2, content, re.DOTALL)


    if not q_match:
        print("no question extracted. original content:", content)
        # breakpoint()
        assert False, f"question cannot be extracted: {e}"
        raise ValueError('no question extracted')
    question = q_match.group(1).strip()

    return user_config_str.strip(), question.strip()


def extract_question_from_text(text):
    """extract question from text"""
    try:
        pattern1 = r"以下是用户问题：\s*(.*?)\s*\n\n生成1个个性化回答"  # V7
        pattern2 = r"以下是用户问题：\s*(.*?)\s*\n\n生成1个完全个性化回答"  # V4

        match = re.search(pattern1, text, re.DOTALL)
        if not match:
            match = re.search(pattern2, text, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return None
    except Exception as e:
        print(f"error extracting question: {e}")
        return None


'\n你需要模拟AI助手对用户问题的回答。请根据提供的用户性格配置，以AI助手的身份回答这位特定用户的问题。\n回答应该考虑用户的性格特点、价值观、兴趣爱好和使用习惯，提供最适合该用户的回答。\n\n重要规则：\n1. 回答应该符合AI助手的语气和风格，专业、友好且有帮助\n2. 考虑用户的偏好和特点，提供个性化的回答\n3. 回答应该能够满足用户的需求，不要太显示的展示出字段信息，要根据已有的Persona Config来推理用户可能想要的个性化回答，而不是简单的结合字段信息。\n4. 回答应该直接解决用户的问题，同时体现对用户偏好的了解\n\n注意，输出的回答不要过长，不要提及太具体的字段信息！\n注意，输出的回答要基于事实，不要强行和字段关联，要自然衔接，不要强行关联！\n\n当前用户配置：\n{\n  "Name": "王婉婷",\n  "Demographics": {\n    "Age": 32,\n    "Gender": "男",\n    "Nationality": "中国",\n    "Language": [\n      "中文",\n      "英文"\n    ],\n    "Career_Information": "上海某金融公司投资分析师"\n  },\n  "Personality": {\n    "Extraversion_or_Introversion": "E",\n    "Sensing_or_Intuition": "S",\n    "Thinking_or_Feeling": "F",\n    "Judging_or_Perceiving": "P",\n    "Values_and_Interests": [\n      "研究经济趋势",\n      "跑步和健身爱好者",\n      "喜欢读科幻小说",\n      "热衷于极限运动"\n    ]\n  },\n  "Preference": {\n    "Local_Life": {\n      "Food_Drinks": "工作日午餐在写字楼轻食餐厅解决，偏好高蛋白低碳水饮食；周末会探索上海新开的分子料理餐厅。随身携带自制蔬果汁，对有机食品有执念但讨厌牛油果",\n      "Entertainment_Shopping": "每月至少一次室内攀岩馆训练，常去静安寺商圈购买运动黑科技装备。偶尔在茑屋书店参加科幻作家签售会",\n      "Daily_Outings": "通勤坚持骑共享电单车，周末清晨在滨江骑行道进行30公里耐力训练"\n    },\n    "Travel": {\n      "Business_Trips": "出差必选带健身房的高端酒店，利用碎片时间体验当地特色运动项目（如哈尔滨出差尝试冰潜）",\n      "Leisure_Travel": "每年挑战一个极限旅行目的地（如勃朗峰速攀/巴厘岛冲浪），行前用三维地形图软件规划路线"\n    },\n    "Social": "在Keep上创建跑步挑战小组，线下组织金融圈户外运动俱乐部。参加科幻大会时主动担任分论坛主持人，但讨厌纯酒局社交",\n    "Productivity": {\n      "Study_Work": "用Python自动抓取全球运动品牌财报数据，Tableau看板实时更新投资组合健康度",\n      "Fitness": "佩戴专业级运动手表监测最大摄氧量，根据数据动态调整训练计划"\n    },\n    "Content_Interests": [\n      "深度追踪可穿戴设备评测",\n      "关注太空殖民主题硬科幻",\n      "研究运动营养学最新论文"\n    ]\n  }\n}\n\n根据用户配置生成此问题的1个完全个性化AI助手回答。回答应反映对配置中定义的用户特征、偏好和习惯的理解。需要结合多个字段信息来进行个性化的回答。\n\n以下是用户问题：\n你能给我一些关于“提高社交技巧以帮助职业发展的”项目名称吗？\n\n生成1个个性化回答：\n'

# 现在需要做一些调整, 
# list_of_dic=[{'content': '\n你需要模拟AI助手对用户问题的回答。请根据提供的用户性格配置，以AI助手的身份回答这位特定用户的问题。\n回答应该考虑用户的性格特点、价值观、兴趣爱好和使用习惯，提供最适合该用户的回答。\n\n重要规则：\n1. 回答应该符合AI助手的语气和风格，专业、友好且有帮助\n2. 考虑用户的偏好和特点，提供个性化的回答\n3. 回答应该能够满足用户的需求，不要太显示的展示出字段信息，要根据已有的Persona Config来推理用户可能想要的个性化回答，而不是简单的结合字段信息。\n4. 回答应该直接解决用户的问题，同时体现对用户偏好的了解\n\n注意，输出的回答不要过长，不要提及太具体的字段信息！\n注意，输出的回答要基于事实，不要强行和字段关联，要自然衔接，不要强行关联！\n\n当前用户配置：\n{\n  "Name": "王婉婷",\n  "Demographics": {\n    "Age": 32,\n    "Gender": "男",\n    "Nationality": "中国",\n    "Language": [\n      "中文",\n      "英文"\n    ],\n    "Career_Information": "上海某金融公司投资分析师"\n  },\n  "Personality": {\n    "Extraversion_or_Introversion": "E",\n    "Sensing_or_Intuition": "S",\n    "Thinking_or_Feeling": "F",\n    "Judging_or_Perceiving": "P",\n    "Values_and_Interests": [\n      "研究经济趋势",\n      "跑步和健身爱好者",\n      "喜欢读科幻小说",\n      "热衷于极限运动"\n    ]\n  },\n  "Preference": {\n    "Local_Life": {\n      "Food_Drinks": "工作日午餐在写字楼轻食餐厅解决，偏好高蛋白低碳水饮食；周末会探索上海新开的分子料理餐厅。随身携带自制蔬果汁，对有机食品有执念但讨厌牛油果",\n      "Entertainment_Shopping": "每月至少一次室内攀岩馆训练，常去静安寺商圈购买运动黑科技装备。偶尔在茑屋书店参加科幻作家签售会",\n      "Daily_Outings": "通勤坚持骑共享电单车，周末清晨在滨江骑行道进行30公里耐力训练"\n    },\n    "Travel": {\n      "Business_Trips": "出差必选带健身房的高端酒店，利用碎片时间体验当地特色运动项目（如哈尔滨出差尝试冰潜）",\n      "Leisure_Travel": "每年挑战一个极限旅行目的地（如勃朗峰速攀/巴厘岛冲浪），行前用三维地形图软件规划路线"\n    },\n    "Social": "在Keep上创建跑步挑战小组，线下组织金融圈户外运动俱乐部。参加科幻大会时主动担任分论坛主持人，但讨厌纯酒局社交",\n    "Productivity": {\n      "Study_Work": "用Python自动抓取全球运动品牌财报数据，Tableau看板实时更新投资组合健康度",\n      "Fitness": "佩戴专业级运动手表监测最大摄氧量，根据数据动态调整训练计划"\n    },\n    "Content_Interests": [\n      "深度追踪可穿戴设备评测",\n      "关注太空殖民主题硬科幻",\n      "研究运动营养学最新论文"\n    ]\n  }\n}\n\n根据用户配置生成此问题的1个完全个性化AI助手回答。回答应反映对配置中定义的用户特征、偏好和习惯的理解。需要结合多个字段信息来进行个性化的回答。\n\n以下是用户问题：\n你能给我一些关于“提高社交技巧以帮助职业发展的”项目名称吗？\n\n生成1个个性化回答：\n', 'role': 'user'}]
# # 示例调用
# user_config, question = extract_user_config_and_question(list_of_dic[0])
# print(user_config)
# print(question)






class BatchRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        # trajectories_str = []
        personas_str=[]
        questions_str=[]
        # prompt_str=""
        for i in range(len(data)):

            chat_dic=data.non_tensor_batch["raw_prompt"][i][0]
            persona_str, question_str = extract_user_config_and_question(chat_dic)

            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            # prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            personas_str.append(persona_str)
            questions_str.append(question_str)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # trajectory = prompt_str + response_str
            # prompts_str.append(prompt_str)
            responses_str.append(response_str)
            # trajectories_str.append(trajectory)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            questions_str=questions_str,
            personas_str=personas_str,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
