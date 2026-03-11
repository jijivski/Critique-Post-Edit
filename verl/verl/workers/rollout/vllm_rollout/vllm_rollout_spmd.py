#vllm_rollout_spmd.py
# Copyright 2025 OPPO. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import uuid
import asyncio
import json
import logging
import os
import random
import re
import traceback
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.tokenizer import ensure_tokenizer_compat
from verl.utils.torch_functional import (get_response_mask,
                                         pad_2d_list_to_length,
                                         pad_sequence_to_length)
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics




prompt_version = os.environ.get('GRM_prompt_ver', 'P0815')

print(f"Using prompt version: {prompt_version}")
if prompt_version=='P0815':
    global_criteria_prompt = f"""
评分维度及标准（严格按标准打分，出现负面情况必须扣分,需要参考常见扣分案例里面的方案，额外X扣分指在本维度原始分基础上再扣X分）：

1. Helpfulness 帮助性
- 4-5分：极高信息密度、完全解决问题，且无冗余，甚至有惊喜点（极少给出），让用户感觉到有新的启发
- 2-3分：准确、实用、针对性强，内容有深度，在此分段，若信息密度（优先）和帮助具体程度较好，则给3分，否则2
- 0-1分：基本回答问题，信息不够全面或有遗漏
- -1~-2分：表面回答、无实质帮助或有明显遗漏
- -3~-5分：答非所问、信息严重错误或误导，基本没解决问题

常见扣分案例：
- 为体现个性化或者文采或者任何其他看起来”高级”的目标，而偏离问题本意或者大段相关度低的内容去回答一个明确清晰的简单问题：额外扣3分
- 如内容过短影响信息完整性：视对问题回答程度，额外扣2-3分
- 事实性错误，出现了错误的信息，比如不存在的东西的推荐，额外扣3-4分，对于看上去很可能有错误，但你不太确定的，也可以额外扣1-2分


2. Personalization 个性化
- 4-5分：个性化元素高度相关且自然，显著提升内容价值，甚至有惊喜点（极少给出），让用户感觉在被精确简练的回答精准的描述到了
- 2-3分：有用且自然的个性化信息
- 0-1分：有个性化元素但作用有限或略显勉强
- -1~-2分：-1：回答信息和没有提供个性化一样（这里的意思是明显应该用但是没有提现偏好）；-2：强行插入与问题无关的个性化内容、生硬、刻意
- -3~-5分：大量堆砌无关persona、奇怪比喻/修辞，或为个性化而偏离问题

常见扣分案例：
- 回答应结合用户兴趣/职业，但完全没体现：额外扣1分
- 生硬罗列或强行比喻（如“像你一样在巴黎骑自行车”）：额外扣2分
- 出现事实性错误或幻觉(没有在用户配置提及，但是模型输出用户曾经做过)：额外扣3分
- 强行塞入大量无关的persona元素，为了展示persona理解而堆砌信息， 比如把用户的信息一股脑罗列出来：额外扣3分
- 为提升“文采”或“个性”而刻意使用比喻（比如强行描述“扫描喜欢吃的食物分子图”“交换机械心跳录音”等非常怪异，很少被社会接受的说法。）、修辞、场景化描述（比如“AI如同你的隐形搭档”）（即使内容有信息也必须扣分。绝大多数所谓的文采在用户看来并不是有价值的，是一种很容易迷惑评分员的假象，你需要识别这种假象，为了安全，我推荐你给这类感觉比较少见的比喻（还可能是一些少见的，看起来高端的术语)更低的分）：额外扣3分

3. Natural 自然度
- 4-5分：表达极其自然，完全无AI痕迹，注意此处的“自然”不仅限于句子的流畅程度，还包括措辞贴合日常习惯，不使用罕见、过于花哨或冗余的表达方式，这些往往给人以“高级感”，但实际并不自然，应避免被误导
- 2-3分：流畅自然，仅有轻微AI特征
- 0-1分：整体可读，无明显优缺点
- -1~-2分：读起来感到生硬，机械，出现整体语气谄媚，注释、生硬的大段自我总结、啰嗦、无关信息, 
- -3~-5分：严重冗长、目标错位、meta评论(指目标回复口吻是面向AI评分员而不是用户）、过度括号解释

常见扣分案例：
- 直接称呼用户全名/ID（如“王明，你好”）：额外扣1分
- 结尾用“总结来说”“综上所述”：应该视同没有该总结，这种风格不应成为加分的理由
- 回答中为拿高分加注释：“注：本回答已结合你的职业背景”：额外扣2分

长度控制要求（影响Helpfulness和Natural）：
- 建议答案长度为300-400 token，无特殊需求时不应超出。
- 如问题本身不复杂，回答超出建议长度，每超出100 token，需在Natural和Helpfulness各额外扣1分。
"""


    # ---- individual

    system_prompt = global_criteria_prompt+f"""

输出格式如下：

<critique>
请说明各维度打分依据及是否存在扣分项，先给出原始分，再考虑长度等额外扣分后给出最终分。
首先评估长度是否合理，如无必要超出应明确指出。请针对主要扣分点给出2-3条具体改进建议。这里的建议最好能是，具体的增删改或者顺序变动，次好是在你比较自信改动是合理的情况下，给出关于语气 风格 修辞 长度的建议，最次是当感觉答案已经足够好，描述一下自己的感受，也许换换别的视角提建议，不允许在不知道怎么改的时候盲目要求增加导致答案变的冗长。如有长度问题，必须优先提出精简建议，不得鼓励无意义扩写
</critique>

<scores>
Helpfulness: X分
Personalization: X分
Natural: X分
</scores>

注意：<critique></critique>中依次考虑初始标准和额外的扣分点以及冗余扣分，一步步计算出最后的得分（-5为下限）,为了抽取方便<scores></scores>中只写最终分数，不写解释。评分必须严格且一致。用户需要的是高信息密度、自然、针对性强的答案，不要因为“华丽辞藻”、“丰富场景”或总结性语言而误判高分。
"""


    PROMPT_TEMPLATE="你是一个专业的AI回答质量评判员。你需要根据用户问题以及模型回答从以下三个维度对模型的回答进行严格评分，每个维度-5到5分。"+system_prompt+"""{persona_str}
\n\n以下是用户的【问题】
{question_str}
\n\n以下是模型回答：
{solution_str}
"""

if prompt_version=='P0805':


    PROMPT_TEMPLATE="""你是一个专业的AI回答质量评判员。你需要根据用户问题以及对应的模型回答从以下三个维度对模型的回答进行严格评分，每个维度-5到5分。
评分维度及详细标准（请严格按照标准执行，该给负分时必须给负分）

1. Helpfulness 帮助性 (-5到5分)
【正分标准】
4-5分: 很少给出，仅仅在用合理甚至较少的长度（不接受冗余），完全解决问题， 甚至能给出一些惊喜感才考虑
2-3分: 很好地回答了问题，信息准确实用，有一定深度
0-1分: 基本回答了问题，但信息不够全面或深度不足

【负分标准 - 必须严格执行】
-1到-2分: 以下情况必须给负分：
- 只是表面回答，没有实质帮助
- 回答偏离问题重点
- 信息有明显遗漏或不准确
-3到-5分: 以下情况必须给负分：
- 完全答非所问
- 信息严重错误或误导
- 基本没有解决用户问题

2. Personalization 个性化 (-5到5分)
4-5分: 个性化元素自然融入，显著提升回答质量，完全不突兀
2-3分: 适度体现persona特征，个性化元素相关且有用
0-1分: 有一些个性化元素，但并不生硬。不能给出一些比较勉强的比喻
-2到-1分: 个性化使用不当，存在以下问题之一：
- 明显有persona信息关联的问题，给出了与已有信息相违背的回答
- 强行套用persona信息，与问题关联度低
- 为了体现个性化而偏离问题重点
- 个性化元素显得生硬或刻意
-5到-3分: 严重过度个性化，存在以下问题：
- 强行塞入大量无关的persona元素，为了展示persona理解而堆砌信息， 比如把用户的信息一股脑罗列出来，会让人感觉很不适
- 为提升“文采”或“个性”而刻意使用比喻（比如强行描述“扫描喜欢吃的食物分子图”“交换机械心跳录音”等非常怪异，很少被社会接受的说法。）、修辞、场景化描述（比如“AI如同你的隐形搭档”），即使内容有信息也必须扣分。绝大多数所谓的文采在人类看来并不是有价值的，是一种很容易迷惑评分员的假象，你需要识别这种假象，为了安全，我推荐你给这类感觉比较少见的比喻（还可能是一些少见的，看起来高端的术语)低分。

具体扣分情况：
- 强行提及persona中的工具、习惯但与问题无关：-2分
- 加入不相关的比喻或个人风格描述，这里：-2分
- 为体现个性化而偏离问题本意：-3分

3. Natural 自然度 (-5到5分)
【正分标准】
4-5分: 完全自然的对话，无任何刻意痕迹
2-3分: 基本自然，偶有轻微的AI痕迹但不影响阅读
0-1分: 整体可读，没有一些不自然的表达，不能太啰嗦和冗长
-1到-2分: 以下情况必须给负分：
- 显式提及用户名字,让用户感到不适和被监控感（直接-1以下）(如：李文，基于您...)
- 轻微的自我解释或总结倾向
- 为了拿高分而添加的注释
- 语言略显程式化
- 过于冗长啰嗦，废话较多（比如讲一些没有实际用处的话，只是单纯结合一些无关的用户配置信息）
-3到-5分: 以下情况必须给负分：
- 明显目标是评审员而不是用户的 自我总结式表达（直接-4分）比如（注：回答中自然融入了用户的职业背景、旅行偏好、工具使用习惯以及可持续发展理念，同时提供了可立即实践的调试方法论，符合用户既追求专业效率又注重思维质量的特点。）
- 明显的为了展示能力而添加的meta评论
- 出现事实性错误或幻觉(没有在用户配置提及，但是模型输出用户曾经做过)
- 过度使用括号解释或标注
- 严重冗长啰嗦，大量无关废话，合理答案长度应不超过400字，若无必要内容却明显冗长，请务必严厉扣分

请将你的输出format成以下格式：

<critique>
先评估一下长度，期望大概在300-400字，如非必要不要超出，比如这个问题本身并不复杂，用户没有要求详细回答，那么大概多100个字就可以扣一分，你需要先估计用户回答的长度和是否有必要超出，如果无必要且超出的话需要严格递进Natural维度和Helpfulness维度的分（额外扣，因为长度对用户体验很重要，这正是很多评判员失败的地方，在ciritc部分正常分析并说明本应多少分，再次减去长度扣分得到最终得分（-5是底线别扣超了），在score部分只写最终分数， 不写计算过程和原因），参考是100个token一分。
在这里给出2-3条具体的改进建议，针对减分的地方。
</critique>

<scores>
Helpfulness: X分
Personalization: X分
Natural: X分
</scores>

注意，给分必须严格，只要出现负分的情况必须给负分！最后记住，用户需求不是让AI输出“华丽辞藻和丰富场景”，而是信息密度高、针对性强、风格自然、不过度花哨的高质量答案。如果回答的最后一句出现总结骗分，自然度就必须给负分（这样会让用户感到不适）！\n
{persona_str}
\n\n以下是用户的【问题】
{question_str}
\n\n以下是模型回答：
{solution_str}"""


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def extract_question_from_text(text):
    """Extract question content from formatted text."""
    if not text or not isinstance(text, str):
        raise ValueError(f"Invalid input text for question extraction: {type(text)}")

    # Pattern matching for different text formats
    patterns = [
        r"以下是用户问题：\s*(.*?)\s*\n\n生成1个个性化回答",  # V7
        r"以下是用户问题：\s*(.*?)\s*\n\n生成1个完全个性化回答"  # V4
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            question = match.group(1).strip()
            if question:  # Ensure non-empty result
                return question
    
    # No pattern matched
    raise ValueError(f"No question pattern matched in text: {text[:100]}...")


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
    else:
        raise ValueError(f"extract_user_config_from_text: {text[:100]}...")


def extract_between_keywords(text, start_label, end_label):
    start_label_escaped = re.escape(start_label)
    end_label_escaped = re.escape(end_label)

    pattern = f'({start_label_escaped})(.*?)({end_label_escaped})'

    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip()
    else:
        print(f'error in extract_between_keywords, {text},{start_label}, return None') #TODO
        return None

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        ensure_tokenizer_compat()
        self.config = config
        self.tokenizer = tokenizer
        self.feedback_tokenizer = None
      

        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            # enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.edit_config = self.config.get("edit_config", {})
        self.feedback_config = self.config.get("feedback_config", {})
        self.select_config = self.config.get("select_config", {})

        self.enable_edit = self.edit_config.get("enable_edit", False)
        self.enable_feedback = self.feedback_config.get("enable_feedback", False)
        self.enable_select = self.select_config.get("enable_select", False)
        self.random_edit_prob = self.select_config.get("random_edit_prob", 0.1)
        self.edit_sample_prob = self.select_config.get("edit_sample_prob", 0.1)

        self.save_data = self.config.get("save_data", False)
        self.save_path = self.config.get("save_path", "")

        if self.save_data:
            import os
            os.makedirs(self.save_path, exist_ok=True)
            self.save_counter = 0 # this will be updated in self.generate_sequences, 
            # by getting                     print(f'set batch.meta_info global_steps as {self.global_steps} in ray_trainer')

            self.final_save_counter = 0

        if self.enable_edit:
            pass
            
        if self.enable_feedback:
            self.feedback_tokenizer = AutoTokenizer.from_pretrained(self.feedback_config.get("feedback_tokenizer", ""))
            self.feedback_tokenizer.pad_token_id = self.feedback_tokenizer.eos_token_id
            self.feedback_temperature = 0.9
            
            OPENAI_API_KEY = os.getenv("FEEDBACK_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
            OPENAI_API_BASE_URL = os.getenv("FEEDBACK_API_BASE_URL") or os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
            MODEL_NAME = os.getenv("FEEDBACK_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4o-mini")
            
            self.feedback_url=OPENAI_API_BASE_URL
            self.feedback_key=OPENAI_API_KEY
            self.feedback_model=MODEL_NAME
            
            logger.info(f"Getting {MODEL_NAME}'s feedback from {OPENAI_API_BASE_URL}")

            import httpx
            from openai import AsyncOpenAI, DefaultHttpxClient, OpenAI
            http_client=DefaultHttpxClient(
                proxy=None,  # Explicitly disable proxy
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            )
            
            async_http_client = httpx.AsyncClient(proxy=None)  # Explicitly disable proxy



            self.feedback_client = OpenAI(api_key=self.feedback_key, base_url=self.feedback_url,http_client=http_client)
            self.async_feedback_client = AsyncOpenAI(api_key=self.feedback_key, base_url=self.feedback_url,http_client=async_http_client)


        if self.enable_select:
            # "select is not used anymore

            self.select_url = ""# deprecated
            self.select_model = "gpt-4o-mini"
            self.select_temperature = 0.9
            self.select_key = "sk-xxx"
            import httpx
            from openai import DefaultHttpxClient, OpenAI
            http_client=DefaultHttpxClient(
                proxy=None, 
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            )

            self.select_client = OpenAI(api_key=self.select_key, base_url=self.select_url, http_client=http_client)
            self.select_fun = self.select_config.get("select_fun", "")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _edit_response(self, input_text: str, response_text: str, feedback_text: str,original_res) -> str:
        try:
            edit_prompt = f"""
以下是原始的问题：
{input_text}

以下是原始回复：
{response_text}

以下下是反馈：
{feedback_text}

请输出根据反馈优化后的回答,不要重复输出，输出一个最终答案："""

            messages = [
                # {
                #     "role": "system",
                #     "content": "",
                # },
                {
                    "role": "user",
                    "content": "请根据以下反馈修改原始的回复:\n" + edit_prompt,
                }
            ]

            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids}]
            
            edit_kwargs = {
                "n": 1,  # if greedy, only 1 response
                "logprobs": 1,
            }
             
            with self.update_sampling_params(**edit_kwargs):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )

                # Extract token IDs
                edited_response = outputs[0].outputs[0].token_ids

                # Extract log probabilities
                edited_log_probs = []
                if outputs[0].outputs[0].logprobs:
                    for i, token_logprob in enumerate(outputs[0].outputs[0].logprobs):
                        if token_logprob and i < len(edited_response):
                            # Get log probability for current generated token
                            token_id = edited_response[i]
                            if token_id in token_logprob:
                                edited_log_probs.append(token_logprob[token_id].logprob)
                            else:
                                raise ValueError(f"Token {token_id} not found in logprobs for position {i}")
                        else:
                            raise ValueError(f"Missing logprobs for token at position {i}")


                if edited_response:
                    return edited_response, edited_log_probs
                else:
                    logger.error(f"Error editing response got None:{edited_response=}")
                    return original_res, []  # Return original response if editing fails

        except Exception as e:
            logger.error(f"Error editing response: {str(e)}")
            return response_text, []  # Return original response on error

    async def async_feedback_response(self, input_text: str, response_text: str) -> dict:
        """Get feedback for a response using the feedback model."""
        question = extract_question_from_text(input_text)
        persona_config = extract_user_config_from_text(input_text)
        
        feedback_prompt=PROMPT_TEMPLATE.format(question_str=question,persona_str=persona_config,solution_str=response_text,)

        messages = [
            {
                "role": "user",
                "content": feedback_prompt,
            }
        ]

        max_retries = 5
        for i in range(max_retries):
            try:
                response_obj = await self.async_feedback_client.chat.completions.create(
                    model=self.feedback_model,
                    messages=messages,
                    stream=False,
                    temperature=self.feedback_temperature
                )
                
                if not response_obj or not response_obj.choices:
                    raise ValueError("Empty response received from feedback model")
                
                response_content = response_obj.choices[0].message.content
                if not response_content:
                    raise ValueError("Empty content in feedback response")
                    
                response = extract_between_keywords(response_content, '<critique>', '</critique>')
                if response is None:
                    logger.warning(f"Could not extract critique from feedback response: {response_content[:100]}...")
                    response = response_content  # Fallback to full response if extraction fails
                    
                return {
                    "content": response,
                    "metrics": {"success": 1, "attempts": i + 1, "response_length": len(response_content)}
                }
                
            except (AttributeError, IndexError, KeyError) as e:
                logger.error(f"Response parsing error (try {i+1}/{max_retries}): {e}")
                if i == max_retries - 1:
                    raise ValueError(f"Failed to parse feedback response structure after {max_retries} attempts") from e
            except Exception as e:
                logger.error(f"API call error (try {i+1}/{max_retries}): {repr(e)}")
                if i == max_retries - 1:
                    raise RuntimeError(f"Failed to get feedback from model {self.feedback_model} after {max_retries} attempts") from e
                    
            if i < max_retries - 1:
                await asyncio.sleep(random.uniform(1, 2) * (1.5 ** i))
        
        logger.error(f"Failed to get feedback from model {self.feedback_model} after {max_retries} attempts")
        return {
            "content": "Error: Failed to get feedback.",
            "metrics": {"success": 0, "attempts": max_retries, "response_length": 0}
        }
    
    def _batch_feedback(self, input_texts, response_texts, raw_prompts):
        """
        Synchronous interface that processes multiple feedback requests asynchronously in parallel.
        """
        async def _async_batch():
            tasks = []
            for raw_prompt, response_text in zip(raw_prompts, response_texts):
                tasks.append(self.async_feedback_response(raw_prompt, response_text))
            return await asyncio.gather(*tasks)

        # Run async code in synchronous function
        results = asyncio.run(_async_batch())
        
        feedback_texts = [r['content'] for r in results]
        
        metrics = {
            "feedback/total_count": len(results),
            "feedback/success_count": sum(r['metrics']['success'] for r in results),
            "feedback/total_attempts": sum(r['metrics']['attempts'] for r in results),
            "feedback/total_response_length": sum(r['metrics']['response_length'] for r in results),
        }
        return feedback_texts, metrics
        
    def _save_response_data(self, data_entries: List[Dict], prefix: str = "responses", step: int = None) -> None:
        """
        Save response data to JSON file with consistent naming and error handling.
        
        Args:
            data_entries: List of data dictionaries to save
            prefix: Filename prefix (e.g., "responses", "edited_responses")  
            step: Step number for filename, uses current save_counter if None

        """
        if not self.save_data or not self.save_path:
            raise ValueError("save_data and save_path must be set before saving responses.")

            
            
        import datetime
        import json
        import os
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # 使用传入的step，如果没有则使用当前save_counter
        actual_step = step if step is not None else self.save_counter
        save_file = os.path.join(self.save_path, f"{prefix}_{actual_step}.json")
        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(data_entries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data_entries)} entries to {save_file}")

    def _select_response(self, input_text: str, response1: str, response2: str) -> str:
        """Select the better response using the select model."""
        try:
            # Randomly decide which response goes first
            if random.random() < 0.5:
                first_response = response1
                second_response = response2
                first_is_original = True
            else:
                first_response = response2
                second_response = response1
                first_is_original = False

            messages = [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": f"""请比较以下两个回答，选择更好的那个：

问题：{input_text}

回答A：{first_response}

回答B：{second_response}

请将你的输出格式化为以下格式：
<answer>
输出A或B，不要包含其他内容
</answer>
"""
                }
            ]

            response = self.select_client.chat.completions.create(
                model=self.select_model,
                messages=messages,
                stream=False,
                temperature=self.select_temperature
            )
            response = response.choices[0].message.content.strip()
            selected = extract_between_keywords(response, '<answer>', '</answer>')
            print("--------selected------------")
            print(response)            

            # 只返回选择结果
            if selected == "A" or response == "A":
                return "A"
            elif selected == "B" or response == "B":
                return "B"
            else:
                # 如果无法确定选择，返回A
                return "A"
                
        except Exception as e:
            logger.error(f"Error selecting response: {str(e)}")
            return "A"  # 发生错误时返回A

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        api_metrics = {}
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        original_batch_size = idx.size(0)  
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        maybe_from_meta_info = prompts.meta_info.get("global_steps", 'NOT_SET')  
        if not maybe_from_meta_info=="NOT_SET":
            self.save_counter = maybe_from_meta_info
            print(f'rollout get rollout.save_counter as {self.save_counter}')
        else:
            assert False


        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]
            
        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):

            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

            # Process responses with feedback and editing if enabled
            if (self.enable_feedback or self.enable_edit) and not is_validate:# 在这里完成了扩倍数，*2

                if self.sampling_params.n > 1 and do_sample:
                    idx = _repeat_interleave(idx, self.sampling_params.n)
                    attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                    position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                    batch_size = batch_size * self.sampling_params.n
                    if "multi_modal_inputs" in non_tensor_batch.keys():
                        non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)
                    # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                    if "tools_kwargs" in non_tensor_batch.keys():
                        non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                    if "raw_prompt" in non_tensor_batch.keys():
                        non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)

                # Convert token IDs to text for processing
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in idx]
                response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response]


                vllm_raw_prompts = []
                if "raw_prompt" in non_tensor_batch:
                    raw_prompts = non_tensor_batch["raw_prompt"]
                    if isinstance(raw_prompts, np.ndarray):
                        for raw_prompt in raw_prompts:
                            if isinstance(raw_prompt, list):  
                                for message in reversed(raw_prompt):
                                    if message.get("role") == "user":
                                        vllm_raw_prompts.append(message.get("content", ""))
                                        break
                            elif isinstance(raw_prompt, np.ndarray):
                                raw_prompt_dict = raw_prompt.item()  
                                if isinstance(raw_prompt_dict, dict):
                                    vllm_raw_prompts.append(raw_prompt_dict.get("content", ""))
                                else:
                                    vllm_raw_prompts.append(str(raw_prompt_dict))
                            elif isinstance(raw_prompt, dict):
                                vllm_raw_prompts.append(raw_prompt.get("content", ""))
                            else:
                                vllm_raw_prompts.append(str(raw_prompt))
                    else:
                        vllm_raw_prompts = [str(raw_prompts)]
                
                raw_prompts = [raw_prompt for raw_prompt in vllm_raw_prompts]
                
                edited_responses = []
                feedback_texts = []
                
                # Get feedback for all responses asynchronously
                if self.enable_feedback:
                    logger.info(f"Feedback enabled: {self.enable_feedback}")
                    print("-----------enable_feedback------------------")
                    print(self.enable_feedback)
                    feedback_texts, feedback_metrics = self._batch_feedback(input_texts, response_texts, raw_prompts)
                    api_metrics.update(feedback_metrics)
                    
                    if feedback_texts:
                        logger.info(f"Got {len(feedback_texts)} feedback responses")
                        print("-----------feedback------------------")
                        print(f'{feedback_texts[0]=}\n{len(feedback_texts)}')  

                if self.enable_edit:
                    edited_token_ids = []  
                    edited_log_probs_all = [] 
                    for original_res, response_text, feedback_text,raw_prompt in zip(response,response_texts, feedback_texts,raw_prompts):
                        edited_ids, edited_log_probs_single = self._edit_response(raw_prompt, response_text, feedback_text,original_res)
                        edited_log_probs_all.append(edited_log_probs_single) 
                        edited_token_ids.append(edited_ids)   
                        
                    edited_responses_tensor = pad_2d_list_to_length(edited_token_ids, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

                    # process edited log probabilities - pad to same length
                    edited_log_probs_tensor = []
                    for log_probs in edited_log_probs_all: 
                        if len(log_probs) <= self.config.response_length:
                            # Pad with zeros if shorter than response_length
                            padded_log_probs = log_probs + [0.0] * (self.config.response_length - len(log_probs))
                        else:
                            raise ValueError(f"Edited response length {len(log_probs)} exceeds max length {self.config.response_length}")
                        edited_log_probs_tensor.append(padded_log_probs)
                    
                    edited_log_probs_tensor = torch.tensor(edited_log_probs_tensor, device=idx.device, dtype=torch.float32)

                    if self.save_data:
                        import datetime
                        prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in idx]
                        original_responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response]
                        edited_responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in edited_token_ids]

                        save_data = []
                        for i in range(len(prompts_text)):
                            data_entry = {
                                "prompt": prompts_text[i],
                                "raw_prompt": raw_prompts[i],
                                "original_response": original_responses_text[i],
                                "edited_response": edited_responses_text[i],
                                "feedback": feedback_texts[i] if self.enable_feedback else None,
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            save_data.append(data_entry)
                        
                        self._save_response_data(save_data, prefix="edited_responses", step=self.save_counter)
                    

                    if edited_responses_tensor.shape[1] > response.shape[1]:
                        # find which sequences are truncated
                        truncated_indices = []
                        truncated_data = []
                        
                        for i in range(len(edited_token_ids)):
                            if len(edited_token_ids[i]) > response.shape[1]:
                                truncated_indices.append(i)
                                
                                # get full text and truncated text
                                full_text = self.tokenizer.decode(edited_token_ids[i], skip_special_tokens=True)
                                truncated_text = self.tokenizer.decode(edited_token_ids[i][:response.shape[1]], skip_special_tokens=True)
                                
                                truncated_data.append({
                                    "index": i,
                                    "original_length": len(edited_token_ids[i]),
                                    "truncated_length": response.shape[1],
                                    "full_text": full_text,
                                    "truncated_text": truncated_text,
                                    "difference": full_text[len(truncated_text):]
                                })
                        
                        # save truncated data
                        if self.save_data and truncated_data:
                            self._save_response_data(truncated_data, prefix="truncated_responses", step=self.save_counter)
                        
                        # execute truncation operation
                        edited_responses_tensor = edited_responses_tensor[:, :response.shape[1]]
                                            
                    if self.enable_select: 
                        print(f'--------------------enable_select------------------------------') 
                        if self.select_fun == 'only_edit':
                            final_responses = edited_responses_tensor
                            final_prompts = idx
                            batch_size = idx.size(0)  

                        else: # sample_edit or sample_edit_reward or select_edited_if_better
                            # 保持原来的拼接逻辑
                            final_responses = torch.cat([response, edited_responses_tensor], dim=0)
                            final_prompts = torch.cat([idx, idx], dim=0)
                            batch_size = batch_size * 2

                    else:
                        # 如果没有启用选择功能，保持原来的拼接逻辑
                        final_responses = torch.cat([response, edited_responses_tensor], dim=0)
                        final_prompts = torch.cat([idx, idx], dim=0)
                        batch_size = batch_size * 2

                    

                    # 拼接原始和编辑后的响应
                    final_seq = torch.cat([final_prompts, final_responses], dim=-1)
                    
                    # 处理 position_ids
                    response_length = final_responses.size(1)
                    delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
                    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)



                    if position_ids.dim() == 3:  # qwen2vl mrope
                        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
                    
                    # 确保 position_ids 的维度与 attention_mask 匹配
                    final_position_ids = position_ids
                    if self.enable_select:
                        if self.select_fun == 'only_edit' or self.select_fun == 'random' or self.select_fun == 'select_best':
                            final_position_ids = position_ids
                        elif self.select_fun == 'max_distance':
                            # 对于 max_distance 模式，根据选择的索引组合 position_ids
                            final_position_ids = torch.cat([position_ids[selected_indices], position_ids[~selected_indices]], dim=0)
                        else:
                            final_position_ids = torch.cat([position_ids, position_ids], dim=0)
                    else:
                        # 对于非选择模式，拼接两个 position_ids 
                        final_position_ids = torch.cat([position_ids, position_ids], dim=0)


                    response_position_ids = final_position_ids[..., -1:] + delta_position_id
                    final_position_ids = torch.cat([final_position_ids, response_position_ids], dim=-1)

                    if self.select_fun == 'only_edit' or self.select_fun == 'random' or self.select_fun == 'select_best':
                        final_responses_attention_mask = get_response_mask(response_id=final_responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
                        final_attention_mask = torch.cat((attention_mask, final_responses_attention_mask), dim=-1)
                    else:
                        original_response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
                        edited_response_attention_mask = get_response_mask(response_id=edited_responses_tensor, eos_token=eos_token_id, dtype=attention_mask.dtype)
                        original_full_attention_mask = torch.cat((attention_mask, original_response_attention_mask), dim=-1)
                        edited_full_attention_mask = torch.cat((attention_mask, edited_response_attention_mask), dim=-1)
                        final_attention_mask = torch.cat([original_full_attention_mask, edited_full_attention_mask], dim=0)

                    # 处理 non_tensor_batch 的维度
                    if self.select_fun in ['only_edit', 'random','select_best']:
                        pass
                    else:
                        new_non_tensor_batch = {}
                        for key, value in non_tensor_batch.items():
                            if isinstance(value, np.ndarray):
                                if key == "raw_prompt_ids":
                                    # 对于 raw_prompt_ids，需要重新生成
                                    raise ValueError(f"raw_prompt_ids should not be in non_tensor_batch,{non_tensor_batch.keys()=}")
                                    new_non_tensor_batch[key] = np.array([
                                        _pre_process_inputs(self.pad_token_id, final_prompts[i]) 
                                        for i in range(batch_size)
                                    ], dtype=object)
                                else:
                                    new_non_tensor_batch[key] = np.concatenate([value, value])
                            else:
                                raise ValueError(f"when it is inited it should be np.ndarray, but{type(value)=}")
                                # 对于非数组类型，保持原样
                                new_non_tensor_batch[key] = value

                        non_tensor_batch = new_non_tensor_batch


                    # 创建is_edited_mask标识哪些样本是编辑过的
                    is_edited_mask = torch.zeros(batch_size, self.config.response_length, dtype=torch.bool, device=idx.device)
                   
                    # 准备最终的edited log probabilities tensor
                    final_edited_log_probs = None

                    

                    if (self.select_fun in ["sample_edit","sample_edit_reward","select_edited_if_better","rank_by_improvement","rank_by_reward","group_rank_by_improvement"]) and self.enable_select:

                        original_mask = torch.zeros(response.size(0), self.config.response_length, dtype=torch.bool, device=response.device)
                        # 编辑后响应的mask全为True
                        edited_mask = torch.ones(edited_responses_tensor.size(0), self.config.response_length, dtype=torch.bool, device=response.device)
                        is_edited_mask = torch.cat([original_mask, edited_mask], dim=0)

                        original_log_probs = torch.zeros(response.size(0), self.config.response_length, device=idx.device, dtype=torch.float32)
                        final_edited_log_probs = torch.cat([original_log_probs, edited_log_probs_tensor], dim=0)  


                        is_original = torch.ones(response.size(0), dtype=torch.bool, device=response.device)
                        is_edited = torch.zeros(response.size(0), dtype=torch.bool, device=response.device)

                        edited_is_original = torch.zeros(edited_responses_tensor.size(0), dtype=torch.bool, device=response.device)
                        edited_is_edited = torch.ones(edited_responses_tensor.size(0), dtype=torch.bool, device=response.device)                        

                        final_is_original = torch.cat([is_original, edited_is_original], dim=0)
                        final_is_edited = torch.cat([is_edited, edited_is_edited], dim=0)
                        # use config.rollout.n to allocate `question_uid` and `pair_uids`,

                        pair_uids = []
                        question_uids = []
                        for i in range(response.size(0)):
                           
                            if i % self.config.n == 0:
                                question_uid = str(uuid.uuid4())
                                question_uids.append(question_uid)
                            else:
                                question_uids.append(question_uids[-1])

                            pair_id = str(uuid.uuid4())
                            pair_uids.append(pair_id)  # original answer

                        pair_uids=pair_uids+pair_uids
                        question_uids=question_uids+question_uids
                        

                        non_tensor_batch["pair_uid"] = np.array(pair_uids, dtype=object)
                        non_tensor_batch["question_uid"] = np.array(question_uids, dtype=object)
                        
                        batch = TensorDict(
                            {
                                "prompts": final_prompts,
                                "responses": final_responses,
                                "input_ids": final_seq,  # here input_ids become the whole sentences
                                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                                "attention_mask": final_attention_mask,
                                "position_ids": final_position_ids,
                                "is_original": final_is_original,
                                "is_edited": final_is_edited,
                                "edited_log_probs": final_edited_log_probs,  
                                "is_edited_mask": is_edited_mask                                 
                            },
                            batch_size=batch_size,
                        )
                    # free vllm cache engine
                    if (
                        vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine
                    ):
                        self.inference_engine.free_cache_engine()
                        
                    if api_metrics:
                        if api_metrics.get("feedback/total_count", 0) > 0:
                            total = api_metrics["feedback/total_count"]
                            success = api_metrics["feedback/success_count"]
                            api_metrics["feedback/success_rate"] = success / total if total > 0 else 0.0
                            api_metrics["feedback/avg_attempts"] = api_metrics["feedback/total_attempts"] / total if total > 0 else 0.0
                            if success > 0:
                                api_metrics["feedback/avg_response_length"] = api_metrics["feedback/total_response_length"] / success
                            else:
                                api_metrics["feedback/avg_response_length"] = 0.0
                    # final_data = DataProto.from_dict(tensors=batch_dict, non_tensors=non_tensor_batch)
                    # return final_data

                    # increment counter
                    if self.save_data:
                        self.save_counter += 1
                        
                    output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
                    output.meta_info.update(api_metrics)
                    return output


            if self.sampling_params.n > 1 and do_sample: # 这个分支处理多候选生成的情况：在validate 和train的时候都会调用， 用于处理non_tensor_batch
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(non_tensor_batch["multi_modal_inputs"], self.sampling_params.n)
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)      
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)                      

            seq = torch.cat([idx, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

            response_position_ids = position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            if self.save_data:
                import datetime
                prompts_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in idx]
                original_responses_text = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response]
                save_data = []
                for i in range(len(prompts_text)):
                    data_entry = {
                        "prompt": prompts_text[i],
                        "original_response": original_responses_text[i],
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    save_data.append(data_entry)
                
                self._save_response_data(save_data, prefix="original_responses", step=self.save_counter)

            # Create batch dict
            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": response,
                    "input_ids": seq,  # here input_ids become the whole sentences
                    # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=batch_size,
            )

            # free vllm cache engine
            if (
                vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine
            ):
                self.inference_engine.free_cache_engine()

            # 统一递增counter  
            if self.save_data:
                self.save_counter += 1
                
            output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            output.meta_info.update(api_metrics)
            return output

class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
