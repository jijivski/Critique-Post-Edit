import asyncio
import os
import random
import re
import string
import traceback
from typing import List, Set

import openai


OPENAI_API_KEY = os.getenv("GRM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_API_BASE_URL = os.getenv("GRM_API_BASE_URL") or os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("GRM_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4o-mini")

print(f'grm using {OPENAI_API_BASE_URL=},{MODEL_NAME=}')


client = openai.AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE_URL,
)


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

elif prompt_version=='P0805':

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

注意，给分必须严格，只要出现负分的情况必须给负分！最后记住，用户需求不是让AI输出“华丽辞藻和丰富场景”，而是信息密度高、针对性强、风格自然、不过度花哨的高质量答案。如果回答的最后一句出现总结骗分，自然度就必须给负分（这样会让用户感到不适）！\n\n\n\n以下是用户配置：
{persona_str}
\n\n以下是用户的【问题】
{question_str}
\n\n以下是模型回答：
{solution_str}
"""


def extract_between_keywords(text, start_label, end_label):
    start_label_escaped = re.escape(start_label)
    end_label_escaped = re.escape(end_label)
    pattern = f'({start_label_escaped})(.*?)({end_label_escaped})'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip()
    else:
        return "None"

def process_individual_response(response):
    
    critique = extract_between_keywords(response, '<critique>', '</critique>')
    scores_text = extract_between_keywords(response, '<scores>', '</scores>')
    
    scores = {
        'helpfulness': 0,
        'personalization': 0,
        'natural': 0
    }

    if scores_text:
        helpfulness_match = re.search(r'Helpfulness:\s*(-?\d+)', scores_text)
        personalization_match = re.search(r'Personalization:\s*(-?\d+)', scores_text)
        natural_match = re.search(r'Natural:\s*(-?\d+)', scores_text)
        
        if helpfulness_match:
            scores['helpfulness'] = int(helpfulness_match.group(1))
        if personalization_match:
            scores['personalization'] = int(personalization_match.group(1))
        if natural_match:
            scores['natural'] = int(natural_match.group(1))

    result = {
        "critique": critique,
        "scores": scores
    }
    
    return result

async def score_trajectory_single(data_source: str, question_str:str, persona_str:str, solution_str: str, ground_truth: str, max_retries: int = 5) -> float:
    """
    Scores a single agent trajectory with a retry mechanism.

    Args:
        solution_str: The string containing the agent's full trajectory.
        ground_truth: The ground truth answer for the given problem.
        max_retries: The maximum number of retries before failing.

    Returns:
        A normalized reward score as a float in the range [0.0, 1.0].
        Returns 0 if all attempts fail.
    """
    do_print = random.randint(1, 64) == 1
    formatted_prompt = PROMPT_TEMPLATE.format(question_str=question_str,persona_str=persona_str,solution_str=solution_str,)
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    # {"role": "system", "content": "You are a meticulous and expert evaluator."},
                    {"role": "user", "content": formatted_prompt},
                ],
                temperature=0.0,
            )

            response_text = response.choices[0].message.content

            result_critic_score=process_individual_response(response_text)
            score_dic=result_critic_score['scores']

            helpfulness=score_dic['helpfulness']
            natural=score_dic['natural']
            personalization=score_dic['personalization']

            # Weighted combination: helpfulness=35%, personalization=40%, natural=25%
            coef_H,coef_P,coef_N=0.35,0.4,0.25
            llm_score=coef_H*helpfulness+coef_P*personalization+coef_N*natural
            output_score = llm_score

            if do_print:
                print("--- Successful Evaluation ---")
                print(f"Data scource: {data_source}")
                print(f"solution_str: {solution_str.strip()}")
                print(f"Response: {response_text.strip()}")
                print(f"Final Score: {output_score}")
                print("---------------------------\n")
            return output_score

        except Exception as e:
            print(f"An error occurred during API call on attempt {attempt + 1}: {e}")
            print("Full traceback:")
            traceback.print_exc() 

        # If not the last attempt, wait before retrying
        if attempt < max_retries:
            print(f"Retrying in 1 second...")
            await asyncio.sleep(1)

        print(f"All {max_retries + 1} attempts failed. Returning a default score of 0.")
        return 0



def compute_score_grm_batch(
    data_sources: list[str],
    questions_str: list[str],
    personas_str: list[str],
    solution_strs: list[str], 
    ground_truths: list[dict],
    extra_infos: list[dict],
    **kwargs
) -> list[float]:
    """
    Synchronously scores a batch of agent trajectories, but processes them asynchronously internally.

    # verl/verl/workers/reward_manager/batch.py L50
        # scores = self.compute_score(
        #     data_sources=data_sources,
        #     solution_strs=responses_str,
        #     ground_truths=ground_truths,
        #     extra_infos=extras,
        #     **self.reward_kwargs,
        # )
        
    """
    if len(solution_strs) != len(ground_truths) !=len(questions_str) != len(personas_str):
        raise ValueError("The number of solution_strs and ground truths must be equal.")

    async def _async_batch_score():
        tasks = [
            score_trajectory_single(data_source, question_str, persona_str, solution_str, gt)
            for data_source, question_str, persona_str, solution_str, gt in zip(data_sources, questions_str, personas_str, solution_strs, ground_truths)
        ]
        return await asyncio.gather(*tasks)

    # Run the async function in an event loop
    return asyncio.run(_async_batch_score())
