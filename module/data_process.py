import vllm
from transformers import LlamaTokenizer
from .prompts_summeval import summarization_prompt_mapping
from .prompts_topicalchat import dialogue_prompt_mapping
from collections import Counter
import random

system_prompt = "You are an impartial and helpful evaluator for natural language generation (NLG)."

task_mapping = {
    "Summarization": summarization_prompt_mapping,
    "Dialogue": dialogue_prompt_mapping
}


def OC_aspect_mapping(aspect):
    if aspect in ["Understandability", "Knowledge Use"]:
        return "OC_2"
    if aspect in ["Naturalness", "Context Maintenance", "Interestingness"]:
        return "OC_3"
    return "OC_5"


def pre_process(ex, task_type, model, tokenizer, **kwargs):
    task = ex["A"]["task"] if task_type == "PC" else ex['task']
    if task_type == "OC":
        task_type = OC_aspect_mapping(ex['aspect'])
    prompt_template = task_mapping[task][task_type]

    if task_type == "PC":
        user_content = prompt_template.format(target_A=ex['A']['target'], target_B=ex['B']['target'], **ex['A'])
    else:
        user_content = prompt_template.format(**ex)
    conversation = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_content}
    ]
    if 'gemma' in model.lower():
        conversation = [
            {"role": "user", "content": system_prompt + "\n\n" + user_content}
        ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
        prompt = prompt.removeprefix(tokenizer.bos_token)
    return prompt


def get_SamplingParams(**generate_kwargs):
    sampling_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1,
        n=10,
    )

    for key, value in generate_kwargs.items():
        if hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    return sampling_params


import re 

def get(st, en):
    pre_sub_fix = ["", "'", "\"", "*", "**", "***", "#", "##", "###", '-']
    sub_subfix = ["", "/{en}", ".0", ".0/{en}", ".", "./{en}", 
                "/{en}.", ".0/{en}.", "./{en}.", 
                "/{en}.0", ".0/{en}.0", "./{en}.0", 
                "[out", "\u200b", "*out"]
    d = dict()

    for pre in pre_sub_fix:
        for sub in pre_sub_fix:
            for sub_sub in sub_subfix:
                d.update(
                    [((pre + "{i}" + sub_sub + sub).format(i=i, en=en), i) for i in range(st, en + 1)]
                )

    return d

subfixs = [":***", ":**", ":*", "**:", "***", "**", "*", " :", ":_", " ###", " ##", " #", "###", "##", "#", ":"]
poss_st = {"rating": [], "score": []}
for sub in subfixs:
    poss_st["rating"].append(("Rating" + sub, len("Rating" + sub)))
    poss_st["score"].append(("Score" + sub, len("Score" + sub)))

skip_st = {"rating": ["### Final Rating: ###", "\"Rating:\"", "'Rating:'", "Explanation for Rating:"], "score": ["### Final Score: ###", "\"Score:\"", "'Score:'", "Explanation for Score:"]}

def filter_rating(v, n, t):
    ratings = get(1, n)
    out = []
    if v is None:
        return 0
    for response in v:
        response_split = response.split('\n')
        is_rating = None
        error_type = '[can\'t find]'
        for i in range(len(response_split)):
            is_skip = False
            for j in skip_st[t]:
                if j in response_split[i]:
                    is_skip = True
            if is_skip:
                continue

            for rating_st, rating_st_len in poss_st[t]:
                if rating_st in response_split[i]:
                    res_split = response_split[i]
                    p = res_split.find(rating_st)
                    rating_part = res_split[p + rating_st_len:].strip()
                    response_split[i] = res_split[:p]
                    is_rating = ratings.get(rating_part.split()[0], "") if rating_part else ""
                    if is_rating == "":
                        if i + 1 < len(response_split):
                            if response_split[i+1].strip():
                                is_rating = ratings.get(response_split[i+1].strip().split()[0], "") 
                            elif i + 2 < len(response_split) and response_split[i+2].strip():
                                is_rating = ratings.get(response_split[i+2].strip().split()[0], "") 
                    if is_rating != "":
                        reason = '\n'.join(response_split[:i]).strip()
                        out.append(float(is_rating))
                    break
            
            if is_rating is not None:
                break
    
    return out


def calculate_mode(numbers):
    if not numbers:
        return None
    counter = Counter(numbers)
    max_count = max(counter.values())

    modes = [num for num, count in counter.items() if count == max_count]
    random_number = random.choice(modes)
    return random_number


def extract(s):
    a = 0
    b = 0
    c = 0
    for sent in s:
        aa = 0
        bb = 0
        cc = 0
        if 'A < B' in sent or 'B > A' in sent or 'Summary A < Summary B' in sent or 'Summary B > Summary A' in sent or 'Response A < Response B' in sent or 'Response B > Response A' in sent:
            aa += 1
        if 'A > B' in sent or 'B < A' in sent or 'Summary A > Summary B' in sent or 'Summary B < Summary A' in sent or 'Response A > Response B' in sent or 'Response B < Response A' in sent:
            bb += 1
        if 'A = B' in sent or 'B = A' in sent or 'Summary A = Summary B' in sent or 'Summary B = Summary A' in sent or 'Response A = Response B' in sent or 'Response B = Response A' in sent:
            cc += 1
        if aa + bb + cc != 1:
            # print(sent)
            pass
        else:
            a += aa
            b += bb
            c += cc
    return (a, b, c)


def post_process(raw_output: list[str], ex, task_type, **kwargs):
    if task_type == "OC":
        task_type = OC_aspect_mapping(ex['aspect'])
        post_output = filter_rating(raw_output, int(task_type.split("_")[-1]), "rating")
        post_output = calculate_mode(post_output)
        if post_output is None:
            post_output = random.randint(1, int(task_type.split("_")[-1]))
    elif task_type.startswith("SC"):
        post_output = filter_rating(raw_output, int(task_type.split("_")[-1]), "score")
        if len(post_output) == 0:
            post_output = random.uniform(1, int(task_type.split("_")[-1]))
        else:
            post_output = sum(post_output) / len(post_output)
    elif task_type == "PC":
        post_output = extract(raw_output)
    else:
        raise NotImplementedError
    
    return post_output