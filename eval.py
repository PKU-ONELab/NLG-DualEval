import os
import json
import argparse
from module import data_process

from vllm import LLM, EngineArgs
from transformers import AutoTokenizer

import numpy as np

def CEM_ORD(
    gold_labels, 
    predict_labels, 
    num_classes = 5,
    class_to_index = lambda x: int(x - 1)
):
    assert len(gold_labels) == len(predict_labels)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for gold_label, pred_label in zip(gold_labels, predict_labels):
        i = class_to_index(pred_label)
        j = class_to_index(gold_label)
        confusion_matrix[i, j] += 1

    N = len(gold_labels)
    gold_class = confusion_matrix.sum(axis=0)
    cusum_gold_class = np.cumsum(gold_class)
    cusum_1_gold_class = np.pad(cusum_gold_class, (1, 0))[:-1]
    row = np.arange(num_classes)[:, None]
    col = np.arange(num_classes)[None, :]
    max_idx = np.maximum(row, col)
    min_idx = np.minimum(row, col)
    weight = - np.log(np.maximum(cusum_gold_class[max_idx] - cusum_1_gold_class[min_idx] - gold_class[row] / 2, 0.5) / N)

    numerator = (weight * confusion_matrix).sum()
    denominator = (weight.diagonal() * gold_class).sum()
    CEM = numerator / denominator
    return CEM


class Namespace(argparse.Namespace):
    model: str
    task_type: str
    test_type: str
    output_dir: str
    test_path: str

    ## vllm config
    tp_size: int
    seed: int

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument("--task_type", required=True, type=str)
    parser.add_argument("--test_type", required=True, type=str, choices=["global", "local"])
    parser.add_argument('--test_path', default=None, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--tp_size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args(namespace=Namespace())

    print(f"args: {args}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = json.load(open(args.test_path))

    all_test_examples = []
    if args.test_type == "global":
        assert args.task_type == "OC"
        for aspect, aspect_dataset in test_dataset.items():
            all_test_examples.extend(aspect_dataset)
    else:
        for aspect, aspect_dataset in test_dataset.items():
            for src, src_dataset in aspect_dataset.items():
                if args.task_type.startswith("SC"):
                    all_test_examples.extend(src_dataset)
                else:
                    for i in range(len(src_dataset) - 1):
                        all_test_examples.append({"A": src_dataset[i], "B": src_dataset[i + 1]})
                        all_test_examples.append({"B": src_dataset[i], "A": src_dataset[i + 1]})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_test_prompts = [data_process.pre_process(ex, task_type=args.task_type, model=args.model, tokenizer=tokenizer) for ex in all_test_examples]
    print("Example:", all_test_prompts[0])

    def process(inputs):
        engine_args = EngineArgs(model=args.model, 
                                 swap_space=8, 
                                 seed=args.seed, 
                                 enable_prefix_caching=True, 
                                 tensor_parallel_size=args.tp_size, 
                                 gpu_memory_utilization=0.9,
                                 trust_remote_code=True)

        model = LLM(**engine_args.__dict__)
        outputs = model.generate(inputs, sampling_params=data_process.get_SamplingParams())

        outs = [[ex.text for ex in out.outputs] for out in outputs]
        return outs

    outs = process(all_test_prompts)
    all_outputs = []
    for out, ex in zip(outs, all_test_examples):
        all_outputs.append({
            "raw_output": out,
            "post_output": data_process.post_process(out, ex=ex, task_type=args.task_type)
        })

    json.dump(all_outputs, open(os.path.join(args.output_dir, ".".join(["completions", args.model.split("/")[-1], args.task_type, "json"])), "w"), indent=4)

    print("###Evaluation Results###")

    if args.test_type == "global":
        assert args.task_type == "OC"
        cur_id = 0
        for aspect, aspect_dataset in test_dataset.items():
            model_label = []
            human_label = []
            for tmp in aspect_dataset:
                human_label.append(tmp["rating"])
            for tmp in all_outputs[cur_id: cur_id + len(aspect_dataset)]:
                model_label.append(tmp["post_output"])
            print(aspect, CEM_ORD(np.array(human_label), np.array(model_label), num_classes=int(data_process.OC_aspect_mapping(aspect).split("_")[-1])))
            cur_id += len(aspect_dataset)
    else:
        cur_id = 0
        for aspect, aspect_dataset in test_dataset.items():
            comp_true = 0
            comp_num = 0
            for src, src_dataset in aspect_dataset.items():
                if args.task_type.startswith("SC"):
                    for tmp in range(cur_id, cur_id + len(src_dataset) - 1):
                        if all_outputs[tmp]["post_output"] > all_outputs[tmp + 1]["post_output"]:
                            comp_true += 1
                        comp_num += 1
                    cur_id += len(src_dataset)
                else:
                    for tmp in range(cur_id, cur_id + len(src_dataset) - 1):
                        j1 = all_outputs[tmp * 2]["post_output"]
                        j2 = all_outputs[tmp * 2 + 1]["post_output"]
                        if j1[1] + j2[0] > j1[0] + j2[1] and j1[1] + j2[0] > j1[2] + j2[2]:
                            comp_true += 1
                        comp_num += 1
                    cur_id += (len(src_dataset) - 1)
            print(aspect, comp_true / comp_num)

