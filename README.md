## A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability

This is the official repository for our ACL 2025 paper [A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](https://aclanthology.org/2025.acl-long.1327.pdf).

We release our new NLG meta-evaluation benchmarks in `benchmark/`, including both global and local perspectives of SummEval and Topical-Chat. Detailed statistics can be found in the paper.

- For the meta-evaluation on the global-perspective benchmark:

```
CUDA_VISIBLE_DEVICES=<GPU_ids> python eval.py \
    --model gemma-2-27b-it \
    --task_type OC \
    --test_type global \
    --test_path benchmark/SummEval_global.json \
    --output_dir result/SummEval_global
```

We select gemma-2-27b-it and SummEval as the example, and you can specify the evaluation model path `model`, the benchmark path `test_path`, and the output path `output_dir`.

- For the meta-evaluation on the local-perspective benchmark:

```
CUDA_VISIBLE_DEVICES=<GPU_ids> python eval.py \
    --model gemma-2-27b-it \
    --task_type SC_10 \
    --test_type local \
    --test_path benchmark/SummEval_Local.json \
    --output_dir result/SummEval_Local
```

```
CUDA_VISIBLE_DEVICES=<GPU_ids> python eval.py \
    --model gemma-2-27b-it \
    --task_type PC \
    --test_type local \
    --test_path benchmark/SummEval_Local.json \
    --output_dir result/SummEval_Local
```

Our default setting is a 1–10 scoring range, while other formats are also supported, such as a broader range of 1–100 (by setting `task_type` to `SC_100`) and pairwise comparison (by setting `task_type` to `PC`).

### Citation

```
@inproceedings{hu-etal-2025-dual,
    title = "A Dual-Perspective {NLG} Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability",
    author = "Hu, Xinyu  and
      Gao, Mingqi  and
      Lin, Li  and
      Yu, Zhenghan  and
      Wan, Xiaojun",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1327/",
    doi = "10.18653/v1/2025.acl-long.1327",
    pages = "27372--27395",
    ISBN = "979-8-89176-251-0"
}
```
