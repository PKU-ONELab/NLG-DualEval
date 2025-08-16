prompt_SC_1_10 = """### Instruction ###
Your task is to evaluate the quality of a summary written for an article.
The evaluation must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise analysis, followed by the corresponding evaluation score from 1 to 10 (higher means better).
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### Example ###
{source_des}:
{source}
{target_des}:
{target}

### Your Evaluation ###
Analysis:
Score:
"""

prompt_SC_1_100 = """### Instruction ###
Your task is to evaluate the quality of a summary written for an article.
The evaluation must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise analysis, followed by the corresponding evaluation score from 1 to 100 (higher means better).
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### Example ###
{source_des}:
{source}
{target_des}:
{target}

### Your Evaluation ###
Analysis:
Score:
"""

prompt_OC = """### Instruction ###
Your task is to evaluate the quality of a summary written for an article.
The evaluation must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise analysis, followed by the corresponding rating on a 5-point Likert scale:
    5 (Good): You strongly agree that the summary has good {aspect}.
    4 (Above Average): You basically agree that the summary has good {aspect}.
    3 (Average): You neither agree nor disagree that the summary has good {aspect}.
    2 (Below Average): You basically disagree that the summary has good {aspect}.
    1 (Poor): You strongly disagree that the summary has good {aspect}.
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### Example ###
{source_des}:
{source}
{target_des}:
{target}

### Your Evaluation ###
Analysis:
Rating:
"""

prompt_PC = """### Instruction ###
Your task is to evaluate and compare the quality of two summaries written for an article.
The evaluation and comparison must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise contrastive analysis, followed by the corresponding judgment from A > B, A < B, and A = B:
    A > B means the quality of Summary A on **{aspect}** is **better** than that of Summary B.
    A < B means the quality of Summary A on **{aspect}** is **worse** than that of Summary B.
    A = B means the quality of Summary A on **{aspect}** is **similar** to that of Summary B.
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### {source_des} ###
{source}

### {target_des} A ###
{target_A}

### {target_des} B ###
{target_B}

### Your Evaluation ###
Analysis:
Judgment:
"""

summarization_prompt_mapping = {
    "OC_5": prompt_OC, 
    "SC_10": prompt_SC_1_10,
    "SC_100": prompt_SC_1_100,
    "PC": prompt_PC
}