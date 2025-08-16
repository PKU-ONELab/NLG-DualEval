prompt_SC_1_10 = """### Instruction ###
Your task is to evaluate the quality of a response for the next turn of a dialogue context between two people.
The evaluation must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise analysis, followed by the corresponding evaluation score from 1 to 10 (higher means better).
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### Example ###
{addition_des}:
{addition}
{source_des}:
{source}
{target_des}:
{target}

### Your Evaluation ###
Analysis:
Score:
"""

prompt_SC_1_100 = """### Instruction ###
Your task is to evaluate the quality of a response for the next turn of a dialogue context between two people.
The evaluation must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise analysis, followed by the corresponding evaluation score from 1 to 100 (higher means better).
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### Example ###
{addition_des}:
{addition}
{source_des}:
{source}
{target_des}:
{target}

### Your Evaluation ###
Analysis:
Score:
"""

prompt_OC_3 = """### Instruction ###
Your task is to evaluate the quality of a response for the next turn of a dialogue context between two people.
The evaluation must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise analysis, followed by the corresponding rating on a 3-point Likert scale:
    3 (Good): You strongly agree that the response has good {aspect}.
    2 (Average): You neither agree nor disagree that the response has good {aspect}.
    1 (Poor): You strongly disagree that the response has good {aspect}.
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### Example ###
{addition_des}:
{addition}
{source_des}:
{source}
{target_des}:
{target}

### Your Evaluation ###
Analysis:
Rating:
"""

prompt_OC_2 = """### Instruction ###
Your task is to evaluate the quality of a response for the next turn of a dialogue context between two people.
The evaluation must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise analysis, followed by the corresponding rating on a 2-point Likert scale:
    2 (Good): You strongly agree that the response has good {aspect}.
    1 (Poor): You strongly disagree that the response has good {aspect}.
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### Example ###
{addition_des}:
{addition}
{source_des}:
{source}
{target_des}:
{target}

### Your Evaluation ###
Analysis:
Rating:
"""

prompt_PC = """### Instruction ###
Your task is to evaluate and compare the quality of two responses for the next turn of a dialogue context between two people.
The evaluation and comparison must be strictly focused on the aspect of **{aspect}**, and based on the given evaluation criterion.
Provide your evaluation with a concise contrastive analysis, followed by the corresponding judgment from A > B, A < B, and A = B:
    A > B means the quality of Response A on **{aspect}** is **better** than that of Response B.
    A < B means the quality of Response A on **{aspect}** is **worse** than that of Response B.
    A = B means the quality of Response A on **{aspect}** is **similar** to that of Response B.
You must understand and follow these instructions carefully and adhere to the strict boundaries of the given evaluation criterion.

### Evaluation Criterion ###
{aspect_des}

### {addition_des} ###
{addition}

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

dialogue_prompt_mapping = {
    "OC_2": prompt_OC_2, 
    "OC_3": prompt_OC_3,
    "SC_10": prompt_SC_1_10,
    "SC_100": prompt_SC_1_100,
    "PC": prompt_PC
}