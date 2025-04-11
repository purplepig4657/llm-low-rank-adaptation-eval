# LLM Low-Rank Adaptation PEFT Reproduction

This repository reproduces the Low-Rank Adaptation (PEFT) experiments from the CorDA paper.

# Results

## GLUE

| Method           | # of Params | SST-2     | MRPC      | CoLA      | QNLI      | RTE       | STS-B     | Avg       |
| ---------------- | ----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Full fine-tuning | 125M        | 94.15     | 88.73     | 58.80     | 92.42     | 64.62     | **90.56** | **83.56** |
| LoRA_HF          | 21M         | **94.38** | 87.01     | 55.93     | 92.33     | 61.73     | 87.61     | 79.83     |
| DoRA_HF          | 21M         | 94.15     | 85.54     | 55.80     | **92.62** | 66.79     | 88.34     | 80.54     |
| CorDA_HF         | 21M         | 93.00     | **89.95** | 57.02     | 91.96     | **75.09** | 90.07     | 82.68     |
| PiSSA_HF         | 21M         | 94.27     | 89.22     | **60.60** | **92.62** | 68.59     | 89.81     | 82.52     |

## Math and Code

| Method | # of Params | GSM8K | Math | HumanEval | MBPP | Avg |
| - | - | - | - | - | - | - |
| LoRA_HF | N/A | 34.34 | 6.7 | N/A | 43 | N/A |
| DoRA_HF | N/A | N/A | N/A | N/A | N/A | N/A |
| CorDA_HF | N/A | N/A | N/A | N/A | N/A | N/A |
| PiSSA_HF | N/A | N/A | N/A | N/A | N/A | N/A |

# Model, Hyper-parameters and LoRA configs

## GLUE

| Parameter | Default Value |
|-----------|---------------|
| model_name | roberta-base |
| lora_r | 128 |
| lora_alpha | 128 |
| lora_dropout | 0.0 |
| num_epochs | 3 |
| learning_rate | 4e-5 |
| batch_size | 32 |
| max_length | 128 |
| lr_scheduler | linear |
| optimizer | AdamW |

All linear layers are adapted except for the classifier heads.

The CorDA covariance matrix is initialized using random 256 samples from each task dataset.
And it uses CorDA's instruction-previewed adaptation.

All of these settings are based on the CorDA paper.

## Math and Code

| Parameter                    | Value                |
|-----------------------------|----------------------|
| model_name                  | meta-llama/Meta-Llama-3-8B |
| model_max_length            | 512                  |
| dataset_split               | train[:100000]       |
| optim                       | adamw_torch          |
| num_epochs                  | 1                    |
| weight_decay                | 0.0                  |
| warmup_ratio                | 0.03                 |
| learning_rate               | 2e-5                 |
| lr_scheduler | cosine |
| batch_size                  | 1                    |
| gradient_accumulation_steps| 128                  |
| seed                        | 0                    |
| lora_r                      | 32                  |
| lora_alpha                  | 32                  |
| lora_dropout                | 0.0                  |


# How to Train and Evaluate Math and Code task

Training large decoder models is expensive, so it decided to separate model training and evaluation.

First, train decoder models with LoRA, DoRA, PiSSA, and CorDA on the math and code fine-tuning datasets, specifically **MetaMathQA** and **CodeFeedback-Filtered-Instruction**.

For CorDA, the covariance matrix is initialized using each sampled data point from the fine-tuning dataset.

```bash
python3 math_train.py  # For math tasks
python3 code_train.py  # For code tasks
```

If you want to change any specific training settings, please check the argument configurations in `math_train.py` and `code_train.py`.

## Evaluating Math Datasets

The CorDA paper evaluates math tasks using the **MATH** and **GSM8K** datasets.

The evaluation code and datasets follow the official [CorDA GitHub repository](https://github.com/iboing/CorDA), and should be installed accordingly.

Before running the evaluation code, install the required libraries:

```bash
pip install vllm fraction
```

Then, run the evaluation with:

```bash
python3 math_evaluation.py
```

Evaluation arguments can be found in `math_evaluation.py`.

---

## Evaluating Code Datasets

The CorDA paper evaluates code generation tasks using the **MBPP** and **HumanEval** datasets.

Following the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness), install it by running:

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -e .
```

Then, execute the evaluation:

```bash
python3 code_evaluation.py
```

**Note:** The default path for `bigcode-evaluation-harness` is assumed to be within the current project directory.  
If you wish to change this path, make sure to set the `--bigcode_eval_path` argument in `code_evaluation.py`.
