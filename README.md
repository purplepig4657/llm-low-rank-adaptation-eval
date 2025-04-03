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

## MATH

No results available yet.

## CODE

No results available yet.

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

## MATH

To be updated.

## CODE

To be updated.
