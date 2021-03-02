# GLOM

This is a simple implementation of the GLOM model ([paper](https://arxiv.org/pdf/2102.12627.pdf)) that heavily builds on the [hugginface implementation](https://github.com/huggingface/transformers/tree/master/src/transformers/models/albert) of the ALBERT model ([paper](https://arxiv.org/abs/1909.11942)).


## Getting Started

1) install requirements from requirements.txt

2) train a model
```
python run_hf_mlm.py --config configs/glom/config.json --model_type glom --tokenizer albert-base-v2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --evaluation_strategy steps --logging_steps 100 --output_dir train/test-mlm
```