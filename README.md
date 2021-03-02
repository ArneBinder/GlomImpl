# GLOM

This is a simple implementation of the GLOM model ([paper](https://arxiv.org/pdf/2102.12627.pdf)). It heavily builds on the [hugginface implementation](https://github.com/huggingface/transformers/tree/master/src/transformers/models/albert) of the ALBERT model ([paper](https://arxiv.org/abs/1909.11942)).

## Approach
* use t layers (t=number of time steps you want to model)
* use L heads (L=number of GLOM layers you want to model)
* apply these small modifications to the ALBERT model:
	1) remove the linear linear projections for query, key, value; just pass through `[(d/L)*i..(d/L)*(i+1)]` to the i'th head
	2) modify/constrain the dense layer that follows the attention in a way that each partition `[(d/L)*i..(d/L)*(i+1)]` of its output is only constructed by the output of the (i-1)-th head and the (i+1)-th head (this models the access to the lower and higher GLOM "layer")
	3) remove the skip connection(s) and the MLP that sits on top of the attention layer


## Getting Started

1) install requirements from requirements.txt

2) train a model
```
run_hf_mlm.py \
--config configs/glom/config.json \
--tokenizer albert-base-v2 \
--dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 \
--do_train \
--max_examples_train 100000 \
--num_train_epochs 1 \
--do_eval \
--evaluation_strategy steps \
--logging_steps 100 \
--output_dir train/test-mlm \
--save_total_limit 1 \
--overwrite_output_dir \
```

for comparison, train a similar (has a bit more parameters) ALBERT model 
```
run_hf_mlm.py \
--config albert-base-v2 \
--tokenizer albert-base-v2 \
--dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 \
--do_train \
--max_examples_train 100000 \
--num_train_epochs 1 \
--do_eval \
--evaluation_strategy steps \
--logging_steps 100 \
--output_dir train/test-mlm \
--save_total_limit 1 \
--overwrite_output_dir \
```
