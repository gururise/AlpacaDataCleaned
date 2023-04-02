## Benchmark llama models on Open Source Datasets

### Usage
```
usage: eval.py [-h] -b BASE_MODEL [-l LORA_WEIGHTS] [-d {wikitext,squadmini,squad}] [-q]

options:
  -h, --help            show this help message and exit
  -b BASE_MODEL, --base-model BASE_MODEL
                        Choose the base model
  -l LORA_WEIGHTS, --lora-weights LORA_WEIGHTS
                        Choose the lora weights (optional)
  -d {wikitext,squadmini,squad}, --datasets {wikitext,squadmini,squad}
                        Choose Evaluation Dataset. [default = squadmini]
  -q, --use-8bit        Use 8-bit quant
```

### Datasets

- [squad](https://huggingface.co/datasets/squad) - validation split (10570 Q/A pairs) - returns avg. f1 score
- squadmini - same as above, but every 10th element (1057 Q/A pairs) - returns avg. f1 score
- [wikitext](https://huggingface.co/datasets/wikitext) - wikitext-2-raw-v1 test split - returns perplexity

### Example
`eval.py --base-model decapoda-research/llama-7b-hf --lora-weights samwit/alpaca7B-lora --datasets squadmini`

### Benchmark Comparison
I compared two different alpaca 7b models on the Squad Dataset:

dataset | model | Squad(Mini) F1 
------- | ----- | ---------
Original Alpaca | samwit/alpaca7B-lora | 34.63
Cleaned Alpaca  | tloen/alpaca-lora-7b | 49.64

The cleaned alpaca trained model was using the cleaned dataset snapshot dated March 27, 2023.

At least on the surface, it appears the cleaning & curation we've been doing has helped significantly.