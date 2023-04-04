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
  -d {wikitext,squadmini,squad,piqa}, --datasets {wikitext,squadmini,squad,piqa}
                        Choose Evaluation Dataset. [default = squadmini]
  -q, --use-8bit        Use 8-bit quant
```

### Datasets

- [squad](https://huggingface.co/datasets/squad) - validation split (10570 Q/A pairs) - returns avg. f1 score
- squadmini - same as above, but every 10th element (1057 Q/A pairs) - returns avg. f1 score
- [piqa](https://huggingface.co/datasets/piqa) - validation split () - returns accuracy
- [wikitext](https://huggingface.co/datasets/wikitext) - wikitext-2-raw-v1 test split - returns perplexity

### Example
`eval.py --base-model decapoda-research/llama-7b-hf --lora-weights samwit/alpaca7B-lora --datasets squadmini`

### Benchmark Comparison
Comparison of models trained on various datasets

**NOTE:** The Piqa tests are not working correctly right now. Do not rely on the piqa scores

Dataset | Model | parameters | SquadMini (f1) | Piqa (acc) 
------- | ----- | ----- | ----- | -----
**Original Alpaca** | samwit/alpaca7B-lora | 7b | 74.271 | ~~50.5~~
**Cleaned Alpaca** (Mar 26)  | tloen/alpaca-lora-7b | 7b | 75.629 | ~~54.0~~
**Cleaned Alpaca** (Apr 2) 8bit | yahma/alpaca-13b-lora | 13b | **77.765** | -
**GPT4All**  | nomic-ai/gpt4all-lora | 7b | 72.643 | -

At least on the surface, it appears the cleaning & curation has helped.