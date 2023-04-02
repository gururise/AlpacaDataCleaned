import os
import sys
import argparse

#import fire
import itertools
import torch

from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from evaluate import load
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def evaluate(
    tokenizer,
    model,
    prompt,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        num_beams=1,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=32,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output

def calc_perplexity(encodings, model, max_length):
    stride = 512
    
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    
    return ppl

def calc_f1(model, tokenizer, dataset, dataset_size, max_tokens):
    # Evaluate the model on the SQuAD dataset
    prompter = Prompter('alpaca')
    squad_metric = load("squad")
    
    f1_scores = []
    print (f"Dataset Size: {dataset_size}")
    count = 1
    
    for example in dataset:
        context = example['context']
        question = "Generate a short, precise answer to this Question in 1 to 5 words: " + example['question']
        prompt = prompter.generate_prompt(question, context)
        ground_truth = example['answers']['text'][0]

        output = evaluate(prompt=prompt,tokenizer=tokenizer,model=model, max_new_tokens=max_tokens)
        prediction = prompter.get_response(output)
        predictions = [{'prediction_text': prediction,'id': example['id']}]
        references = [{'answers': {'answer_start': example['answers']['answer_start'], 'text': example['answers']['text']}, 'id': example['id']}]
        results = squad_metric.compute(predictions=predictions, references=references)

        #f1_score = compute_f1(prediction, ground_truth)
        f1_scores.append(results['f1'])
        print(f"\n({count}/{dataset_size}):\nQ: {example['question']}\nPrediction: {prediction}\nGround Truth: {ground_truth}\nf1: {round(results['f1'],3)} - avg_f1: {round(sum(f1_scores) / len(f1_scores),3)}")
        count+=1
    f1 = sum(f1_scores) / len(f1_scores)
    
    return f1

# Define a function to compute the F1 score
def compute_f1(prediction, ground_truth):
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    if len(common_tokens) == 0:
        return 0
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def main():
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model', default='decapoda-research/llama-7b-hf', required=True, type=str, help="Choose the base model")
    parser.add_argument('--lora-weights', type=str, help="Choose the lora weights (optional)")
    parser.add_argument('--datasets', default='squadmini', choices=['wikitext','squadmini','squad'], help="Choose Evaluation Dataset. [default = squadmini]")
    parser.add_argument('--use-8bit', action="store_true", default=False, help="Use 8-bit quant")
    args = parser.parse_args()
    
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=args.use_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if args.lora_weights != None:
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if args.lora_weights != None:
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if args.lora_weights != None: 
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
            )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    #model.seqlen = 2048

    if not args.use_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32" and args.datasets != 'wikitext':
        model = torch.compile(model)
        
    if args.datasets == 'squad':
        ds = load_dataset("squad", split="validation")
        f1 = calc_f1(model,tokenizer, ds, len(ds), 1024)
        print(f"Squad F1 Score: {round(f1,3)}")
    elif args.datasets == 'squadmini':
        ds = load_dataset("squad", split="validation")
        ds_size = len(ds)//10
        ds = itertools.islice(ds, 0, None, 10)
        f1 = calc_f1(model,tokenizer, ds, ds_size, 1024)
        print(f"Squad 'Mini' F1 Score: {round(f1,3)}")        
    elif args.datasets == 'wikitext':
        ds = load_dataset("wikitext","wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
        ppl = calc_perplexity(encodings, model,1024)
        print(f"wikitext perplexity: {round(ppl,3)}")
    else:
        print("Unsupported Dataset")

if __name__ == "__main__":
    main()
