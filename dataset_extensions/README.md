# Dataset Extensions in Alpaca Format
The purpose of these extensions are to make it easy to further fine-tune a base LLAMA model trained on the Alpaca Cleaned Dataset. All the following datasets have been converted to the alpaca JSON format. They can be merged with a base alpaca dataset, or used for further fine-tuning. These datasets have not been curated and may contain offensive data.

## Open Instruction Generalist (OIG) Small Chip2 (~200,000)
Released by LAION-AI, the [chip2 dataset](https://github.com/LAION-AI/Open-Instruction-Generalist/tree/main/small_instruction_set) is a small high-quality dataset with the purpose of making it easy to convert a language model pretrained on large amounts of text into an instruction following model using a small amount of additional compute via finetuning or softprompt tuning. The unified chip consists of the following examples:
* Python Code Examples
* Natural Instruction Examples
* Generic Harmless Instruction Examples
* Instruction/Responses with Lists
* Follow up questions
* Wikipedia Toxic Adversarial Questions
* Grade School Math
* Reasoning Instructions
* Character and Scene Descriptions

More information at [LAION OIG Blog](https://laion.ai/blog/oig-dataset/). I took a quick look through this dataset, and it appears to have many duplicates and some questionable content. Use at your own risk.

License: Apache 2.0

## Grade School Math 8k (~7,500)
GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems created by human problem writers. The dataset is segmented into 7.5K training problems and 1K test problems. I've only converted the training problems. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. A bright middle school student should be able to solve every problem. It can be used for multi-step mathematical reasoning. (https://github.com/openai/grade-school-math)

License: Permissive Unknown

## GPTeacher
Github User @teknium1 released the [GPTeacher Dataset](https://github.com/teknium1/GPTeacher):
> A collection of modular datasets generated by GPT-4, General-Instruct - Roleplay-Instruct - Code-Instruct - and Toolformer

>The General-Instruct used many of the same seed prompts as alpaca, but also had specific examples of things we didnt see much in with alpaca. Such as Chain of Thought Reasoning, Logic Puzzles, Wordplay, Role Playing (lightly), and was asked to include reasoning behind and thought steps where appropriate in example responses, among other things. The General-Instruct dataset is about 20,000 examples with just deduplication.

The only change was modifying 'response' field to 'output' to match ALPACA's JSON format.

License: MIT
Additional Restriction(s): OpenAI [TOS](https://openai.com/policies/terms-of-use)

## Databricks Dolly 15k

An open source dataset of instruction-following records used in training databricks/dolly-v2-12b that was generated by thousands of Databricks employees in several of the behavioral categories outlined in the InstructGPT paper, including brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization. The file `databricks-dolly-15k-parsed.json` contains the parsed JSON in alpaca format with wikipedia context removed per the Databricks suggestion in their README.

Blog post: [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)

License: CC BY-SA 3.0 - Copyright 2023 Databricks, Inc.