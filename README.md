# 🦙🛁 Cleaned Alpaca Dataset
Welcome to the Cleaned Alpaca Dataset repository! This repository hosts a cleaned and curated version of a dataset used to train the Alpaca LLM (Large Language Model). The original dataset had several issues that are addressed in this cleaned version.

## Dataset Quality and its Impact on Model Performance

One of the theories behind the lack of a significant improvement in performance from fine-tuning the 7B Alpaca model to the 13B model is the quality of the original dataset. The original dataset used to train the GPT model was generated with GPT-3, which itself may have had limitations due to data quality. Additionally, fine-tuning on the original dataset resulted in [poor loss curves](https://twitter.com/abacaj/status/1637310768780648448).

The quality of the dataset plays a crucial role in determining the performance of the natural language processing models trained on it. A dataset that is noisy, inconsistent, or incomplete can result in poor performance even with the most advanced models. In contrast, a high-quality dataset can enable a model to perform well with smaller parameters.

Therefore, it is possible that with better data, we could improve the performance of the models more than what would be gained by simply increasing model size. By providing a cleaned and curated dataset for fine-tuning language models, we can help researchers and developers to better understand the true impact of the model size on performance, independent of the quality of the dataset.

## Data Cleaning and Curation
The Alpaca LLM is a language model designed to learn and understand code snippets and their associated natural language descriptions. It is trained on a dataset that consists of pairs of code snippets and their corresponding descriptions. The original dataset used to train the Alpaca LLM was found to have many issues that impacted its quality and usefulness for training a machine learning model.
Issues with the Original Dataset

The original dataset had several issues that likely affect the final performance of the fine-tuned model. These issues include:

1. **Hallucinations:** Many instructions in the original dataset had instructions referencing data on the internet, which just caused GPT3 to hallucinate an answer.
```
"instruction":"Summarize the given article in 200 Words.",
"input": "https://www.bbc.com/news/world-51461830",
"output": "The recent protests in Belarus are part of a growing wave of discontent against the rule of Alexander Lukashenko, who has led the country since 1994. Demonstrations against his government have been ongoing since August, and come in the wake of an election widely seen as rigged. As police and security forces have cracked down on protesters, the movement has only grown, with hundreds of thousands of people taking to the streets to call for new leadership. Human rights activists have accused the government of using extreme violence in order prosecute protesters, leading to an international outcry and sanctions from the European Union. As the situation in Belarus continues to unfold, the international community is monitoring the situation closely - hoping for a peaceful resolution, while also standing firm against the oppressive tactics employed by the Belarusian state."

"instruction": "Classify the following shape according to simple geometry.",
"input": "<Picture Attached>",
"output": "Triangle"
```
2. **Empty outputs:** Some entries in the original dataset had empty outputs.
```
"instruction": "Calculate the area of the following shape in square centimeters.",
"input": "rectangle of size 4 cm x 5 cm",
"output": ""
```
3. **Empty code examples:** Some descriptions in the original dataset were missing code examples, making it difficult to understand the intended behavior of the code.
4. **Instructions to generate images:** Some descriptions in the original dataset included instructions to generate images, something obviously not possible.
```
"instruction": "Create a graphic or logo that visually represents the word \"courage\".",
"input": "",
"output": "<No Output>"
```
5. **N/A outputs:** Some code snippets in the original dataset had N/A outputs.
6. **Inconsistent input field:** The original dataset had inconsistent usage of the input field when it was supposed to be empty.
```
"input":"<no input>"
"input":"No input"
"input":"noinput"
"input":"<noinput>"
```
7. **Wrong answers:** Some instructions/questions in the original dataset had incorrect answers.
8. **Extraneous escape and control characters:** The original dataset had several entries with extraneous escape and control characters.

## Contributions
With over 52k entries, several issues still exist. Please help out by submitting a pull-request.

## Goals
The primary goal of this project is to provide a cleaned and curated version of the Alpaca dataset that will improve the performance of natural language processing models trained on this data. By removing errors and inconsistencies, the goal is to improve performance of the fine-tuned llama models and reduce the likelihood of hallucinations.

## Acknowledgments
The original version of the Alpaca dataset was sourced from tatsu-lab's [github repository](https://github.com/tatsu-lab/stanford_alpaca). We would like to thank the original creators of these datasets for making their data available to the public. We would also like to thank the team at Meta AI for their work in developing the [Llama LLM](https://github.com/facebookresearch/llama), which was trained using this dataset.