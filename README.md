# Cleaned Alpaca Dataset
Welcome to the Cleaned Alpaca Dataset repository! This repository hosts a cleaned and curated version of a dataset used to train the Alpaca LLM (Language Learning Model). The original dataset had several issues that were addressed in this cleaned version.

## Dataset Quality and its Impact on Model Performance

One of the theories behind the lack of a significant improvement in performance from fine-tuning the 7B Alpaca model to the 13B model is the quality of the original dataset. The original dataset used to train the GPT model was generated with GPT-3, which itself may have had limitations due to data quality. Additionally, fine-tuning on the original dataset resulted in [poor loss curves](assets/train_loss.png).

The quality of the dataset plays a crucial role in determining the performance of the natural language processing models trained on it. A dataset that is noisy, inconsistent, or incomplete can result in poor performance even with the most advanced models. In contrast, a high-quality dataset can enable a model to perform well with smaller parameters.

Therefore, it is possible that with better data, we could improve the performance of the models more than what would be gained by simply increasing model size. By providing a cleaned and curated dataset for fine-tuning language models, we can help researchers and developers to better understand the true impact of the model size on performance, independent of the quality of the dataset.

## Data Cleaning and Curation
The Alpaca LLM is a language model designed to learn and understand code snippets and their associated natural language descriptions. It is trained on a dataset that consists of pairs of code snippets and their corresponding descriptions. The original dataset used to train the Alpaca LLM was found to have many issues that impacted its quality and usefulness for training a machine learning model.
Issues with the Original Dataset

The original dataset had several issues that made it difficult to use for training the Alpaca LLM. These issues include:

1. **Empty outputs:** Some code snippets in the original dataset had empty outputs, which made it impossible to evaluate the correctness of the code.
2. **Empty code examples:** Some descriptions in the original dataset were missing code examples, making it difficult to understand the intended behavior of the code.
3. **Instructions to generate images:** Some descriptions in the original dataset included instructions to generate images, which made it challenging to use the data for training the Alpaca LLM.
4. **N/A outputs:** Some code snippets in the original dataset had N/A outputs, which made it impossible to evaluate the correctness of the code.
5. **Inconsistent input field:** The original dataset had inconsistent empty input fields, which made it difficult to process the data.
6. **Wrong answers:** Some descriptions in the original dataset had incorrect answers, which made it challenging to use the data for training the Alpaca LLM.
7. **References to data on the internet:** Some descriptions in the original dataset had instructions that referenced data on the internet, which made it difficult to use the data for training the Alpaca LLM.
8. **Extraneous escape and control characters:** The original dataset had several entries with extraneous escape and control characters.

## Goals
The primary goal of this project is to provide a cleaned and curated version of the Alpaca LLM dataset that will improve the performance of natural language processing models trained on this data. By removing errors and inconsistencies, this cleaned dataset will allow for better fine-tuning of language models and reduce the likelihood of hallucinations.

## Acknowledgments
The original version of the Alpaca dataset was sourced from tatsu-lab's [github repository](https://github.com/tatsu-lab/stanford_alpaca). We would like to thank the original creators of these datasets for making their data available to the public. We would also like to thank the team at Meta AI for their work in developing the [Llama LLM](https://github.com/facebookresearch/llama), which was trained using this dataset.