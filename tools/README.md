# Tools

The tools for helping curate and clean the alpaca dataset include:

1. **tool_automatic_check.py** - Asks GPT-3.5 or ChatGPT whether an output is correct for the given instruction and input. Generates a CSV list of potential issues with the passed in dataset.

2. **tool_generate_chat_dataset.py** - Uses ChatGPT to take the input dataset and generate outputs. Using this tool, you can regenerate the alpaca dataset using ChatGPT instead of GPT-3.

3. **tool_inputcheck.py** - checks the 'input' fields of the dataset for potential issues.

4. **tool_mergedInstructions.py** - checks the dataset for merged or combined instructions.

5. **tool_manage_dataset.py** - Compares the orignal alpaca dataet to the cleaned dataset and attempts to identify instructions in the cleaned dataset that have not been modified and therefore, potentially need to be checked.