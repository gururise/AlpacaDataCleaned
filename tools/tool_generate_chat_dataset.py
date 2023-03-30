import json
import sys
import os
import time
import argparse
from typing import List

import openai
import openai.error as openai_error

openai.api_key = "Enter your OpenAI API Key Here"

CHAT_MODEL = "gpt-3.5-turbo"

# Initialize the ArgumentParser object
parser = argparse.ArgumentParser(description='Generate Output using ChatGPT')

def read_alpaca_dataset(filename: str, num_samples: int, starting_sample: int = 0) -> json:
    """
    Read Alpaca data from a JSON file.

    Args:
        filename: The path to the JSON file.
        num_samples: The number of samples to read.
        starting_sample: The index of the first sample to read.

    Returns:
        A pandas DataFrame containing the data.
    """
    
    print(f"Reading {filename}...")
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
     
    if starting_sample < 0 or starting_sample > num_samples:
        raise ValueError(f"Invalid starting_sample: {starting_sample}. Must be between 0 and the number of samples.")
        
    if num_samples > 0:
        print(f"Total number of samples: {len(data)}")
        print(f"Using {num_samples} samples. Starting sample: {starting_sample}")
        data = data[starting_sample:num_samples]
    
    if num_samples < 0:
        print(f"Number of samples must be greater than zero.")
        exit()
    print("Read data successfully.")
    return data

def openai_gpt(prompt: str, verbose: bool = False, max_attempts: int = 3) -> str:
    """
    This function sends a prompt to the OpenAI GPT API and returns the response.
    It tries the creation several times (max_attempts) in case of exception.
    If the model is text-davinci-003, it uses the Completion API, otherwise it uses the ChatCompletion API.

    Args:
        prompt (str): Prompt to send to the API.
        config (_type_): Configuration object.
        verbose (bool, optional): If True, print the prompt and response. Defaults to False.
        max_attempts (int, optional): Number of attempts to make in case of exception. Defaults to 3.

    Returns:
        str: The response from the API.
    """
    # send the prompt to gpt and return the response
    # try the creation several times in case of exception
    for attempt in range(1, max_attempts + 1):
        try:
            messages = [
                {"role": "system", "content": "Given the following instruction and optional input, generate a helpful response."},
                {"role": "user", "content": f"{prompt}"},
            ]
            response = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=3072,
                temperature=0.5,
                top_p=1.0,
                stop=["\n20", "20.", "20."]
            )
            choices = [choice["message"]["content"] for choice in response["choices"]]

            if verbose:
                print("*" * 20)
                print(f"Model: {CHAT_MODEL}")
                print(f"Prompt: {prompt}")
                print(f"Chat Response: {response['choices'][0]['message']['content']}")

            return choices[0]
        except openai_error.OpenAIError as e:
            if attempt < max_attempts:
                print(f"Error on attempt {attempt}: {e}. Retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(f"Error on attempt {attempt}: {e}. All attempts failed.")
                # we will return None if all attempts failed because raising an exception will stop the program and we will lose all the data we have collected so far
                return None
            
def chat_completions(dataset, num_samples, output_file, verbose):
    """ Generate completions using the Chat Model

    Args:
        dataset (str): Filename of the dataset to read
        num_samples (int): How many instructions in the dataset to process
        output_file (str): output filename
        verbose (bool, optional): If True, print the prompt and response. Defaults to False.
    """
    # generate the prompt for the row
    rows = read_alpaca_dataset(dataset,num_samples)
    num_samples = min(num_samples,len(rows))
    
    with open(output_file,"w") as outfile:
        outfile.write("[\n")
        for i in range(num_samples):
            print (f"Working on {i+1} of {num_samples}")
            prompt = "instruction: '" + rows[i]['instruction'] + "'\ninput: '" + rows[i]['input'] + "'"
            rows[i]['output'] = openai_gpt(prompt,verbose)
            if rows == None:
                break
            data = json.dumps(rows[i], indent=4)
            outfile.write(f"{data}")
            if i < num_samples-1:
                outfile.write(",\n")
        outfile.write("\n]")
        
def main():
    global parser
    
    # Define the arguments
    parser.add_argument('--dataset', type=str, default="alpaca_data_cleaned.json", help='Alpaca Dataset name')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples')
    parser.add_argument('--output_file', type=str, default="chat_dataset.json", help="output filename")
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose output")
    
    # Parse the arguments
    args = parser.parse_args()
    chat_completions(args.dataset, args.num_samples, args.output_file, args.verbose)

if __name__ == "__main__":
    main()