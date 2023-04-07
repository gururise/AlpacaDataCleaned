import gradio as gr
import openai
import json
from setfit import SetFitModel

OAI_PROMPT = "You are a helpful assistant. You answer in a concise and accurate manner. Your responses are short and to the point."

class AlpacaModifier:
	def __init__(self):
		self.input = ''
		self.instruction = ''
		self.old_output = ''
		self.modified_output = ''

	def ask_gpt(self, instruction='', input='', output='', model='', key=''):
		openai.api_key = key

		composite_content = f"""Given the following JSON prompt for a large language model, your task is to suggest a better prompt in the same JSON format.
{{
    "instruction": "{instruction}",
    "input": "{input}",
    "output: "{output}"
}}

Improved Prompt:
"""
		print(f'Sending:\n{composite_content}')

		completion = openai.ChatCompletion.create(
			model=model,
			messages=[
					{"role": "system", "content": OAI_PROMPT},
					{"role": "user", "content": composite_content}
				]
		)
		gpt_out = None
		try:
			gpt_out = json.loads(completion["choices"][0]["message"]["content"])
		except:
			pass
		if gpt_out:
			return gpt_out["instruction"],gpt_out["input"],gpt_out["output"]
		return "","",""

	def run(self):
		with gr.Blocks(css=".warning {background-color: yellow}; .danger {background-color: red}", themes=gr.themes.Soft()) as demo:
			with gr.Column():
				gr.Markdown("""
				## 🦙 Alpaca Dataset Editor
				Cleaned Dataset: [Github](https://github.com/gururise/AlpacaDataCleaned) - [Hugging Face](https://huggingface.co/datasets/yahma/alpaca-cleaned)
				
				*To use GPT to generate answers, OpenAI API key is required*
				""")
				instruction_text = gr.Textbox(lines=2, label="Original Instruction", value=self.instruction, interactive=True, elem_classes="")
				input_text = gr.Textbox(lines=1, label="Original Input", value=self.input, interactive=True, elem_classes="")
				output_text = gr.Textbox(lines=15, label="Original Output", value=self.old_output, interactive=True, elem_classes="")	
				bar_plot = gr.BarPlot(vertical=False,title="Predicted Quality",x="label",y="score", y_lim=[0,1.0], show_label=False,color="score",caption="",width=400)
    
				with gr.Accordion("GPT Suggestion", open=False):
					modified_instruction_text = gr.Textbox(lines=2, label="Suggested Instruction", value=self.instruction, interactive=False)
					modified_input_text = gr.Textbox(lines=1, label="Suggested Input", value=self.input, interactive=False)
					modified_output_text = gr.Textbox(lines=5, label="Suggested Output", value=self.modified_output, interactive=False)

			with gr.Row():
				index = gr.Number(label="Index", value=0, every=1, interactive=True)
				index.change(self.get_index,
					inputs=[index], 
					outputs=[index, instruction_text, input_text, output_text, bar_plot])
				button_reset = gr.Button(value="Reset Index")
				button_reset.click(self.reset_callback, 
					outputs=[index, instruction_text, input_text, output_text, bar_plot])
				button_reset = gr.Button(value="Find Next Bad")
				button_reset.click(self.find_next_bad, 
					outputs=[index, instruction_text, input_text, output_text, bar_plot])
    
			with gr.Row():
				button_previous = gr.Button(value="Previous")
				button_previous.click(self.previous_callback, 
					outputs=[index, instruction_text, input_text, output_text, bar_plot])
				button_next = gr.Button(value="Next")
				button_next.click(self.next_callback,
					outputs=[index, instruction_text, input_text, output_text, bar_plot])
				button_save = gr.Button(value="Save Entry")
				button_save.click(self.save_callback,
					inputs=[index, instruction_text, input_text, output_text])
				button_delete= gr.Button(value="Delete Entry")
				button_delete.click(self.delete_callback,
					outputs=[index, instruction_text, input_text, output_text])
				button_export = gr.Button(value="Export File")
				button_export.click(self.export_callback)

			with gr.Row():
				gpt_api_key = gr.Textbox(label="API key", placeholder="Enter your OpenAI API Key (optional)")
				gpt_model = gr.Dropdown(label="OpenAI Model", choices=["gpt-3.5-turbo","text-davinci-003"],value="gpt-3.5-turbo")			
				button_ask_gpt = gr.Button(value="Ask GPT")
				button_ask_gpt.click(self.ask_gpt,
					inputs=[instruction_text, input_text, output_text, gpt_model, gpt_api_key], 
					outputs=[modified_instruction_text, modified_input_text, modified_output_text])
				

		demo.launch()