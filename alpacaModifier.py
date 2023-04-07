import gradio as gr
import openai
from setfit import SetFitModel

OAI_PROMPT = "You are a helpful assistant. You answer in a concise and accurate manner. Your responses are short and to the point."

class AlpacaModifier:
	def __init__(self):
		self.input = ''
		self.instruction = ''
		self.old_output = ''
		self.modified_output = ''
		
	def ask_gpt(self, instruction='', input='', old_output='', modified_output='', key=''):
		openai.api_key = key

		composite_content = f"{instruction}\n\n{input}" if input else instruction
		print(f'Sending:\n{composite_content}')

		completion = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=[
					{"role": "system", "content": OAI_PROMPT},
					{"role": "user", "content": composite_content}
				]
		)
		modified_output = completion["choices"][0]["message"]["content"]
		return instruction, input, old_output, modified_output

	def run(self):
		
		with gr.Blocks() as demo:
			with gr.Column():
				gr.Markdown("""
				## 🦙 Alpaca Dataset Editor
				Cleaned Dataset: [Github](https://github.com/gururise/AlpacaDataCleaned) - [Hugging Face](https://huggingface.co/datasets/yahma/alpaca-cleaned)
				
				*To use GPT to generate answers, OpenAI API key is required*
				""")
				instruction_text = gr.Textbox(lines=2, label="Original Instruction", value=self.instruction, interactive=True)
				input_text = gr.Textbox(lines=1, label="Original Input", value=self.input, interactive=True)
				output_text = gr.Textbox(lines=15, label="Original Output", value=self.old_output, interactive=True)	
				bar_plot = gr.BarPlot(vertical=False,title="Predicted Quality",x="label",y="score", show_label=False,color="score",caption="",width=400)
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
    
			with gr.Row():
				button_previous = gr.Button(value="Previous")
				button_previous.click(self.previous_callback, 
					outputs=[index, instruction_text, input_text, output_text, bar_plot])
				button_next = gr.Button(value="Next")
				button_next.click(self.next_callback,  
					outputs=[index, instruction_text, input_text, output_text, bar_plot])
				button_save = gr.Button(value="Save")
				button_save.click(self.save_callback,
					inputs=[modified_instruction_text, modified_input_text, modified_output_text])
				button_save = gr.Button(value="Delete")
				button_save.click(self.delete_callback,
					outputs=[index, instruction_text, input_text, output_text])

			with gr.Row():
				gpt_api_key = gr.Textbox(label="API key", placeholder="Enter your OpenAI API Key (optional)")
				gpt_model = gr.Dropdown(label="OpenAI Model", choices=["gpt-3.5-turbo","text-davinci-003"],value="gpt-3.5-turbo")			
				button_ask_gpt = gr.Button(value="Ask GPT")
				button_ask_gpt.click(self.ask_gpt,
					inputs=[instruction_text, input_text, output_text, gpt_model, gpt_api_key], 
					outputs=[modified_instruction_text, modified_input_text, modified_output_text])
				

		demo.launch()