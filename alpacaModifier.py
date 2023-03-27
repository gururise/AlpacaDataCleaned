import gradio as gr
import openai

class AlpacaModifier:
	def __init__(self):
		self.input = ''
		self.instruction = ''
		self.old_output = ''
		self.modified_output = ''
		
	def next_callback(self, instruction='', input='', old_output='', modified_output=''):
		# returns the next instruction_text, input_text, old_output_text, modified_output_text.
		pass
		
	def save_callback(self, instruction='', input='', old_output='', modified_output=''):
		# When this is called, all the changes done until this moment will be saved.
		pass
	
	def reset_callback(self, instruction='', input='', old_output='', modified_output=''):
		# Reset to the begining of the file.
		pass

	def skip_ahead(self, steps, instruction='', input='', old_output='', modified_output=''):
		while steps > 1:
			steps -= 1
			instruction, input, old_output, modified_output = self.next_callback(instruction, input, old_output, old_output)
		if steps == 1:
			return self.next_callback(instruction, input, old_output, old_output)
		return instruction, input, old_output, modified_output
		
	def ask_gpt(self, instruction='', input='', old_output='', modified_output=''):
		openai.api_key = ""
		completion = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=[
					{"role": "system", "content": "You are a helpful assistant. You answer in a consise and accurate manner. Your responses are short and to the point."},
					{"role": "user", "content": f"{instruction}\n\n{input}"}
				]
		)
		modified_output = completion["choices"][0]["message"]["content"]
		return instruction, input, old_output, modified_output

	def modify_output(self):
		# Automatically modify the output in some way or just return it as it is.
		pass

	def run(self):
		with gr.Blocks() as demo:
			instruction_text = gr.Textbox(lines=2, label="Instruction", value=self.instruction, interactive=True)
			input_text = gr.Textbox(lines=1, label="Input", value=self.input, interactive=True)
			old_output_text = gr.Textbox(lines=2, label="Old Output", value=self.old_output, interactive=False)
			modified_output_text = gr.Textbox(lines=10, label="Modified Output", value=self.modified_output, interactive=True)
			
			with gr.Row():
				button_next = gr.Button(value="Next")
				button_next.click(self.next_callback, 
					inputs=[instruction_text, input_text, old_output_text, modified_output_text], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])
				button_save = gr.Button(value="Save")
				button_save.click(self.save_callback,
					inputs=[instruction_text, input_text, old_output_text, modified_output_text])
				button_reset = gr.Button(value="Reset")
				button_reset.click(self.reset_callback,
					inputs=[instruction_text, input_text, old_output_text, modified_output_text], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])

			with gr.Row():
				skip_ahead = gr.Number(value=0, interactive=True)
				button_skip = gr.Button(value="Skip Ahead")
				button_skip.click(self.skip_ahead,
					inputs=[skip_ahead, instruction_text, input_text, old_output_text, modified_output_text], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])
			with gr.Row():
				button_ask_gpt = gr.Button(value="Ask GPT")
				button_ask_gpt.click(self.ask_gpt,
					inputs=[instruction_text, input_text, old_output_text, modified_output_text], 
					outputs=[instruction_text, input_text, old_output_text, modified_output_text])

		demo.launch()