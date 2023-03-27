import gradio as gr

class AlpacaModifier:
    def __init__(self):
        self.input = ''
        self.instruction = ''
        self.old_output = ''
        self.modified_output = ''
        
    def next_callback(self):
        # returns the next instruction_text, input_text, old_output_text, modified_output_text
        pass
        
    def save_callback(self):
        # When this is called, all the changes done until this moment will be saved.
        pass
        
    def modify_output(self):
        # Automatically modify the output in some way or just return it as it is.
        pass
        
    def run(self):
        with gr.Blocks() as demo:
            instruction_text = gr.Textbox(lines=2, label="Instruction", value=self.instruction, interactive=True)
            input_text = gr.Textbox(lines=1, label="Input", value=self.input, interactive=True)
            old_output_text = gr.Textbox(lines=2, label="Old Output", value=self.old_output, interactive=False)
            modified_output_text = gr.Textbox(lines=10, label="Modified Output", value=self.modified_output, interactive=True)
            
            button_next = gr.Button(value="Next")
            button_next.click(self.next_callback, 
                inputs=[instruction_text, input_text, old_output_text, modified_output_text], 
                outputs=[instruction_text, input_text, old_output_text, modified_output_text])
            button_save = gr.Button(value="Save")
            button_save.click(self.save_callback,
                inputs=[instruction_text, input_text, old_output_text, modified_output_text])

        demo.queue(concurrency_count=3)
        demo.launch()