import re
import json
from alpacaModifier import AlpacaModifier
from setfit import SetFitModel
import pandas as pd

SAVE_FILE = 'alpaca_new.json'
def swap_rows(df, i1, i2):
    a, b = df.iloc[i1, :].copy(), df.iloc[i2, :].copy()
    df.iloc[i1, :], df.iloc[i2, :] = b, a
    return df

class MyAlpacaModifier(AlpacaModifier):
	index = 0
	def fill_values(self, instruction='', input='', output='', modified_instruction='', modified_input='', modified_output=''):
		self.instruction = instruction
		self.input = input
		self.output = output
		self.modified_output = modified_output

	def save_prev_item(self):
		if self.prev_item != None:
			self.prev_item['instruction'] = self.instruction
			self.prev_item['input'] = self.input
			self.prev_item['output'] = self.modified_output
			print(f'saving as:\n{self.prev_item}')
			self.prev_item = None
	
	def clamp(self, n, minn, maxn):
		return max(min(maxn, n), minn)

	def get_index(self, idx):
		idx = int(idx)
		idx = self.clamp(idx, 0, len(self.data)-1)
		val = self.check_garbage_collector(self.data[idx]['instruction'], self.data[idx]['input'], self.data[idx]['output'])
		df = pd.DataFrame(val)
		
		df = swap_rows(df,0,1)
		return idx, self.data[idx]['instruction'], self.data[idx]['input'], self.data[idx]['output'], df

	def next_callback(self):
		self.index += 1
		if self.index > len(self.data)-1:
			self.index = 0
		return self.get_index(self.index)

	def previous_callback(self):
		self.index -= 1
		if self.index < 0:
			self.index = len(self.data)-1
		return self.get_index(self.index)

	def delete_callback(self):
		# delete current index
		del self.data[self.index]
		if self.index > len(self.data):
			self.index = 0
		return self.get_index(self.index)
		
	def save_callback(self, instruction='', input='', old_output='', modified_output=''):
		self.fill_values(instruction, input, old_output, modified_output)
		self.save_prev_item()
		
		print(f"Saving modified data to {SAVE_FILE}...", end = '')
		
		with open(SAVE_FILE, 'w', encoding='utf-8') as f:
			json.dump(self.data, f, indent = 4)

		print(f" Done.")

	def get_num_matches(self):
		res = 0
		for item in self.data:
			res += (re.search(self.pattern_instruction, item['instruction']) or 
					re.search(self.pattern_input, item['input']) or
					re.search(self.pattern_output, item['output'])) != None
		return res

	def reset_callback(self):
		self.index = 0
		return self.get_index(self.index)

	def check_garbage_collector(self, instruction,input,output):
		labels = ["GOOD", "BAD"]
		text = f"""INSTRUCTION:\n{instruction}\nINPUT:\n{input}\nOUTPUT:\n{output}"""
		probas = self.model.predict_proba([text], as_numpy=True)
		return [{"label":labels[0],"score":probas[0][0]},{"label":labels[1],"score":probas[0][1]}]

	def __init__(self):
		super().__init__()

		with open('alpaca_data_cleaned.json', 'r', encoding='utf-8') as f:
			self.data = json.load(f)
			self.model = SetFitModel.from_pretrained(
  				"argilla/alpaca-garbage-collector-multilingual"
			)

		self.prev_item = None
		self.size = len(self.data)
		self.reset_callback()

if __name__ == '__main__':
	modifier = MyAlpacaModifier()
	modifier.run()