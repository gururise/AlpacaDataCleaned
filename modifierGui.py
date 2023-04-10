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
	
	def clamp(self, n, minn, maxn):
		return max(min(maxn, n), minn)

	def get_index(self, idx):
		idx = int(idx)
		idx = self.clamp(idx, 0, len(self.data)-1)
		self.index = idx
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

	def find_next_bad(self):
		for i in range(self.index+1,len(self.data)-1):
			text = f"""INSTRUCTION:\n{self.data[i]["instruction"]}\nINPUT:\n{self.data[i]["input"]}\nOUTPUT:\n{self.data[i]["output"]}"""
			bp = self.model.predict_proba([text],as_numpy=True)[0][1]
			print(bp)
			if (bp > 0.5):
				self.index = i
				return self.get_index(i)
		return -1,"","",""		
  
	def save_callback(self, idx, instruction='', input='', output=''):
		print("here)")
		idx = int(idx)
		print(idx)
		print(instruction)
		self.data[idx]["instruction"] = instruction
		print(self.data[idx])
		self.data[idx]["input"] = input
		self.data[idx]["output"] = output
		print(f"Index #{idx} Saved.")
		return None

	def export_callback(self):
		print(f"Exporting modified data to {SAVE_FILE}...", end = '')
		
		with open(SAVE_FILE, 'w', encoding='utf-8') as f:
			json.dump(self.data, f, indent = 4)

		self.modified = False
		print(f" Done.")
		return None

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
		self.modified = False
		self.size = len(self.data)
		self.reset_callback()

if __name__ == '__main__':
	modifier = MyAlpacaModifier()
	modifier.run()