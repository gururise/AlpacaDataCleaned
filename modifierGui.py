import re
import json
from alpacaModifier import AlpacaModifier

# The regex
REGEX_INSTRUCTION = r'(?:http|https|www\.|\.com|\.org|\.net|\.edu)'
REGEX_INPUT = r'(?:http|https|www\.|\.com|\.org|\.net|\.edu)'
REGEX_OUTPUT = r'(?:http|https|www\.|\.com|\.org|\.net|\.edu)'

SAVE_FILE = 'alpaca_new.json'

class MyAlpacaModifier(AlpacaModifier):
	def fill_values(self, instruction='', input='', old_output='', modified_output=''):
		self.instruction = instruction
		self.input = input
		self.old_output = old_output
		self.modified_output = modified_output

	def save_prev_item(self):
		if self.prev_item != None:
			self.prev_item['instruction'] = self.instruction
			self.prev_item['input'] = self.input
			self.prev_item['output'] = self.modified_output
			print(f'saving as:\n{self.prev_item}')
			self.prev_item = None

	def next_generator(self):
		for item in self.data:
			self.save_prev_item()
			
			match_instruction = re.search(self.pattern_instruction, item['instruction'])
			match_input = re.search(self.pattern_input, item['input'])
			match_output = re.search(self.pattern_output, item['output'])
			if match_instruction or match_input or match_output:
				new_instruction = item['instruction']
				new_input = item['input']
				old_output = item['output']
				modified_output = self.modify_output(match_output, item)

				self.fill_values(new_instruction, new_input, old_output, modified_output)
				self.prev_item = item
				yield self.instruction, self.input, self.old_output, self.modified_output

		self.save_prev_item()
		self.fill_values(instruction='End of matches reached.')

		while True:
			yield self.instruction, self.input, self.old_output, self.modified_output

	def next_callback(self, instruction='', input='', old_output='', modified_output=''):
		self.fill_values(instruction, input, old_output, modified_output)
		print(f'Currently processing match {self.current_match}/{self.num_matches}')
		self.current_match += 1
		return self.generator.__next__()
		
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

	def reset_callback(self, instruction='', input='', old_output='', modified_output=''):
		self.fill_values(instruction, input, old_output, modified_output)
		self.save_prev_item()
		self.generator = self.next_generator()
		self.num_matches = self.get_num_matches()
		self.current_match = 1
		return self.next_callback()

	def modify_output(self, match, item):
		return item['output']

	def __init__(self):
		super().__init__()

		with open('alpaca_data_cleaned.json', 'r', encoding='utf-8') as f:
			self.data = json.load(f)
		
		self.pattern_instruction = re.compile(REGEX_INSTRUCTION)
		self.pattern_input = re.compile(REGEX_INPUT)
		self.pattern_output = re.compile(REGEX_OUTPUT)

		self.prev_item = None
		self.reset_callback()

if __name__ == '__main__':
	modifier = MyAlpacaModifier()
	modifier.run()