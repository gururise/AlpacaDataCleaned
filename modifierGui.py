import re
import json
from alpacaModifier import AlpacaModifier

# Can be played with.
MIN_NUMBER_OF_ITEMS_IN_LIST = 4
# There are quite a few questions of type: 'Sort the following list in alphabetical order.' We usually don't want to format those items.
CHECK_IF_SORT_QUESTION = True
# Check if there is a number in the instruction. Using numbered lists helps the LLM correctly answer queries that want a specific number of answers.
CREATE_NUMBERED_LIST_IF_NUMBER_IN_INSTRUCTION = True
NUM_MAP = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

SAVE_FILE = 'alpaca_new.json'

class MyAlpacaModifier(AlpacaModifier):
	def save_prev_item(self):
		if self.prev_item != None:
			self.prev_item['output'] = self.modified_output
			print(f'saving as:\n{self.prev_item}')
			self.prev_item = None

	def next_generator(self):
		for item in self.data:
			self.save_prev_item()
			if CHECK_IF_SORT_QUESTION and 'sort' in item['instruction'].lower():
				continue
			
			match = re.search(self.pattern, item['output'])
			if match:
				self.instruction = item['instruction']
				self.input = item['input']
				self.old_output = item['output']
				self.modified_output = self.modify_output(match)
				self.prev_item = item
				yield self.instruction, self.input, self.old_output, self.modified_output

		self.save_prev_item()

		self.instruction = 'End of matches reached.'
		self.input = ''
		self.old_output = ''
		self.modified_output = ''

		yield self.instruction, self.input, self.old_output, self.modified_output

	def next_callback(self, instruction='', input='', old_output='', modified_output=''):
		self.instruction = instruction
		self.input = input
		self.old_output = old_output
		self.modified_output = modified_output
		return self.generator.__next__()
		
	def save_callback(self, instruction='', input='', old_output='', modified_output=''):
		self.instruction = instruction
		self.input = input
		self.old_output = old_output
		self.modified_output = modified_output
		self.save_prev_item()
		
		print(f"Saving modified data to {SAVE_FILE}...", end = '')
		
		with open(SAVE_FILE, 'w') as f:
			json.dump(self.data, f, indent = 4)

		print(f"Done.")
		
	def modify_output(self, match):
		comma_sep_list = match.group(1)
		md_list = re.sub(r', (and |or )?', '\n- ', comma_sep_list)
		md_list = '\n'.join(['- ' + x.strip().title() for x in md_list.split('\n- ')])
		
		if CREATE_NUMBERED_LIST_IF_NUMBER_IN_INSTRUCTION:
			num_match = re.search(r'\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b', self.instruction.lower())
			if num_match:
				num_items = num_match.group(0)
				if num_items.isdigit():
					num_items = int(num_items)
				else:
					num_items = NUM_MAP.get(num_match.group(0), -1)
				md_list_items = md_list[2:].split('\n- ')
				md_list = '\n'.join([f'{i+1}. {x.title()}' for i, x in enumerate(md_list_items)])

		return self.old_output.replace(match.group(0), md_list)

	def __init__(self):
		super().__init__()

		with open('alpaca_data_cleaned.json', 'r') as f:
			self.data = json.load(f)
		
		self.pattern = re.compile(r'^((?:(?:[a-zA-Z]+[ \-]){0,2}[a-zA-Z]+, ){' + str(MIN_NUMBER_OF_ITEMS_IN_LIST-1) + r',}(?:and |or )?(?:[a-zA-Z]+[ \-]){0,2}[a-zA-Z]+)\.?$')
		self.prev_item = None
		self.generator = self.next_generator()
		self.next_callback()

if __name__ == '__main__':
    modifier = MyAlpacaModifier()
    modifier.run()