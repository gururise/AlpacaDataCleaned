import re
import json
from alpacaModifier import AlpacaModifier

# The regex
REGEX_INSTRUCTION = r'^.*$'
REGEX_INPUT = r'^([\d\^\+\-\/yx=,\(\) ]|\\u00b\d)+$'
REGEX_OUTPUT = r'^.*$'

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
			if match_instruction and match_input and match_output:
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
		return self.generator.__next__()
		
	def save_callback(self, instruction='', input='', old_output='', modified_output=''):
		self.fill_values(instruction, input, old_output, modified_output)
		self.save_prev_item()
		
		print(f"Saving modified data to {SAVE_FILE}...", end = '')
		
		with open(SAVE_FILE, 'w') as f:
			json.dump(self.data, f, indent = 4)

		print(f" Done.")

	def reset_callback(self, instruction='', input='', old_output='', modified_output=''):
		self.fill_values(instruction, input, old_output, modified_output)
		self.save_prev_item()
		self.generator = self.next_generator()
		return self.next_callback()

	def decimal_to_roman(self, number):
		roman_numerals = {
			1000: 'M',
			900: 'CM',
			500: 'D',
			400: 'CD',
			100: 'C',
			90: 'XC',
			50: 'L',
			40: 'XL',
			10: 'X',
			9: 'IX',
			5: 'V',
			4: 'IV',
			1: 'I'
		}
		
		roman = ''
		for value, numeral in roman_numerals.items():
			while number >= value:
				roman += numeral
				number -= value
		
		return roman

	def modify_output(self, match, item):
		low_instr = item['instruction'].lower()

		number_strings = re.findall(r'\d+', item['input'])
		if number_strings:
			numbers = [int(num_str) for num_str in number_strings]

		if 'mean' in low_instr or 'the average' in low_instr:
			mean_val = sum(numbers)/len(numbers)

			return f"""The mean of the given list is equal to ({' + '.join(number_strings)})/{len(numbers)} = {sum(numbers)}/{len(numbers)} = {mean_val}."""
		elif 'median' in low_instr:
			sorted_numbers = sorted(numbers)
			length = len(sorted_numbers)
			mid = length // 2
			if length % 2 == 0:
				median = (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
				median_string = f'({sorted_numbers[mid - 1]} + {sorted_numbers[mid]})/2 = {median}'
			else:
				median = sorted_numbers[mid]

			return f"""The sorted list is {sorted_numbers}. Therefore, the median is {median_string if length % 2 == 0 else str(median)}."""
		elif 'roman' in low_instr:
			if len(numbers) > 1:
				result = ', '.join([self.decimal_to_roman(num) for num in numbers])
				return f'{number_strings} in Roman numerals are {result}.'
			else:
				return f'{number_strings[0]} in Roman numerals is {self.decimal_to_roman(numbers[0])}.'
		elif 'binary' in low_instr and 'decimal' not in low_instr:
			if len(numbers) > 1:
				result = ', '.join([bin(num)[2:] for num in numbers])
				return f'The binary representation of {number_strings} are {result}.'
			else:
				return f'The binary representation of {number_strings[0]} is {bin(numbers[0])[2:]}.'
		elif 'hexadecimal' in low_instr:
			if len(numbers) > 1:
				result = ', '.join([hex(num) for num in numbers])
				return f'The hexidecimal representation of {number_strings} are {result}.'
			else:
				return f'The hexidecimal representation of {number_strings[0]} is {hex(numbers[0])}.'

		return item['output']

	def __init__(self):
		super().__init__()

		with open('alpaca_data_cleaned.json', 'r') as f:
			self.data = json.load(f)
		
		self.pattern_instruction = re.compile(REGEX_INSTRUCTION)
		self.pattern_input = re.compile(REGEX_INPUT)
		self.pattern_output = re.compile(REGEX_OUTPUT)

		self.prev_item = None
		self.reset_callback()

if __name__ == '__main__':
	modifier = MyAlpacaModifier()
	modifier.run()