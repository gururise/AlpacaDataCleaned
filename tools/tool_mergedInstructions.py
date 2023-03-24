import json
import re

dataset = "alpaca_data.json"
# Load JSON data
with open(dataset, "r") as f:
    json_data = json.load(f)

# regex for locating 'merged' instructions. Seem to mostly be in the output
noinput_pattern = re.compile(r"\d{1,2}\.\sInstruction:", re.IGNORECASE)

issue_cnt = 0
# Loop through JSON data and output items that contain "input" elements matching the regex
print("Merged Instruction problems:")
for item in json_data:
    if "output" in item and noinput_pattern.search(item["output"]):
        print(item)
        issue_cnt += 1
print(f"Identified {issue_cnt} potential merged instructions.")
