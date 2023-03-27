import json
import re

dataset = "alpaca_data.json"
# Load JSON data
with open(dataset, "r") as f:
    json_data = json.load(f)

# regex for "<noinput>", "No Input", "<No input>", "noinput", "<no input>"
noinput_pattern = re.compile(r"[\[\(\<]?no[ ]?input[\]\)\>\.]?", re.IGNORECASE)

# regex for "![any string](http" to detect if internet data is being passed into input
img_pattern = re.compile(r"!\[.*\]\(http|img[ ]?src=|image:")

issue_cnt = 0
# Loop through JSON data and output items that contain "input" elements matching the regex
print("<noinput> problems:")
for item in json_data:
    if "input" in item and noinput_pattern.search(item["input"]):
        print(item)
        issue_cnt += 1
print(f"Identified {issue_cnt} potential <noinput> issues.")

issue_cnt = 0
# Loop through JSON data and output items that contain "input" elements matching the regex
print("![alt text] problems:")
for item in json_data:
    if "input" in item and img_pattern.search(item["input"]):
        print(item)
        issue_cnt += 1
print(f"Identified {issue_cnt} potential ![alt text] issues.")
