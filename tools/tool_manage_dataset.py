import json

DATA_ORIGINAL_PATH = "alpaca_data.json"         # Path to original dataset
DATA_CLEANED_PATH = "alpaca_data_cleaned.json"  # Path to cleaned dataset
START = 0                                       # Starting index
END = 10000                                     # Ending index
PRINT_CLEANED = False                           # True: print all cleaned items, False: print all potentially to be cleaned items

# Load original dataset
with open(DATA_ORIGINAL_PATH, encoding="utf-8") as file_raw:
    data_original = json.load(file_raw)

# Load cleaned dataset
with open(DATA_CLEANED_PATH, encoding="utf-8") as file_cleaned:
    data_cleaned = json.load(file_cleaned)

# Iterate over cleaned dataset and print item based on variable "PRINT_CLEANED"
# Then print number of all items who met condition
count = 0
for i in range(START, END):
    item = data_cleaned[i]
    if PRINT_CLEANED:
        if item in data_original:
            count += 1
            print(f"index: {i}")
            print(json.dumps(item, indent=4))
            print("="*10)
    else:
        if item not in data_original:
            count += 1
            print(f"index: {i}")
            print(json.dumps(item, indent=4))
            print("="*10)

if PRINT_CLEANED:
    print(f"NUMBER OF CLEANED ITEMS: {count}")
else:
    print(f"NUMBER OF POTENTIALLY TO BE CLEANED ITEMS: {count}")