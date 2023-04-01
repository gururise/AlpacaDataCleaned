import json

objects = []

for i in range(52000):
    obj = {"id": i}
    objects.append(obj)

with open("alpaca_index.json", "w") as outfile:
    json.dump(objects, outfile)