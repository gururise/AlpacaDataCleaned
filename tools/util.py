import json

# Create an empty list to hold the objects
objects = []

# Loop 52,000 times and create an object with an "id" key that increments by 1 each time
for i in range(52000):
    obj = {"id": i}
    objects.append(obj)

# Write the objects to a JSON file
with open("index.json", "w") as outfile:
    json.dump(objects, outfile)