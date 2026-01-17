import json

with open('llama_shoe_dataset.json', 'r') as f:
    data = json.load(f)

for item in data:
    if 'image_path' in item:
        path = item['image_path']
        # Extract image number from the path
        image_name = path.split('image_')[-1]
        # Rebuild with correct escaping
        item['image_path'] = 'C:\\Users\\zaima\\OneDrive\\Documents\\GitHub\\AisleAI\\extracted_images\\image_' + image_name

with open('llama_shoe_dataset.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Fixed all image paths!")
