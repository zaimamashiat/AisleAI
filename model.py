# ===============================================
# BERT-Based AI Shoe Recommendation System
# ===============================================

import pandas as pd
import ast
import random
import torch
from torch.utils.data import Dataset
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from IPython.display import display, Image
import os

# ---------------------------
# 1. Configuration
# ---------------------------
EXCEL_PATH = r"C:\Users\zaima.nabi\Documents\Projects\Shoes\dataset.xlsx"
IMAGE_FOLDER = r"C:\Users\zaima.nabi\Documents\Projects\Shoes\extracted_images"
MAX_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3

# ---------------------------
# 2. Load & preprocess dataset
# ---------------------------
df = pd.read_excel(EXCEL_PATH)

# Convert Colourway column from string to list
df['Colourway'] = df['Colourway'].apply(ast.literal_eval)

# Create a combined description field
df['description'] = df.apply(
    lambda x: ' '.join([' '.join(x['Colourway']), x['Shoe Category'], f"Price {x['Price']}"]),
    axis=1
)

# Add full image path
df['Image_Path'] = df['Picture'].apply(lambda x: os.path.join(IMAGE_FOLDER, x))

# ---------------------------
# 3. Prepare dataset for BERT fine-tuning
# ---------------------------
query_templates = [
    "Can you show me {color} {category} shoes?",
    "I want {color} {category} sneakers.",
    "Show me {color} shoes for {category}.",
    "Do you have {color} {category} shoes?",
    "Looking for {color} {category} shoes."
]

train_data = []

for _, row in df.iterrows():
    colors = row['Colourway']
    category = row['Shoe Category']
    # Positive examples
    for template in query_templates:
        color = random.choice(colors)
        query = template.format(color=color, category=category)
        train_data.append({"query": query, "shoe_description": row['description'], "label": 1})
    
    # Negative examples: random shoes not matching this query
    negative_samples = df.sample(2)
    for _, neg in negative_samples.iterrows():
        train_data.append({"query": query, "shoe_description": neg['description'], "label": 0})

train_df = pd.DataFrame(train_data)

# ---------------------------
# 4. Create PyTorch Dataset
# ---------------------------
class ShoeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.queries = df['query'].tolist()
        self.descriptions = df['shoe_description'].tolist()
        self.labels = df['label'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            self.descriptions[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = ShoeDataset(train_df, tokenizer, MAX_LEN)

# ---------------------------
# 5. Fine-tune BERT
# ---------------------------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./bert_shoe_model',
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=2e-5
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

print("Starting BERT fine-tuning...")
trainer.train()
print("Fine-tuning completed!")

# ---------------------------
# 6. Recommendation function
# ---------------------------
def recommend_shoes(query, df, model, tokenizer, top_n=5):
    """
    Input: natural language query
    Output: top N shoes with images
    """
    scores = []
    for _, row in df.iterrows():
        encoding = tokenizer(
            query, row['description'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(**encoding)
            probs = softmax(outputs.logits, dim=1)
            scores.append(probs[0][1].item())  # probability of match
    
    df['score'] = scores
    top_shoes = df.sort_values('score', ascending=False).head(top_n)
    
    print(f"Top {top_n} recommendations for: '{query}'\n")
    for idx, row in top_shoes.iterrows():
        print(f"{row['Shoe Category'].title()} - Colors: {row['Colourway']} - Price: {row['Price']}")
        display(Image(filename=row['Image_Path']))

# ---------------------------
# 7. Test the model
# ---------------------------
recommend_shoes("Can you show me yellow gym shoes?", df, model, tokenizer, top_n=5)
recommend_shoes("I want chocolate lifestyle sneakers under 1200", df, model, tokenizer, top_n=5)
recommend_shoes("Bright running shoes in size 42", df, model, tokenizer, top_n=5)
