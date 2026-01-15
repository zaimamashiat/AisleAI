import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import os
from tqdm import tqdm

# ----------------------------
# 1. Paths
# ----------------------------
DATA_PATH = r"C:\Users\zaima.nabi\Documents\Projects\Shoes\dataset.xlsx"
IMAGE_FOLDER = r"C:\Users\zaima.nabi\Documents\Projects\Shoes\extracted_images"
EMBEDDINGS_FOLDER = r"C:\Users\zaima.nabi\Documents\Projects\Shoes\embeddings"
JSON_OUTPUT = r"C:\Users\zaima.nabi\Documents\Projects\Shoes\llama_shoe_dataset.json"
CACHE_FILE = r"C:\Users\zaima.nabi\Documents\Projects\Shoes\embeddings_cache.pt"

os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

# ----------------------------
# 2. Load dataset
# ----------------------------
df = pd.read_excel(DATA_PATH)
print(f"Loaded {len(df)} shoes from dataset")

# Clean up the Number column - use index if Number is missing
if 'Number' not in df.columns or df['Number'].isna().all():
    df['Number'] = range(len(df))
    print("Warning: 'Number' column missing or empty, using row index")
else:
    # Fill missing Number values with index
    df['Number'] = df['Number'].fillna(pd.Series(range(len(df)), index=df.index))

# Ensure Picture column exists
if 'Picture' not in df.columns:
    print("ERROR: 'Picture' column not found in dataset!")
    exit(1)

# Clean Price column - convert to numeric, handle "stock out" and other text
if 'Price' in df.columns:
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    print(f"Price column cleaned: {(df['Price'] == 0).sum()} items with missing/invalid prices")

# Standardize size columns to strings '38' -> '45' and convert to numeric
size_cols = []
for s in range(38, 46):
    col_name = str(s) if str(s) in df.columns else s
    if col_name in df.columns:
        df[str(s)] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
        if col_name != str(s):
            df.drop(columns=[col_name], inplace=True)
        size_cols.append(str(s))
    else:
        print(f"Warning: Column {s} not found, skipping.")

# Lowercase color names if stored as list string
def parse_colors(c):
    try:
        return [x.lower() for x in eval(c) if isinstance(x, str)]
    except:
        return []

df['Colourway'] = df['Colourway'].apply(parse_colors)

# ----------------------------
# 3. Load CLIP model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_embedding(path):
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        return None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs).cpu()

def text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_text_features(**inputs).cpu()

# ----------------------------
# 4. Convert row to text
# ----------------------------
def tabular_to_text(row):
    colors = ", ".join(row['Colourway']) if row['Colourway'] else "various colors"
    sizes_available = " ".join([s for s in size_cols if pd.notna(row[s]) and row[s] > 0])
    price = row['Price'] if pd.notna(row['Price']) else "N/A"
    category = row['Shoe Category'] if pd.notna(row['Shoe Category']) else "shoe"
    return f"{category} shoe, colors: {colors}, price {price}, sizes available: {sizes_available}"

# ----------------------------
# 5. Generate embeddings & JSON (with skip existing)
# ----------------------------
def generate_embeddings_and_json(force_regenerate=False):
    """Generate embeddings and JSON, skipping existing files unless forced."""
    
    output_json = []
    successful = 0
    skipped = 0
    failed = 0
    
    print(f"\n{'='*80}")
    print("GENERATING EMBEDDINGS AND JSON")
    print(f"{'='*80}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing shoes"):
        image_path = os.path.join(IMAGE_FOLDER, row['Picture'])
        emb_file = os.path.join(EMBEDDINGS_FOLDER, f"{row['Number']}_embedding.pt")
        
        # Skip if embedding already exists and not forcing regeneration
        if os.path.exists(emb_file) and not force_regenerate:
            skipped += 1
            # Still add to JSON
            if row['Colourway']:  # Only if colors exist
                json_row = create_json_entry(row, image_path)
                output_json.append(json_row)
            continue
        
        # Generate embeddings
        img_emb = image_embedding(image_path)
        if img_emb is None:
            failed += 1
            continue
            
        text_emb = text_embedding(tabular_to_text(row))
        combined_emb = (img_emb + text_emb) / 2

        torch.save(combined_emb, emb_file)
        successful += 1
        
        # Generate fine-tune JSON
        if row['Colourway']:  # Only if colors exist
            json_row = create_json_entry(row, image_path)
            output_json.append(json_row)

    # Save JSON
    with open(JSON_OUTPUT, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Embeddings: {successful} generated, {skipped} skipped, {failed} failed")
    print(f"JSON entries: {len(output_json)}")
    print(f"Embeddings saved to: {EMBEDDINGS_FOLDER}")
    print(f"JSON saved to: {JSON_OUTPUT}")
    print(f"{'='*80}\n")

def create_json_entry(row, image_path):
    """Helper function to create JSON entry for a shoe."""
    # Safe conversion helpers
    def safe_int(val):
        try:
            return int(float(val)) if pd.notna(val) else None
        except (ValueError, TypeError):
            return None
    
    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def safe_str(val, default="N/A"):
        return str(val) if pd.notna(val) and str(val).lower() not in ['nan', 'none', ''] else default
    
    return {
        "instruction": "Recommend shoes based on user query",
        "input": f"User query: I want {row['Colourway'][0] if row['Colourway'] else 'colorful'} {safe_str(row.get('Shoe Category', 'shoe'))} shoes in size 42 under 3000",
        "output": f"Best match: {tabular_to_text(row)}",
        "image_path": image_path,
        "metadata": {
            "Number": safe_int(row['Number']),
            "Price": safe_float(row['Price']),
            "Sizes": {s: safe_int(row[s]) for s in size_cols},
            "Category": safe_str(row.get('Shoe Category', 'Unknown')),
            "Colourway": row['Colourway'] if row['Colourway'] else [],
            "Code": safe_str(row.get('Code', 'N/A'))
        }
    }

# ----------------------------
# 6. Optimized retrieval with caching
# ----------------------------
class ShoeRecommender:
    def __init__(self):
        self.embeddings_cache = None
        self.load_embeddings_cache()
    
    def load_embeddings_cache(self):
        """Load all embeddings into memory once."""
        if os.path.exists(CACHE_FILE):
            print("Loading embeddings from cache file...")
            self.embeddings_cache = torch.load(CACHE_FILE)
            print(f"Loaded {len(self.embeddings_cache)} embeddings from cache")
        else:
            print("Building embeddings cache...")
            self.embeddings_cache = {}
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings"):
                emb_file = os.path.join(EMBEDDINGS_FOLDER, f"{row['Number']}_embedding.pt")
                if os.path.exists(emb_file):
                    self.embeddings_cache[idx] = torch.load(emb_file)
            
            # Save cache for next time
            torch.save(self.embeddings_cache, CACHE_FILE)
            print(f"Cached {len(self.embeddings_cache)} embeddings to {CACHE_FILE}")
    
    def recommend(self, query, top_k=5, price_max=None, price_min=None, 
                  size_required=None, category=None, colors=None):
        """
        Recommend shoes based on query with optional filters.
        
        Args:
            query: Text description of desired shoes
            top_k: Number of recommendations to return
            price_max: Maximum price filter
            price_min: Minimum price filter
            size_required: Size availability filter (e.g., '42')
            category: Shoe category filter (e.g., 'sneakers')
            colors: List of required colors (e.g., ['red', 'black'])
        """
        q_emb = text_embedding(query)
        
        similarities = []
        filtered_out = 0
        
        for idx, row in df.iterrows():
            if idx not in self.embeddings_cache:
                continue
            
            # Apply filters with null checks
            if price_max and pd.notna(row['Price']) and row['Price'] > price_max:
                filtered_out += 1
                continue
            if price_min and pd.notna(row['Price']) and row['Price'] < price_min:
                filtered_out += 1
                continue
            if size_required and (pd.isna(row.get(str(size_required))) or row.get(str(size_required), 0) <= 0):
                filtered_out += 1
                continue
            if category and pd.notna(row['Shoe Category']) and category.lower() not in row['Shoe Category'].lower():
                filtered_out += 1
                continue
            if colors:
                row_colors = [c.lower() for c in row['Colourway']]
                if not any(c.lower() in row_colors for c in colors):
                    filtered_out += 1
                    continue
            
            shoe_emb = self.embeddings_cache[idx]
            sim = torch.nn.functional.cosine_similarity(q_emb, shoe_emb)
            similarities.append((sim.item(), idx))
        
        # Sort and get top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS: '{query}'")
        print(f"{'='*80}")
        print(f"Total matches: {len(similarities)} | Filtered out: {filtered_out}")
        print(f"\nTop {min(top_k, len(similarities))} recommendations:")
        print("-" * 80)
        
        if not similarities:
            print("No shoes found matching your criteria.")
            return []
        
        results = []
        for rank, (score, idx) in enumerate(similarities[:top_k], 1):
            row = df.loc[idx]
            colors_str = ", ".join(row['Colourway']) if row['Colourway'] else "N/A"
            available_sizes = [s for s in size_cols if pd.notna(row[s]) and row[s] > 0]
            
            result = {
                'rank': rank,
                'category': row['Shoe Category'] if pd.notna(row['Shoe Category']) else 'Unknown',
                'colors': row['Colourway'] if row['Colourway'] else [],
                'price': row['Price'] if pd.notna(row['Price']) else 0.0,
                'sizes': available_sizes,
                'code': row['Code'] if pd.notna(row['Code']) else 'N/A',
                'image': row['Picture'] if pd.notna(row['Picture']) else 'N/A',
                'score': score,
                'number': row['Number'] if pd.notna(row['Number']) else idx
            }
            results.append(result)
            
            print(f"{rank}. {result['category'].upper()}")
            print(f"   Colors: {colors_str}")
            print(f"   Price: {result['price']} BDT")
            print(f"   Available Sizes: {', '.join(available_sizes) if available_sizes else 'None'}")
            print(f"   Code: {result['code']}")
            print(f"   Image: {result['image']}")
            print(f"   Match Score: {score:.4f}")
            print()
        
        return results

# ----------------------------
# 7. Main execution
# ----------------------------
if __name__ == "__main__":
    # Generate embeddings (skip if already exist)
    generate_embeddings_and_json(force_regenerate=False)
    
    # Initialize recommender
    recommender = ShoeRecommender()
    
    # Example queries
    print("\n" + "="*80)
    print("SHOE RECOMMENDATION SYSTEM - READY")
    print("="*80)
    
    # Query 1: With filters
    print("\n[Query 1] Yellow gym shoes size 42 under 3000 BDT")
    recommender.recommend(
        "Yellow gym shoes", 
        top_k=5, 
        price_max=3000, 
        size_required=42
    )
    
    # Query 2: Category specific
    print("\n[Query 2] Black sneakers")
    recommender.recommend("Black sneakers", top_k=3, colors=['black'])
    
    # Query 3: Price range
    print("\n[Query 3] Expensive running shoes")
    recommender.recommend("Running shoes", top_k=3, price_min=5000)