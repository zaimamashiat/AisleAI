import pandas as pd
from groq import Groq

# -----------------------------
# 1️⃣ Initialize Groq client
# -----------------------------
client = Groq(api_key="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")

# -----------------------------
# 2️⃣ Load your dataset
# -----------------------------
file_path = r"C:\Users\zaima.nabi\Documents\Projects\customer_chatbot\dataset.xlsx"
df = pd.read_excel(file_path)

# Get size columns once at the module level
SIZE_COLUMNS = [col for col in df.columns if str(col).isdigit()]
print(f"Available size columns: {SIZE_COLUMNS}")
print(f"All columns in dataset: {list(df.columns)}")

# -----------------------------
# 3️⃣ Normalize color using Groq
# -----------------------------
def normalize_color(user_color_phrase, dataset_colors):
    if not user_color_phrase:
        return None

    prompt = f"""
You are an AI that maps vague user color descriptions to a dataset of available shoe colors.
Dataset colors: {dataset_colors}
User color description: '{user_color_phrase}'

Return the single closest matching color from the dataset.
- Use exact spelling from the dataset.
- If no exact match, pick the most similar one.
- Respond ONLY with the color name, nothing else.
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=50,
            top_p=1,
            stream=False
        )
        normalized_color = completion.choices[0].message.content.strip()
        if normalized_color not in dataset_colors:
            return user_color_phrase.lower()
        return normalized_color
    except Exception as e:
        print("Groq AI failed:", e)
        return user_color_phrase.lower()

# -----------------------------
# 4️⃣ Parse user query
# -----------------------------
def parse_user_query(query):
    query = query.lower()
    color_phrase = None
    size = None
    max_price = None
    category = None

    words = query.split()
    for i, w in enumerate(words):
        if w in ["size", "sz"]:
            if i + 1 < len(words):
                try:
                    size = int(words[i + 1])
                except:
                    pass
        if w in ["under", "<", "below"]:
            if i + 1 < len(words):
                try:
                    max_price = float(words[i + 1])
                except:
                    pass
        if w in ["gym", "lifestyle", "running", "formal"]:
            category = w

    stopwords = ["i","want","shoes","in","size","under","<","below","for", category]
    for w in words:
        if w not in stopwords:
            color_phrase = w
            break

    return color_phrase, size, max_price, category

# -----------------------------
# 5️⃣ Filter dataset with debug
# -----------------------------
def filter_shoes(color, size, max_price, category):
    filtered = df.copy()

    if category:
        filtered = filtered[filtered['Shoe Category'].str.lower() == category.lower()]
        print(f"After category filter: {len(filtered)}")

    if color:
        filtered = filtered[filtered['Colourway'].apply(lambda x: color.lower() in str(x).lower())]
        print(f"After color filter: {len(filtered)}")

    if size is not None:
        # Check if the size column exists
        if size in SIZE_COLUMNS:
            filtered = filtered[filtered[size] > 0]
            print(f"After size {size} filter: {len(filtered)}")
        else:
            print(f"⚠️ Size {size} column not found in dataset. Available sizes: {SIZE_COLUMNS}")
            return pd.DataFrame()  # Return empty DataFrame

    if max_price is not None:
        filtered = filtered[filtered['Price'] <= max_price]
        print(f"After price filter: {len(filtered)}")

    return filtered

# -----------------------------
# 6️⃣ Main function
# -----------------------------
def recommend_shoes(user_query):
    color_phrase, size, max_price, category = parse_user_query(user_query)
    print(f"Parsed query - Color: {color_phrase}, Size: {size}, Max Price: {max_price}, Category: {category}")

    dataset_colors = list({c.strip() for row in df['Colourway'] for c in eval(row)})
    normalized_color = normalize_color(color_phrase, dataset_colors)
    print(f"Normalized color: {normalized_color}")

    matches = filter_shoes(normalized_color, size, max_price, category)

    if matches.empty:
        return "No shoes found matching your criteria."

    # Limit to top 5 results
    matches = matches.head(5)
    
    response = []
    for idx, (_, shoe) in enumerate(matches.iterrows(), 1):
        available_sizes = []
        for s in SIZE_COLUMNS:
            if shoe[s] > 0:
                available_sizes.append(str(s))
        
        # Use 'Picture' column instead of 'Image'
        image_path = f"C:\\Users\\zaima.nabi\\Documents\\Projects\\customer_chatbot\\extracted_images\\{shoe['Picture']}"

        response.append(
            f"✅ Result {idx}/5: {shoe['Shoe Category'].title()} Shoe\n"
            f"   Colors: {shoe['Colourway']}\n"
            f"   Price: ৳{shoe['Price']}\n"
            f"   Sizes Available: {', '.join(available_sizes)}\n"
            f"   Image: {image_path}\n"
        )
    
    return "\n".join(response)

# -----------------------------
# 7️⃣ Example usage
# -----------------------------
if __name__ == "__main__":
    user_input = "I want grey lifestyle shoes size 42 under 1500"
    print("\n" + "="*60)
    print("SHOE RECOMMENDATION RESULTS")
    print("="*60 + "\n")
    print(recommend_shoes(user_input))