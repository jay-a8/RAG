from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np

# Load CLIP model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess the image
image = Image.open("bunny.jpg")  # Replace with your image file
inputs = processor(images=image, return_tensors="pt", padding=True)

# Get image embeddings (feature vectors)
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# Normalize the image features
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

# Simulating a database of image embeddings (just for illustration)
# In a real case, you would already have a collection of image embeddings
image_database = np.random.randn(10, 512).astype('float32')  # 10 random image embeddings for example

# FAISS index to search for the most similar image
index = faiss.IndexFlatL2(512)  # 512 is the dimension of CLIP's image embedding
index.add(image_database)  # Add the database to FAISS

# Query the FAISS index for the most similar image
D, I = index.search(image_features.numpy(), k=2)  # Get top 2 most similar images

# Simulate retrieving the description of the most similar image
# In reality, you'd want to match this to an actual description in your database
print(f"Top 2 similar image indices: {I[0]}")
print(f"Top 2 similarity scores: {D[0]}")

# Example query
query = "What is in the image?"

# Process the query text
text_inputs = processor(text=[query], return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

# Normalize the text features
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# Compute similarity between the image and the query
similarity = (text_features @ image_features.T).squeeze().numpy()

# Print the similarity
print(f"Similarity score between query and image: {similarity}")
