import os
import clip
import torch
import faiss
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# load clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# image paths
image_paths = ["./images/graph.png", "./images/workflow.png", "./images/dog.png", "./images/bunny.jpg", "./images/lll.png", "./images/population.png"]
image_embeddings = []

# convert images into embeddings
for img_path in image_paths:
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        image_embeddings.append(embedding.cpu().numpy())

# convert embeddings into numpy array
image_embeddings = np.vstack(image_embeddings)

# build FAISS vector
# IndexFlatIP = consin similarity
# IndexFlatL2 = Euclidean
embedding_dim = image_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(image_embeddings)

# compare query with images
query = "How did Digital twins helps in NAMAC architecture?"
with torch.no_grad():
    text_embedding = model.encode_text(clip.tokenize([query]).to(device))
    text_embedding = text_embedding.cpu().numpy()

# return the most similar image
k = 3
distances, indices = index.search(text_embedding, k)

for i in range(k):
    matched_image_path = image_paths[indices[0][i]]
    similarity_score = distances[0][i]
    print(f"{i+1}. Image Path: {matched_image_path}, Distance: {similarity_score}")
