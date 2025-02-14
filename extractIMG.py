# # 1. Use a Vision-Language Model.to process the image and provide embeddings.

# # Load an image into the model (no text required).
# # Generate an embedding that represents the visual content.
# # Query the model to answer questions about the image.


# # https://python.langchain.com/docs/integrations/document_loaders/image/
# # https://blog.langchain.dev/semi-structured-multi-modal-rag/

# # Multimodal Model can work better with this

# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import torch
# import ollama


# # load CLIP model - Contrastive Language-Image Pre-Training
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # load img
# img = Image.open("bunny.jpg")

# # calculate embedding
# inputs = clip_processor(images=img, return_tensors="pt")
# with torch.no_grad():
#     image_embedding = clip_model.get_image_features(**inputs)

# # Normalization embedding 
# # to scaling high-dimensional vectors so that their length becomes 1, thereby ensuring that the calculation of cosine similarity is not affected by the vector length.
# image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

# # print(image_embedding.shape)  # torch.Size([1, 512])

# # Convert embedding to string
# embedding_text = ", ".join(map(str, image_embedding.squeeze().tolist()))

# # Let Ollama process the embedding and infer the image content
# query = f"""
# These numbers represent a CLIP image embedding, which is a high-dimensional vector that captures the semantic meaning of an image.
# CLIP embeddings are trained to align images and text, meaning that images with similar content have similar embeddings.

# Here is the CLIP embedding of an image:
# [{embedding_text}]

# Based on this embedding, what do you think is in the image?
# """
# response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": query}])

# print(response['message']['content'])
import requests
import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_llava(image_path, prompt="You are a consice and accurte AI assistence. Now, Based on this Konowledge Graph, can you tell me whats on the Knowleage graph and resoning to me what concerns trustworthiness has, please?"):
    image_base64 = encode_image(image_path)

    with requests.post("http://localhost:11434/api/generate",
                       json={"model": "llava", "prompt": prompt, "images": [image_base64]},
                       stream=True) as response:  # enable stream response
        full_response = ""  # use to store response
        for line in response.iter_lines():  # anaylsis per line
            if line:
                data = json.loads(line)  # anaylsis JSON
                full_response += data["response"]  # connect response

        return full_response  # return compelete response 

image_path = "flowchart2.png"
description = ask_llava(image_path)
print("LLaVA's Response:", description)  # output
