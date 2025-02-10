from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# 1️⃣ 加载 CLIP 模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2️⃣ 读取并预处理图片
image = Image.open("bunny.jpg")
inputs = processor(images=image, return_tensors="pt")

# 3️⃣ 获取图片的 embedding
with torch.no_grad():
    image_embedding = model.get_image_features(**inputs)

# 4️⃣ 归一化 embedding
image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

print(image_embedding.shape)  # 512 维向量


# 1️⃣ 处理用户问题
query_text = "这张图片里有什么？"
inputs = processor(text=query_text, return_tensors="pt")

# 2️⃣ 计算问题的 embedding
with torch.no_grad():
    query_embedding = model.get_text_features(**inputs)

# 3️⃣ 归一化
query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

print(query_embedding.shape)  # 512 维向量


# import torch.nn.functional as F

# # 计算余弦相似度
# cosine_similarity = F.cosine_similarity(image_embedding, query_embedding)

# print(f"图片和问题的相似度: {cosine_similarity.item()}")


from langchain.llms import OpenAI

# 1️⃣ 让 LLM 知道图片的 embedding（格式化为文本）
image_description = f"这张图片的向量表示是 {image_embedding.tolist()}，请描述它的内容。"

# 2️⃣ 让 GPT-4 生成回答
llm = OpenAI(model_name="gpt-4")
response = llm(image_description)

print("AI 生成的答案:", response)
