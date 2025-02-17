from transformers import LlavaProcessor, LlavaForConditionalGeneration

from pdf2image import convert_from_path
from PIL import Image

import os

pdf_path = "./pdf/NAMAC.pdf"
images = convert_from_path(pdf_path)
first_page = images[0]

# check llava-hf / llava-1.5-7b-hf

processor = LlavaProcessor.from_pretrained("llava-hf/llava-13b")
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-13b")


# 让 Llava 处理图片
inputs = processor(text="这张图片的内容是什么？", images=first_page, return_tensors="pt")
output = model.generate(**inputs)
print(processor.decode(output[0], skip_special_tokens=True))
