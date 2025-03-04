import os
import json
import base64
import requests

file_path = "data2.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text_content = file.read()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_llava(image_path, prompt=f"""You are a consice and accurte AI assistence. Now, 
    {text_content}
    use the part that is helpful in text to combine with image to answer that:What is happening in the section labeled “A”?

"""):

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


image_path = "./images/tcp.png"
# print(os.path.exists(image_path))
description = ask_llava(image_path)
print("LLaVA's Response:", description)  # output
