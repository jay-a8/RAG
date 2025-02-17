import requests
import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_llava(image_path, prompt="You are a consice and accurte AI assistence. Now, what do you see in the image I provided, and what does it imply?"):
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


import os
image_path = "./images/graph.png"
# print(os.path.exists(image_path))
description = ask_llava(image_path)
print("LLaVA's Response:", description)  # output
