import requests
import json
from bs4 import BeautifulSoup
import os
from transformers import BertModel, BertTokenizer

# from dotenv import load_dotenv

# load_dotenv()

url = "https://api.awanllm.com/v1/completions"  

payload = json.dumps({
  "model": "Meta-Llama-3-8B-Instruct",
  "prompt": "What is the meaning of life?"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {'f947d77f-f534-4ff4-b03d-eaca09d8243d'	}"
}

# response = requests.request("POST", url, headers=headers, data=payload)

# print('\n' + response.json()['choices'][0]['text'] + '\n')

test_URL = "https://itu.edu.pk/cet/courses/"
page = requests.get(test_URL)
soup = BeautifulSoup(page.content, "html.parser")
text = soup.get_text()

file_path = "data.txt"

if os.path.exists(file_path):
    os.remove(file_path)

with open(file_path, "w", encoding="utf-8") as file:
    file.write(text)
file.close()

# with open(file_path, "r+", encoding="utf-8") as file:
#     lines = file.readlines()
#     file.seek(0)

#     file.writelines(line for line in lines if line.strip())
#     file.truncate()

with open(file_path, "r+", encoding="utf-8") as file:
    lines = file.readlines()
    file.seek(0)

    text = ''.join(line for line in lines if line.strip())
    print(type(text))
    file.write(text)
    file.truncate()