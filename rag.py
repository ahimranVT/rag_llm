import requests
import json
from bs4 import BeautifulSoup
import os
from transformers import BertModel, BertTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# load_dotenv()

# Configure the LLM
url = "https://api.awanllm.com/v1/completions"  

payload = json.dumps({
  "model": "Meta-Llama-3-8B-Instruct",
  "prompt": "What is the meaning of life?"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {'f947d77f-f534-4ff4-b03d-eaca09d8243d'	}"
}

# Scrape data from test_url using Beautiful soup 
test_URL = "https://itu.edu.pk/cet/courses/"
file_path = "data.txt"

page = requests.get(test_URL)
soup = BeautifulSoup(page.content, "html.parser")
text = soup.get_text()

# Overwrite existing file, put scraped data in data.txt
if os.path.exists(file_path):
    os.remove(file_path)

with open(file_path, "w", encoding="utf-8") as file:
    file.write(text)
file.close()

# Remove new line characters and spacing from text in data.txt
with open(file_path, "r+", encoding="utf-8") as file:
    lines = file.readlines()
    file.seek(0)

    # file.writelines(line for line in lines if line.strip())
    text = ''.join(line for line in lines if line.strip())
    file.write(text)

    file.truncate()

# Split the stored text into chunks
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=3500, chunk_overlap=0, separators=[" ", ",", "\n", "."]
)
chunks = text_splitter.split_text(text)

# Tokenize the chunks
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_chunks = []
for chunk in chunks:
  tokenized_chunks.append(tokenizer.tokenize(chunk))

print(len(tokenized_chunks))



















# print('\n' + response.json()['choices'][0]['text'] + '\n')