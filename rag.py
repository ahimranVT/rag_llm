import os
import requests
import json
from bs4 import BeautifulSoup
import numpy as np
from helpers import embed, save_chunks, read_chunks
from transformers import BertModel, BertTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# Scrape data from test_url using Beautiful soup 
scraping_URL = "https://itu.edu.pk/cet/courses/"
scraping_data_file_path = "data.txt"
embeddings_file_path = "embeddings.npy"
chunks_file_path = "chunks.npy"

page = requests.get(scraping_URL)
soup = BeautifulSoup(page.content, "html.parser")
text = soup.get_text()

# Overwrite existing file, put scraped data in data.txt
if os.path.exists(scraping_data_file_path):
    os.remove(scraping_data_file_path)

with open(scraping_data_file_path, "w", encoding="utf-8") as file:
    file.write(text)
file.close()

# Remove new line characters and spacing from text in data.txt
with open(scraping_data_file_path, "r+", encoding="utf-8") as file:
    lines = file.readlines()
    file.seek(0)

    # file.writelines(line for line in lines if line.strip())
    text = ''.join(line for line in lines if line.strip())
    file.write(text)

    file.truncate()

# Split the stored text into chunks
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=750, chunk_overlap=0, separators=["\n", "."]
)
chunks = text_splitter.split_text(text)

chunk_dict = {}
embedding_list = []

if not os.path.isfile(embeddings_file_path):
    for i, chunk in enumerate(chunks[:10]):
        chunk_dict[str(i)] = chunk
        embedded_chunk = embed(chunk)
        embedding_list.append(embedded_chunk)

    embedding_list = [np.array(embedding) for embedding in embedding_list]

    save_chunks(chunk_dict, "chunks.csv")
    np.save("embeddings.npy", embedding_list)
else:
    embedding_list = np.load(embeddings_file_path, allow_pickle=True)
    read_chunks(chunk_dict, "chunks.csv")


def retrieval(query, embedding_list, chunk_dict, n):
    most_similar_chunks = {}

    query_embedding = embed(query)

    for i, embedding in enumerate(embedding_list):
        similarity = cosine_similarity([query_embedding], [embedding])
        most_similar_chunks[i] = similarity[0][0] 

    most_similar_chunks = dict(sorted(most_similar_chunks.items(), key=lambda item: item[1], reverse=True)[:n])

    for key, val in most_similar_chunks.items():
        print(str(key) + ":", val)
        print('\n')
        print(chunk_dict[str(key)] + '\n')

    return most_similar_chunks

print(len(chunk_dict))
retrieval("Who holds a BS in Computer Science and worked with Software", embedding_list, chunk_dict, 10)


















# print('\n' + response.json()['choices'][0]['text'] + '\n')



# Tokenize the chunks
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenized_chunks = []
# for chunk in chunks:
#   tokenized_chunks.append(tokenizer.tokenize(chunk))









# from dotenv import load_dotenv
# load_dotenv()

# # Configure the LLM
# hf_token = "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"
# model_id = "sentence-transformers/all-MiniLM-L6-v2"

# url = "https://api.awanllm.com/v1/completions"  

# payload = json.dumps({
#   "model": "Meta-Llama-3-8B-Instruct",
#   "prompt": "What is the meaning of life?"
# })
# headers = {
#   'Content-Type': 'application/json',
#   'Authorization': f"Bearer {'f947d77f-f534-4ff4-b03d-eaca09d8243d'	}"
# }
