import requests
import numpy as np
import csv
import os
from dotenv import load_dotenv


load_dotenv()

hf_token = os.getenv("HF_TOKEN")
model_id = os.getenv("EMBEDDING_MODEL_ID")

# Creates (384,) embeddings for a given chunk of text
def embed(texts):

    embed_model_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    embed_model_headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(embed_model_url, headers=embed_model_headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    embedding = np.array(response.json())
    return embedding

# Saves chunks in a dictionary to a csv file
def save_chunks(chunk_dict, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        for chunk in chunk_dict.values():
            f.write(chunk)

# Reads the dictionary of chunks from the csv file
def read_chunks(chunk_dict, filename):
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for idx, chunk in enumerate(reader):
            chunk_dict[str(idx)] = chunk[0]
    return chunk_dict

# Segments chunks greater than the context of the embedding model into
# subchunks and returns them
def segment_chunk(chunk, segment_length):
    chunk_segments = []
    for i in range(0, len(chunk), segment_length):
        start = i
        end = min(i + segment_length, len(chunk))
        segment = chunk[start:end]
        chunk_segments.append(segment)
    return chunk_segments
