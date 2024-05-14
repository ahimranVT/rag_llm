import os
import pickle
import numpy as np
from helpers import embed, save_chunks, read_chunks, segment_chunk
from sklearn.metrics.pairwise import cosine_similarity

chunks = []
cleaned_data_filepath = "ITU_data.txt"
embeddings_filepath = "embeddings.pkl"
embedding_model_context = 250

chunk_dict = {}
embedding_dict = {}

with open(cleaned_data_filepath, 'r', encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:
        chunks.append(line)

if not os.path.isfile(embeddings_filepath):
    for i, chunk in enumerate(chunks[:4]):
        if len(chunk) > embedding_model_context:
            subchunk_embeddings = []

            subchunks = segment_chunk(chunk, embedding_model_context)
            subchunk_embeddings = [embed(subchunk) for subchunk in subchunks]

            embedding_dict[i] = subchunk_embeddings
        else:
            embedding_dict[i] = [embed(chunk)]

        chunk_dict[str(i)] = chunk
        
    save_chunks(chunk_dict, "chunks.csv")

    with open(embeddings_filepath, 'wb') as file:
        pickle.dump(embedding_dict, file)

else:
    chunk_dict = read_chunks(chunk_dict, "chunks.csv")

    with open(embeddings_filepath, 'rb') as file:
        embedding_dict = pickle.load(file)

    # print(f" The loaded embedding_dict has {len(embedding_dict)} embedding lists.")

    # for embedding in embedding_dict.values():
    #     print(f"Each {type(embedding)} contains vectors for one chunk. This chunk has {len(embedding)} subchunks")
    #     print(f"This subchunk is a {type(embedding[0])} with {len(embedding[0])} vectors")
    #     print("____________________________________________________________________________________")

def retrieval(query, embedding_dict, chunk_dict, n):
    most_similar_chunks = {}

    query_embedding = embed(query)

    for chunk_id, embedding_list in embedding_dict.items():
        current_chunk_similarity = []
        
        for subchunk_embedding in embedding_list:
            similarity = cosine_similarity([query_embedding], [subchunk_embedding])
            current_chunk_similarity.append(similarity[0][0])

        most_similar_chunks[chunk_id] = max(current_chunk_similarity)

    most_similar_chunks = dict(sorted(most_similar_chunks.items(), key=lambda item: item[1], reverse=True)[:n])

    for key, val in most_similar_chunks.items():
        print(str(key) + ":", val)
        print(chunk_dict[str(key)] + '\n')

    return most_similar_chunks

retrieval("What is the name of teacher who is graduated from LUMS", embedding_dict, chunk_dict, 1)



                 



                 

