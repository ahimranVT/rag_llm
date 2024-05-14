import os
import pickle
import requests
import json
from dotenv import load_dotenv
from helpers import embed, save_chunks, read_chunks, segment_chunk
from sklearn.metrics.pairwise import cosine_similarity

chunks = []
cleaned_data_filepath = "ITU_data.txt"
embeddings_filepath = "embeddings.pkl"
text_chunks_file_path = "chunks.csv"
embedding_model_context = 250

chunk_dict = {}
embedding_dict = {}

load_dotenv()

llm_url = os.getenv("LLM_URL")
llm_name = os.getenv("LLM_NAME")
llm_auth_token = os.getenv("LLM_AUTH_TOKEN")

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
        
    save_chunks(chunk_dict, text_chunks_file_path)

    with open(embeddings_filepath, 'wb') as file:
        pickle.dump(embedding_dict, file)

else:
    chunk_dict = read_chunks(chunk_dict, text_chunks_file_path)

    with open(embeddings_filepath, 'rb') as file:
        embedding_dict = pickle.load(file)

    # print(f" The loaded embedding_dict has {len(embedding_dict)} embedding lists.")

    # for embedding in embedding_dict.values():
    #     print(f"Each {type(embedding)} contains vectors for one chunk. This chunk has {len(embedding)} subchunks")
    #     print(f"This subchunk is a {type(embedding[0])} with {len(embedding[0])} vectors")
    #     print("____________________________________________________________________________________")

def retrieval(query, embedding_dict, chunk_dict, n):
    best_similarity_scores = {}

    query_embedding = embed(query)

    for chunk_id, embedding_list in embedding_dict.items():
        current_chunk_similarity = []
        
        for subchunk_embedding in embedding_list:
            similarity = cosine_similarity([query_embedding], [subchunk_embedding])
            current_chunk_similarity.append(similarity[0][0])

        best_similarity_scores[chunk_id] = max(current_chunk_similarity)

    best_similarity_scores = dict(sorted(best_similarity_scores.items(), key=lambda item: item[1], reverse=True)[:n])

    # for key, val in most_similar_chunks.items():
    #     print(str(key) + ":", val)
    #     print(chunk_dict[str(key)] + '\n')
    most_similar_chunks = [chunk_dict[str(key)] for key, val in best_similarity_scores.items()]

    return most_similar_chunks

question = "How many startups has Syed Basit Ali Jafri found?"
rag = retrieval(question, embedding_dict, chunk_dict, 1)

# print(rag)
payload = json.dumps({
  "model": llm_name,
  "prompt": f"""You are a question answer model. You will be given a question inside this delimiter #### question ####. 
  Following the question, you will be provided with the relevant data. You being an execellent retrieval have to find the question's 
  answer from the data given inside this delimiter @@@@ data @@@@.  Return an informative answer to the question as your response.
  Make sure this answer is within 200 characters.
  The following is the information provided to you.
  ####{question}#### @@@@{rag}@@@@.
  """
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {llm_auth_token}"
}

response = requests.request("POST", llm_url, headers=headers, data=payload)
answer = response.json()['choices'][0]['text']

start_index = answer.find("Answer:") + len("Answer:")
end_index = answer.find("Explanation:")

extracted_text = answer[start_index:end_index].strip()

print('\n' +  extracted_text + '\n')


                 



                 

