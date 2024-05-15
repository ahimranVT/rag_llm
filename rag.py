import os
import pickle
import requests
import json
from dotenv import load_dotenv
from helpers import embed, save_chunks, read_chunks, segment_chunk, retrieval

def execute_rag(question):
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


    rag = retrieval(question, embedding_dict, chunk_dict, 1)

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

    return extracted_text


                 



                 

