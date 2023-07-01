import os
import re
from time import time
import json
import PyPDF2
import numpy as np
from typing import List
from joblib import Parallel, delayed

def batchify(list, num_batches):
    batch_len = len(list) // num_batches
    batches = []
    current_batch = []
    for i in list:
        current_batch.append(i)
        if len(current_batch) >= batch_len:
            batches.append(current_batch)
            current_batch = []
    return batches

def flatten(stacked_list):
    flattened_list = []
    for batch in stacked_list:
        for i in batch:
            flattened_list.append(i)
    return flattened_list

# function must be in the form of fn(thread_idx, batch, all_items)
def parallelize(fn, num_threads, list):
    batches = batchify(list, num_threads)
    ret_batches = Parallel(n_jobs=num_threads)(
        delayed(fn)(idx, batch, list) for (batch, idx) in zip(batches, range(0, num_threads))
    )
    return flatten(ret_batches)

def load_glove_embeddings(embeddings_file):
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def create_text_embedding(string: str, embeddings: dict) -> List[int]:
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
    embedding = []
    found_words = 0
    for wd in cleaned_string.split(' '):
        wd = wd.lower()
        if wd in embeddings:
            found_words += 1
            wd_embedding = embeddings[wd].tolist()
            if len(embedding) > 0:
                for i in range(0, len(wd_embedding)):
                    embedding[i] += wd_embedding[i]
            else:
                embedding = wd_embedding
    if len(embedding) == 0:
        return []
    for i in range(0, len(embedding)):
        embedding[i] /= found_words
    return embedding

def chunk_text(text: str, chunk_size: int) -> List[str]:
    chunks = []
    while len(text) > 0:
        chunks.append(text[:chunk_size])
        text = text[chunk_size:]
    return chunks

def process_paper(file_path: str, glove_embeddings: dict):
    pdf_reader = PyPDF2.PdfReader(f'papers/{file_path}')
    file_text = ''
    # TODO: add more metadata from page sections and citations
    for page_num in range(0, len(pdf_reader.pages)):
        file_text += pdf_reader.pages[page_num].extract_text()

    chunk_size = 1000
    chunks = chunk_text(file_text, chunk_size)
    data = []
    idx = 0
    for chunk in chunks:
        data.append({
            "file": file_path,
            "text": chunk,
            "char_index": idx * chunk_size,
            "embedding": create_text_embedding(chunk, glove_embeddings)
        })
        idx += 1
    return data

def main():
    if not os.path.exists('glove.txt'):
        print('Error: There is no glove vector file present. Add the glove.txt file to the project directory')
        return

    if not os.path.exists('papers'):
        print('Error: There is no papers directory, add papers to the "papers" folder to import them')
        return
    
    print('Loading Glove embeddins')
    glove_start = time()
    glove_embeddings = load_glove_embeddings('glove.txt')
    print(f'Vector embeddings file loaded in {time() - glove_start:.2f}s')

    files = os.listdir('papers')
    papers = []
    for file in files:
      papers.append(process_paper(file, glove_embeddings))

    with open('embeddings.json', 'w') as f:
        f.write(json.dumps(papers))

main()