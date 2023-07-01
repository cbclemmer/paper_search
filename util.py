import os
import re
from time import time
import json
import PyPDF2
import numpy as np
from typing import List

def flatten(stacked_list):
    flattened_list = []
    for batch in stacked_list:
        for i in batch:
            flattened_list.append(i)
    return flattened_list

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

def get_pdf_text(file_path: str) -> str:
    pdf_reader = PyPDF2.PdfReader(f'papers/{file_path}')
    file_text = ''
    for page_num in range(0, len(pdf_reader.pages)):
        file_text += pdf_reader.pages[page_num].extract_text()
    return file_text

def process_paper(file_path: str, glove_embeddings: dict):
    # TODO: add more metadata from page sections and citations
    file_text = get_pdf_text(file_path)

    chunk_size = 1000
    chunks = chunk_text(file_text, chunk_size)
    data = []
    idx = 0
    for chunk in chunks:
        data.append({
            "file": file_path,
            "char_index": idx * chunk_size,
            "embedding": create_text_embedding(chunk, glove_embeddings)
        })
        idx += 1
    return (data, file_text)

def import_papers():
    if not os.path.exists('glove.txt'):
        print('Error: There is no glove vector file present. Add the glove.txt file to the project directory')
        return 1

    if not os.path.exists('papers'):
        print('Error: There is no papers directory, add papers to the "papers" folder to import them')
        return 1
    
    files = os.listdir('papers')
    if len(files) == 0:
        print('Error: No files found in papers folder')
        return 1
    
    print('Loading Glove embeddings')
    glove_start = time()
    glove_embeddings = load_glove_embeddings('glove.txt')
    print(f'Vector embeddings file loaded in {time() - glove_start:.2f}s')
    
    papers = []
    paper_texts = []
    start_parse = time()
    for file in files:
      (file_data, file_text) = process_paper(file, glove_embeddings)
      papers.append(file_data)
      paper_texts.append({
          "file": file,
          "text": file_text
      })
    print(f'Parsed papers in {time() - start_parse:.2f}s')

    with open('embeddings.json', 'w') as f:
        f.write(json.dumps(flatten(papers)))
    with open('texts.json', 'w') as f:
        f.write(json.dumps(paper_texts))
    return 1