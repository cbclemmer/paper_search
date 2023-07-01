import os
from time import time
import json
import PyPDF2
from typing import List
import requests

def flatten(stacked_list):
    flattened_list = []
    for batch in stacked_list:
        for i in batch:
            flattened_list.append(i)
    return flattened_list

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

def create_glove_embedding(text: str):
    res = requests.post('localhost:5000/create')
    if res.status_code != 200:
        print('Error creating embedding:')
        print(res)
        return None
    return res.json()

def process_paper(file_path: str):
    # TODO: add more metadata from page sections and citations
    file_text = get_pdf_text(file_path)

    chunk_size = 1000
    chunks = chunk_text(file_text, chunk_size)
    data = []
    idx = 0
    for chunk in chunks:
        embedding = create_glove_embedding(chunk)
        idx += 1
        if embedding == None:
            continue
        data.append({
            "file": file_path,
            "char_index": (idx - 1) * chunk_size,
            "embedding": embedding
        })
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
    print(f'Vector embeddings file loaded in {time() - glove_start:.2f}s')
    
    papers = []
    paper_texts = []
    start_parse = time()
    for file in files:
      (file_data, file_text) = process_paper(file)
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