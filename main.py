import json
import os
from time import time
from typing import List

import numpy as np
from numpy.linalg import norm

import util

def cosine_similarity(embedding1: List[float], embedding2: List[float]):
    # Compute the dot product of the embeddings
    dot_product = np.dot(embedding1, embedding2)
    
    # Compute the norms of the embeddings
    norm_embedding1 = norm(embedding1)
    norm_embedding2 = norm(embedding2)
    
    # Compute the cosine similarity
    similarity = dot_product / (norm_embedding1 * norm_embedding2)
    
    return similarity

def loop(paper_texts, paper_embeddings, glove_embeddings):
    query = input('Query: ')
    query_embedding = util.create_text_embedding(query, glove_embeddings)

    most_similar = (0, {})
    for embedding in paper_embeddings:
        similarity = cosine_similarity(embedding['embedding'], query_embedding)
        if similarity > most_similar[0]:
            most_similar = (similarity, embedding)

    back_track_amount = 0
    char_index = most_similar[1]['char_index']
    char_index -= back_track_amount
    if char_index < 0:
        char_index = 0
    
    text = ''
    for paper in paper_texts:
        if paper['file'] == most_similar[1]['file']:
            text = paper['text']
            break

    preview_amount = char_index + 1000
    if preview_amount > len(text):
        preview_amount = len(text) - 1

    file_name = most_similar[1]['file']

    print('\n\n')
    print(f'Similarity: {most_similar[0]}')
    print(f'File: {file_name}\n')
    print(f'{text[char_index:preview_amount]}')
    
def main():
    if not os.path.exists('embeddings.json') or not os.path.exists('texts.json'):
        print('Running first time import...')
        if util.import_papers() == 0:
           return
    
    print('Loading paper embeddings')
    start_embed_load = time()
    paper_embeddings = []
    with open('embeddings.json', 'r') as f:
        paper_embeddings = json.loads(f.read())
    
    paper_texts = []
    with open('texts.json', 'r') as f:
        paper_texts = json.loads(f.read())
    
    print(f'Reading Papers took {time() - start_embed_load:.2f}s')

    print('Loading Glove embeddings')
    glove_start = time()
    glove_embeddings = util.load_glove_embeddings('glove.txt')
    print(f'Vector embeddings file loaded in {time() - glove_start:.2f}s')
    while True:
        loop(paper_texts,  paper_embeddings, glove_embeddings)

main()