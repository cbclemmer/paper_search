import re
import numpy as np
from typing import List
from flask import Flask, request, jsonify

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

app = Flask(__name__)
app.debug = False

print('Loading Glove embeddings')
glove_embeddings = load_glove_embeddings('glove.txt')

@app.route('/create', methods=['POST'])
def process():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text data found.'}), 400
    
    text = data['text']
    embedding = create_text_embedding(text, glove_embeddings)

    return jsonify(embedding), 200

if __name__ == '__main__':
    app.run(5000)