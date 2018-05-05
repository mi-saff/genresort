import generate_data
import random
import numpy as np
from collections import defaultdict

def word_to_embedding():
    embedding_dict = defaultdict(list)
    with open("./glove/glove.6B.50d.txt") as f:
        for line in f.readlines():
            result = []
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            for i in range(0, 50):
                result.append(float(tokens[i+1]))
            embedding_dict[word] = result
    return embedding_dict

def generate_embeddings(paragraphs, embedding_dict, embed_size, threshold):
    embedded_paragraphs = []
    labels = []
    print len(paragraphs)
    for paragraph in paragraphs:
        new_paragraph = [0] * embed_size
        counter = 0
        for word in paragraph[0]:
            if word in embedding_dict:
                new_paragraph = [sum(x) for x in zip(new_paragraph, embedding_dict[word])]
                counter += 1
        if counter >= threshold:
            new_paragraph = [x / counter for x in new_paragraph]
            embedded_paragraphs.append(new_paragraph) 
            labels.append(paragraph[1])
    return embedded_paragraphs, labels

