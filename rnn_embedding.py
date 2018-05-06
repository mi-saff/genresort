import embedding
import generate_data
import numpy as np
from collections import defaultdict

def return_stuff(foldername):
    embedding_dict = embedding.word_to_embedding()
    new_dict = defaultdict(int)
    counter = 0

    for key in embedding_dict:
        new_dict[key] = counter
        counter += 1
    paragraphs = generate_data.main(foldername)
    word_pars = []
    label_pars = []
    for paragraph in paragraphs:
        new_list = []
        check = 0
        for word in paragraph[0]:
            new_list.append(new_dict[word])
        word_pars.append(new_list)
        label_pars.append(paragraph[1])
    #if foldername == "bigtest":
    #    return np.array(word_pars[:100]), np.array(label_pars[:100])
    return np.array(word_pars), np.array(label_pars)
