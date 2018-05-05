import os
import string
import random
from nltk.corpus import stopwords

def remove_stop_words(paragraph):
    s = set(stopwords.words('english'))
    return filter(lambda w: not w in s, paragraph)

def clean_paragraphs(paragraphs, fnum):
    clean = lambda x: x.decode('utf8').encode('ascii', errors='ignore').translate(string.maketrans("",""), string.punctuation).strip("\"").lower().split(" ")[0:-1]
    new_paragraphs = [(clean(curr_paragraph), fnum) for curr_paragraph in paragraphs if len(clean(curr_paragraph)) > 20]
    return new_paragraphs

def get_paragraphs(filename, fnum):
    with open(filename, "r") as fp:
        lines = fp.readlines()
    paragraphs = []
    curr_line = ""
    for line in lines:
        if line != "\n":
            curr_line += line[0:-1] + " "
        else:
            paragraphs.append(curr_line)
            curr_line = ""
    label = [0] * 2
    label[fnum - 1] = 1
    new_paragraphs = clean_paragraphs(paragraphs, label)
    return new_paragraphs

def shuffle(x, y, batch_size):
    if len(x) == len(y):
        c = list(zip(x, y))
        random.shuffle(c)
        a, b = zip(*c)
    return a[:batch_size], b[:batch_size]

def main(folder_name):
    fnum = 0
    tot_paragraphs = []
    main_folder = "./" + folder_name + "/"
    for folder in os.listdir(main_folder):
        if folder != ".DS_Store":
            fnum += 1
            for filename in os.listdir(main_folder+folder):
                fullname = main_folder+folder+"/"+filename
                tot_paragraphs += get_paragraphs(fullname, fnum)
    #tot_paragraphs += get_paragraphs("./data/crime/baskervilles.txt", 2)
    return tot_paragraphs
