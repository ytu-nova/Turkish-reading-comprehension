import re
import json
import numpy as np
from math import sqrt
from itertools import chain
from bert_serving.client import BertClient

bc = BertClient()


def cosine_sim(u, v):
    return np.dot(u, v) / (sqrt(np.dot(u, u)) * sqrt(np.dot(v, v)))


def corpus2vectors(corpus):
    def vectorize(sentence, vocab):
        return [sentence.split().count(i) for i in vocab]

    vectorized_corpus = []
    vocab = sorted(set(chain(*[i.lower().split() for i in corpus])))
    for i in corpus:
        vectorized_corpus.append([vectorize(i, vocab)])
    return vectorized_corpus


def sanitize(t):
    return re.sub(r'[^a-zğüşıöç\s+]|\n|\t|\r', '', t.lower(), flags=re.IGNORECASE)


def similarity(a, b):
    # x = bc.encode([sanitize(a)])
    # y = bc.encode([sanitize(b)])
    # print(x[0], y[0])
    x, y = corpus2vectors([sanitize(a), sanitize(b)])
    #print(x[0], y[0])
    return cosine_sim(x[0], y[0])


corrects = 0
with open('data.json') as json_file:
    data = json.load(json_file)
    for q in data:
        sim = []
        sim.append({'answer': 'A', 'similarity': similarity(q['paragraph'], q['answers']['A'])})
        sim.append({'answer': 'B', 'similarity': similarity(q['paragraph'], q['answers']['B'])})
        sim.append({'answer': 'C', 'similarity': similarity(q['paragraph'], q['answers']['C'])})
        sim.append({'answer': 'D', 'similarity': similarity(q['paragraph'], q['answers']['D'])})
        sim.append({'answer': 'E', 'similarity': similarity(q['paragraph'], q['answers']['E'])})
        sim.sort(key=lambda x: x['similarity'], reverse=True)
        i = 0 if q['isSimilarity'] else 4
        # print(q['no'], sim[i]['answer'], q['correct'], q['isSimilarity'])
        if sim[i]['answer'] == q['correct']:
            corrects += 1
        print(q['no'], corrects, corrects / int(q['no']))
        # if q['no'] == '10':
        #    break
