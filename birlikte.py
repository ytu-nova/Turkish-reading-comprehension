import json
import numpy as np
from ngram import NGram
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from char_ngram import predict_char_ngram, ngram_compare

def nb_process_data(q, doSaitize):
    labels = ['A', 'B', 'C', 'D', 'E']
    texts = [q['answers']['A'], q['answers']['B'], q['answers']['C'], q['answers']['D'], q['answers']['E']]
    vectorizer = TfidfVectorizer(preprocessor=sanitize, ngram_range=(3, 4), analyzer="char")
    data = vectorizer.fit_transform(texts)
    test = q['paragraph'].replace('?', '.').replace('!', '.').split('.')
    test = vectorizer.transform(test)
    return data, labels, test


def nb_matrix_prob(q, isSimilarity):
    data, labels, test = nb_process_data(q, True)
    nb_clf = MultinomialNB()
    nb_clf.fit(data, labels)
    return nb_clf.predict_proba(test)


def ngram_v2_matrix_prob(N, paragraph, options, isSimilarity):
    sentences = paragraph.replace("?",".").replace("!", ".").split(".")

    d = np.zeros(shape=(len(sentences), len(options)))
    for i, sentence in enumerate(sentences):
        sentence = sanitize(sentence)
        for j, secenek in enumerate(options):
            secenek = sanitize(secenek)
            d[i, j] = ngram_compare(sentence, secenek, N=N)
    return d


def sentenceBased_matrix2vector(d, isSimilarity):
    similarities = [max(d[:, j]) for j in range(d.shape[1])]
    s = sum(similarities)
    if isSimilarity:
        prob = [x/s for x in similarities]
    else:
        similarities = [x+0.001 for x in similarities]  # 0'a bolme hatasi icin
        prob = [(1/x)/s for x in similarities]
    return prob


def sanitize(t):
    return re.sub(r'[^a-zğüşıöç\s+]|\n|\t|\r', '', t.lower(), flags=re.IGNORECASE)


def run(data):
    simQuestionCount = 0
    correctSimQuestion = 0
    correctAnswers = set()

    for qNo, q in enumerate(data):
        print(qNo+1, end='\r')

        paragraph = q["paragraph"]
        question = None
        isSimilarity = q["isSimilarity"]

        correctIndex = "ABCDE".index(q["correct"])
        options = [ q["answers"][c] for c in "ABCDE" ]

        if isSimilarity:
            simQuestionCount += 1
        else:
            pass # len(data) - simQuestionCount


        probA = predict_char_ngram(3, paragraph, question, options, isSimilarity)
        probA = np.array(probA)
        # probB = predict_char_ngram_v2(4, paragraph, question, options, isSimilarity)

        m1 = ngram_v2_matrix_prob(4, paragraph, options, isSimilarity)
        probB = sentenceBased_matrix2vector(m1, isSimilarity)
        probB = np.array(probB)

        m2 = nb_matrix_prob(q, isSimilarity)
        probC = sentenceBased_matrix2vector(m2, isSimilarity)
        probC = np.array(probC)

        if isSimilarity:
            prob = probA
        else:
            prob = probB + probC

        # prob = probA + probB + probC

        predicted = np.argmax(prob)

        if predicted == correctIndex:
            correctAnswers.add(qNo)
            if isSimilarity:
                correctSimQuestion += 1
            else:
                pass # len(correctAnswers) - correctSimQuestion

    notSimQuestionCount = len(data) - simQuestionCount
    correctNotSimQuestion = len(correctAnswers) - correctSimQuestion

    if simQuestionCount > 0:
        print("olumlu basari: %.2f (%d / %d)" % (100 * float(correctSimQuestion) / simQuestionCount, correctSimQuestion, simQuestionCount))
    if notSimQuestionCount > 0:
        print("olumsuz basari: %.2f (%d / %d)" % (100 * float(correctNotSimQuestion) / notSimQuestionCount, correctNotSimQuestion, notSimQuestionCount))
    print("basari: %.2f" % (100 * float(len(correctAnswers)) / len(data)))

    return correctAnswers


if __name__ == "__main__":
    for filePath in ('data.json', 'yenisorular/farkliYeniSorular.json'):
        print(filePath)
        data = json.load(open(filePath, encoding="utf-8"))
        run(data)
