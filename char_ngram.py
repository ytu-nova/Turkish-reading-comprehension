# -*- coding: utf-8 -*-

"""
terminalde once su komutun calistirilmasi gerekli
  export PYTHONIOENCODING=utf-8

veya preprocess oncesinde
  iconv -f windows-1254 -t utf8 Sorular.txt > Sorular-utf8.txt
"""

import json
import numpy as np
from ngram import NGram
import re
from pprint import pprint

def sanitize(t):
    return re.sub(r'[^a-zğüşıöç\s+]|\n|\t|\r', '', t.lower(), flags=re.IGNORECASE)


# ngram v2: https://stackoverflow.com/questions/14059444/python-ngram-calculation

def lower_tr_utf8(s):
    return s  # FIXME: bu fonksiyona gerek yok galiba
    # return s.replace(u"I",u"ı").replace(u"İ", u"i").lower()

def removePunctuation(s):
    return sanitize(s)
    # return re.sub(r"[^a-zA-ZçöğüşıA-ZÇÖĞÜŞİ ]+", "_", s)

def ngram_compare(a, b, N):
    ngram = NGram(N=N)
    A = set(ngram.ngrams(a))
    B = set(ngram.ngrams(b))
    return len(A & B) / len(A | B)


def predict_char_ngram(N, paragraph, question, options, isSimilarity):
    paragraph = removePunctuation(paragraph)
    options = [removePunctuation(x) for x in options]
    # similarities = [NGram.compare(paragraph, s, N=N) for s in options]
    # similarities = [ngram_compare(paragraph, s, N=N) for s in options]
    similarities = [NGram.compare(paragraph, s, N=N, pad_len=0) for s in options]

    if isSimilarity:
        prob = [x / sum(similarities) for x in similarities]
    else:
        similarities = [x+0.00001 for x in similarities]  # 0'a bolme hatasi icin
        prob = [(1/x) / sum(1/y for y in similarities) for x in similarities]
    return prob

def predict_char_ngram_v2(N, paragraph, question, options, isSimilarity):
    sentences = paragraph.replace("?",".").replace("!", ".").split(".")

    d = np.zeros(shape=(len(sentences), len(options)))
    for i, sentence in enumerate(sentences):
        sentence = removePunctuation(sentence)
        for j, secenek in enumerate(options):
            secenek = removePunctuation(secenek)
            # d[i, j] = NGram.compare(sentence, secenek, N=N)
            d[i, j] = ngram_compare(sentence, secenek, N=N)
            # d[i, j] = NGram.compare(sentence, secenek, N=N, pad_len=0)

    if isSimilarity:
        max_sim = len(options) * [0]
        for i in range(len(sentences)):
            for j in range(len(options)):
                max_sim[j] = max(max_sim[j], d[i, j])

        prob = [x / sum(max_sim) for x in max_sim]
        return prob

    else:
        # her secenege en cok benzeyen cumlenin indexini bulur
        ind_most_sim = len(options) * [ 0 ]
        for j in range(len(options)):
            for i in range(len(sentences)):
                if d[ind_most_sim[j], j] < d[i, j]:
                    ind_most_sim[j] = i

        # optionsin en yakin cumlelere benzerlikleri
        similarities = len(options) * [ 0 ]
        for j in range(len(options)):
            similarities[j] = d[ind_most_sim[j], j]

        similarities = [x+0.001 for x in similarities]  # 0'a bolme hatasi icin
        prob = [(1/x) / sum(1/y for y in similarities) for x in similarities]
        return prob

def char_ngram(data, N=4, sentenceComparison=False):

    simQuestionCount = 0
    correctSimQuestion = 0
    correctAnswers = set()

    for qNo, q in enumerate(data):
        # print(qNo+1, end='\r')
        paragraph = q["paragraph"]
        question = q["text"]
        isSimilarity = q["isSimilarity"]

        correctIndex = "ABCDE".index(q["correct"])
        options = [ q["answers"][c] for c in "ABCDE" ]

        paragraph = lower_tr_utf8(paragraph)
        question = lower_tr_utf8(question)
        options = [ lower_tr_utf8(x) for x in options ]

        if isSimilarity:
            simQuestionCount += 1
        else:
            pass # len(data) - simQuestionCount

        if sentenceComparison:
            prob = predict_char_ngram_v2(N, paragraph, question, options, isSimilarity)
        else:
            prob = predict_char_ngram(N, paragraph, question, options, isSimilarity)

        predicted = np.argmax(prob)

        if predicted == correctIndex:
            correctAnswers.add(qNo)
            if isSimilarity:
                correctSimQuestion += 1
            else:
                pass # len(correctAnswers) - correctSimQuestion

        # print("basari: %.2f" % (float(len(correctAnswers)) / (qNo+1)))

    notSimQuestionCount = len(data) - simQuestionCount
    correctNotSimQuestion = len(correctAnswers) - correctSimQuestion

    if simQuestionCount > 0:
        print("olumlu basari: %.2f (%d / %d)" % (100 * float(correctSimQuestion) / simQuestionCount, correctSimQuestion, simQuestionCount))
    if notSimQuestionCount > 0:
        print("olumsuz basari: %.2f (%d / %d)" % (100 * float(correctNotSimQuestion) / notSimQuestionCount, correctNotSimQuestion, notSimQuestionCount))
    print("basari: %.2f" % (100 * float(len(correctAnswers)) / len(data)))

    return correctAnswers, np.array((correctSimQuestion, simQuestionCount, correctNotSimQuestion, notSimQuestionCount))


if __name__ == "__main__":

    # correctAnswers, correctSimQuestion, simQuestionCount, correctNotSimQuestion, notSimQuestionCount

    parameters = ("paragraph-N3", "paragraph-N4", "sentence-N3", "sentence-N4")

    results = dict()

    results = { x:np.zeros((4,),dtype=int) for x in parameters }

    for filePath in ('data.json', 'yenisorular/farkliYeniSorular.json'):
        print("\n==============")
        print(filePath)

        data = json.load(open(filePath, encoding="utf-8"))

        print("\nchar_ngram N=3, sentenceComparison=False")
        _, r = char_ngram(data, N=3, sentenceComparison=False)
        results["paragraph-N3"] += r

        print("\nchar_ngram N=4, sentenceComparison=False")
        _, r = char_ngram(data, N=4, sentenceComparison=False)
        results["paragraph-N4"] += r

        print("\nchar_ngram N=3, sentenceComparison=True")
        _, r = char_ngram(data, N=3, sentenceComparison=True)
        results["sentence-N3"] += r

        print("\nchar_ngram N=4, sentenceComparison=True")
        _, r = char_ngram(data, N=4, sentenceComparison=True)
        results["sentence-N4"] += r

    print("=================================================")

    for p in parameters:
        print(p)
        correctSimQuestion, simQuestionCount, correctNotSimQuestion, notSimQuestionCount = (x.item() for x in results[p])

        try:
            print("olumlu basari: %.2f" % (100.0 * correctSimQuestion / simQuestionCount))
            print("olumsuz basari: %.2f" % (100.0 * correctNotSimQuestion / notSimQuestionCount))
            correctAnswers = correctSimQuestion + correctNotSimQuestion
            questionCount = simQuestionCount + notSimQuestionCount
            print("basari: %.2f" % (100.0 * correctAnswers / questionCount))
        except ZeroDivisionError:
            print("ZeroDivisionError")

        print("")

    print("simQuestionCount:", simQuestionCount)
    print("notSimQuestionCount:", notSimQuestionCount)
    print("total:", simQuestionCount+notSimQuestionCount)
