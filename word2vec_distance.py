import numpy as np
import pickle
import json
import re
from scipy import spatial
import os
import lzma
from gensim.models import Word2Vec, KeyedVectors
import io

VECTOR_SIZE = 100

def load_vectors(fname):
    print("loading vectors:", fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for i, line in enumerate(fin):
        print("%d / %d" % (i, n), end="\r")
        tokens = line.rstrip().split(' ')
        # data[tokens[0]] = tuple(tokens[1:])
        # data[tokens[0]] = " ".join(tokens[1:])
        try:
            data[tokens[0]] = tuple(map(float, tokens[1:]))
        except:
            print("error", tokens[0])

    return data

# from sklearn.metrics import jaccard_similarity_score

def lower_tr_utf8(s):
    return s.replace(u"I",u"ı").replace(u"İ", u"i").lower()


def removePunctuation(s):
    return re.sub(r"[^a-zA-ZçöğüşıA-ZÇÖĞÜŞİ ]+", "", s)


def distance(A, B):
    if sum(B) == 0 or sum(A) == 0:
        return 1
    return spatial.distance.cosine(A, B)


def sentenceVector(w2v, s):
    paragraphVector = np.zeros(VECTOR_SIZE)
    for i, word in enumerate(s.split(" ")):
        if word not in w2v:
            # kelime yoksa sondan harf silerek bakiyoruz
            for j in range(len(word)-1, 3, -1):
                if word[:j] in w2v:
                    paragraphVector += np.array(w2v[word[:j]])
                    break
        else:
            paragraphVector += np.array(w2v[word])
    if i == 0:
        return np.zeros(VECTOR_SIZE)
    return paragraphVector / i


def predict(w2v, paragraph, question, options, isSimilarity):
    sentences = paragraph.replace("?",".").replace("!", ".").split(".")

    d = np.zeros(shape=(len(sentences), len(options)))
    for i, sentence in enumerate(sentences):
        sentence = removePunctuation(lower_tr_utf8(sentence))
        sVector = sentenceVector(w2v, sentence)

        for j, option in enumerate(options):
            optionVector = sentenceVector(w2v, option)
            d[i, j] = distance(optionVector, sVector)

    if isSimilarity:

        min_distance = len(options) * [99999999999]

        for i in range(len(sentences)):
            for j in range(len(options)):
                min_distance[j] = min(min_distance[j], d[i, j])

        # olasilik uzaklik ile ters orantili
        distances = [x+0.001 for x in min_distance]  # 0'a bolme hatasi icin
        prob = [(1/x) / sum(1/y for y in distances) for x in distances]
        return prob

    else:
        # bir cumleye yakin mi?
        # her secenegin en yakin cumlesini bul
        a = len(options) * [ 0 ]  # en yakin cumlenin indexi
        for j in range(len(options)):
            for i in range(len(sentences)):
                if d[a[j], j] > d[i, j]:
                    a[j] = i

        distances = [ d[a[j], j] for j in range(len(options)) ]
        # olasilik uzaklik ile dogru orantili
        prob = [x / sum(distances) for x in distances]
        return prob


def word2vec_distance(vectorFilePath, data):

    SAVE_VECTOR_DICT = False

    # try:
    #     if SAVE_VECTOR_DICT:
    #         w2v = pickle.load(open("/tmp/word2vec-distance.pickle", "rb"))
    #     else:
    #         raise
    # except:
    #     w2v = dict()
    #     print("loading word vectors: %s" % vectorFilePath)
    #     lines = sum(1 for line in open(vectorFilePath, encoding="utf8"))
    #     for i, line in enumerate(open(vectorFilePath, encoding="utf8")):
    #         print("%d / %d" % (i, lines), end='\r')
    #         index = line.find(" ")
    #         word = line[:index]
    #         vectorStr = line[index+1:]
    #         vector = np.array([float(x) for x in vectorStr.split()])
    #         w2v[word] = vector
    #     if SAVE_VECTOR_DICT:
    #         pickle.dump(w2v, open("/tmp/word2vec-distance.pickle", "wb"))

    # w2v = load_vectors("_tmp/wiki.tr.vec")


    # w2v = load_vectors("/mnt/odev-4/fasttext-vectors/haber_ve_koseyazilari-cbow-default.txt.vec")
    w2v = load_vectors("/mnt/odev-4/fasttext-vectors/haber_ve_koseyazilari-skipgram-default.txt.vec")

    # w2v = load_vectors("/mnt/odev-4/fasttext-vectors/tamami-cbow-default.txt.vec")
    # w2v = load_vectors("/mnt/odev-4/fasttext-vectors/skipgram-cbow-default.txt.vec")

    # w2v = KeyedVectors.load_word2vec_format('./_tmp/trmodel', binary=True)


    simQuestionCount = 0
    correctSimQuestion = 0
    correctAnswers = set()

    for qNo, q in enumerate(data):
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

        prob = predict(w2v, paragraph, question, options, isSimilarity)

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
        print("olumlu basari: %.2f (%d / %d)" % (float(correctSimQuestion) / simQuestionCount, correctSimQuestion, simQuestionCount))
    if notSimQuestionCount > 0:
        print("olumsuz basari: %.2f (%d / %d)" % (float(correctNotSimQuestion) / notSimQuestionCount, correctNotSimQuestion, notSimQuestionCount))
    print("basari: %.2f" % (float(len(correctAnswers)) / len(data)))

    return correctAnswers


if __name__ == "__main__":

    if not os.path.exists("./_tmp"):
        os.mkdir("./_tmp")

    vectorFile = "haber_ve_koseyazilari_sg_size100_iter10_window5.txt"
    vectorFilePath = os.path.join("./_tmp/", vectorFile)

    if not os.path.exists(vectorFilePath):
        vectorFileUrl = "https://drive.google.com/uc?export=download&id=19HXSSlwdB6rBYw_RgvdZTk4DIP7znHSk"
        vectorFilePathXz = "%s.xz" % vectorFilePath
        if not os.path.exists(vectorFilePathXz):
            print("downloading %s -> %s" % (vectorFileUrl, vectorFilePathXz))
            import urllib.request
            urllib.request.urlretrieve(vectorFileUrl, vectorFilePathXz)

        print("extracting %s" % vectorFilePathXz)
        open(vectorFilePath, "w", encoding="utf-8").write(lzma.open(vectorFilePathXz).read().decode("utf-8"))

    data = json.load(open('data.json', encoding="utf-8"))
    # data = json.load(open('yenisorular/farkliYeniSorular.json', encoding="utf-8"))

    word2vec_distance(vectorFilePath, data)
