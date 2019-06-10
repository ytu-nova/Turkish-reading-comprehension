import json
import numpy as np
from ngram import NGram
import re
import io

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from scipy import spatial

from char_ngram import predict_char_ngram, ngram_compare


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
        data[tokens[0]] = tuple(map(float, tokens[1:]))

    return data


def distance(A, B):
    if sum(B) == 0 or sum(A) == 0:
        return 1
    return spatial.distance.cosine(A, B)

def lower_tr_utf8(s):
    return s.replace(u"I",u"ı").replace(u"İ", u"i").lower()

def removePunctuation(s):
    return re.sub(r"[^a-zA-ZçöğüşıA-ZÇÖĞÜŞİ ]+", "", s)


def sentenceVector(w2v, s, vectorSize):
    vec = np.zeros(vectorSize)
    for i, word in enumerate(s.split(" ")):
        if word not in w2v:
            # kelime yoksa sondan harf silerek bakiyoruz
            for j in range(len(word)-1, 3, -1):
                if word[:j] in w2v:
                    vec += np.array(w2v[word[:j]])
                    break
        else:
            vec += np.array(w2v[word])
    if i == 0:
        return np.zeros(vectorSize)
    return vec / i


def wordVectors_matrix_prob(w2v, paragraph, options, isSimilarity, vectorSize):
    sentences = paragraph.replace("?",".").replace("!", ".").split(".")

    d = np.zeros(shape=(len(sentences), len(options)))
    for i, sentence in enumerate(sentences):
        sentence = sanitize(lower_tr_utf8(sentence))
        sVector = sentenceVector(w2v, sentence, vectorSize)

        for j, option in enumerate(options):
            option = sanitize(option)
            optionVector = sentenceVector(w2v, option, vectorSize)
            d[i, j] = 1.0 / distance(optionVector, sVector)

    return d

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

def most_frequent(List):
    return max(set(List), key = List.count)


def run(data, wordVectors, size_wv, wordVectors_2=None, size_wv_2=None):
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
        probA2 = sentenceBased_matrix2vector(m1, isSimilarity)
        probA2 = np.array(probA2)


        m2 = nb_matrix_prob(q, isSimilarity)
        probC = sentenceBased_matrix2vector(m2, isSimilarity)
        probC = np.array(probC)


        # UYARI: burada noktalama kaldirilirsa paragraf-cumle benzerligi olarak calisir. (cumleleri boldukten sonra kendisi noktalamayi kaldiriyor)
        # paragraph = removePunctuation(paragraph)

        if wordVectors:
            m3 = wordVectors_matrix_prob(wordVectors, paragraph, options, isSimilarity, size_wv)
            probD = sentenceBased_matrix2vector(m3, isSimilarity)
            probD = np.array(probD)

        if wordVectors_2 != None:
            m4 = wordVectors_matrix_prob(wordVectors_2, paragraph, options, isSimilarity, size_wv_2)
            probE = sentenceBased_matrix2vector(m4, isSimilarity)


        # prob = probD

        # prob = probA + probC

        # en iyi sonuc
        prob = probA + probC + probD

        # prob = probA + probD
        # prob = probC + probD

        # prob = probA + probC + probD + probE
        # prob = probA + probA2 + probC + probD + probE
        predicted = np.argmax(prob)

        # A = [probA, probA2, probC, probD, probE]
        # A = [probA, probC, probD]
        # predicted = most_frequent([np.argmax(a) for a in A])

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

    return correctAnswers, np.array((correctSimQuestion, simQuestionCount, correctNotSimQuestion, notSimQuestionCount))


if __name__ == "__main__":

    size_wv = 100
    wordVectors = load_vectors("./_tmp/haber_ve_koseyazilari_sg_size100_iter10_window5.txt")  # %37.93  - p-c:??
    # wordVectors = None
    # wordVectors = load_vectors("/mnt/odev-4/vectors/haber_ve_koseyazilari_cbow_size100_iter10_window5.txt")  # 34.40
    # wordVectors = load_vectors("/mnt/odev-4/vectors/tamami_sg_window5_size100.txt")  # 36.38  - p-c:??
    # wordVectors = load_vectors("/mnt/odev-4/vectors/tamami_cbow_window5_size100.txt")  # 31.42
    # wordVectors = load_vectors("/mnt/odev-4/fasttext-vectors/haber_ve_koseyazilari-skipgram-default.txt.vec")  # ??? -p-c:?
    # wordVectors = load_vectors("/mnt/odev-4/fasttext-vectors/haber_ve_koseyazilari-cbow-default.txt.vec")  # 31.53
    # wordVectors = load_vectors("/mnt/odev-4/fasttext-vectors/tamami-cbow-default.txt.vec")  # 30.54
    # wordVectors = load_vectors("/mnt/odev-4/fasttext-vectors/skipgram-cbow-default.txt.vec")  # 34.73   - p-c: ???


    # size_wv = 300
    # wordVectors = load_vectors("./_tmp/wiki.tr.vec")  # ???  - p-c:

    wordVectors_2 = None
    size_wv_2 = None
    # size_wv_2 = 100
    # wordVectors_2 = load_vectors("./_tmp/haber_ve_koseyazilari_sg_size100_iter10_window5.txt")  # %37.93  - p-c:??


    results = np.zeros((4,),dtype=int)

    for filePath in ('data.json', 'yenisorular/farkliYeniSorular.json'):
        print(filePath)
        data = json.load(open(filePath, encoding="utf-8"))
        _, r = run(data, wordVectors, size_wv, wordVectors_2, size_wv_2)
        results += r
        print("")


    print("==============")
    correctSimQuestion, simQuestionCount, correctNotSimQuestion, notSimQuestionCount = (x.item() for x in results)

    print("olumlu basari: %.2f" % (100.0 * correctSimQuestion / simQuestionCount))
    print("olumsuz basari: %.2f" % (100.0 * correctNotSimQuestion / notSimQuestionCount))
    correctAnswers = correctSimQuestion + correctNotSimQuestion
    questionCount = simQuestionCount + notSimQuestionCount
    print("basari: %.2f" % (100.0 * correctAnswers / questionCount))
