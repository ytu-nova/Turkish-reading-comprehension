import json
import re
from ngram import NGram
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def nb(data, labels, test, isSimilarity):
    nb_clf = MultinomialNB()
    nb_clf.fit(data, labels)
    predicted = zip(labels, nb_clf.predict_proba(test)[0])
    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    # print(predicted)
    return predicted[0][0] if isSimilarity else predicted[4][0]


def sgd(data, labels, test, isSimilarity):
    sgd_clf = SGDClassifier(loss='log')
    sgd_clf.fit(data, labels)
    predicted = zip(labels, sgd_clf.predict_proba(test)[0])
    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    # print(predicted)
    return predicted[0][0] if isSimilarity else predicted[4][0]


def rf(data, labels, test, isSimilarity):
    rf_clf = RandomForestClassifier()
    rf_clf.fit(data, labels)
    predicted = zip(labels, rf_clf.predict_proba(test)[0])
    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    # print(predicted)
    return predicted[0][0] if isSimilarity else predicted[4][0]


def lr(data, labels, test, isSimilarity):
    lr_clf = LogisticRegression()
    lr_clf.fit(data, labels)
    predicted = zip(labels, lr_clf.predict_proba(test)[0])
    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    # print(predicted)
    return predicted[0][0] if isSimilarity else predicted[4][0]


def kn(data, labels, test, isSimilarity):
    kn_clf = KNeighborsClassifier()
    kn_clf.fit(data, labels)
    predicted = zip(labels, kn_clf.predict_proba(test)[0])
    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    # print(predicted)
    return predicted[0][0] if isSimilarity else predicted[4][0]


def ngram_str(s, N=3):
    s = s.replace(" ", "_")
    ngram = NGram(N=N)
    return " ".join(ngram.split(s))


def sanitize(t):
    s = re.sub(r'[^a-zğüşıöç\s+]|\n|\t|\r', '', t, flags=re.IGNORECASE)
    return s  # ngram_str(s)


def process_data(q, doSaitize):
    labels = ['A', 'B', 'C', 'D', 'E']
    texts = [q['answers']['A'], q['answers']['B'], q['answers']['C'], q['answers']['D'], q['answers']['E']]
    #vectorizer = TfidfVectorizer(preprocessor=sanitize, ngram_range=(1, 2), analyzer="word")
    vectorizer = TfidfVectorizer(preprocessor=sanitize, ngram_range=(3, 4), analyzer="char")
    data = vectorizer.fit_transform(texts)
    # print(vectorizer.get_feature_names())
    test = vectorizer.transform([q['paragraph']])
    return data, labels, test


results = []
corrects = {'nb': 0, 'sgd': 0, 'rf': 0, 'lr': 0, 'kn': 0}
sim = {'nb': 0, 'sgd': 0, 'rf': 0, 'lr': 0, 'kn': 0}
unsim = {'nb': 0, 'sgd': 0, 'rf': 0, 'lr': 0, 'kn': 0}
simCount = 0
unsimCount = 0

with open('data-907.json') as json_file:
    data = json.load(json_file)
    for q in data:
        data, labels, test = process_data(q, True)

        if q['isSimilarity']:
            simCount += 1
        else:
            unsimCount += 1

        result = {'no': q['no'], 'isSimilarity': q['isSimilarity'],
                  'nb': False,
                  'sgd': False,
                  'rf': False,
                  'lr': False,
                  'kn': False}

        answer = nb(data, labels, test, q['isSimilarity'])
        if answer == q['correct']:
            result['nb'] = True
            corrects['nb'] += 1
            if q['isSimilarity']:
                sim['nb'] += 1
            else:
                unsim['nb'] += 1

        answer = sgd(data, labels, test, q['isSimilarity'])
        if answer == q['correct']:
            result['sgd'] = True
            corrects['sgd'] += 1
            if q['isSimilarity']:
                sim['sgd'] += 1
            else:
                unsim['sgd'] += 1

        answer = rf(data, labels, test, q['isSimilarity'])
        if answer == q['correct']:
            result['rf'] = True
            corrects['rf'] += 1
            if q['isSimilarity']:
                sim['rf'] += 1
            else:
                unsim['rf'] += 1

        answer = lr(data, labels, test, q['isSimilarity'])
        if answer == q['correct']:
            result['lr'] = True
            corrects['lr'] += 1
            if q['isSimilarity']:
                sim['lr'] += 1
            else:
                unsim['lr'] += 1

        answer = kn(data, labels, test, q['isSimilarity'])
        if answer == q['correct']:
            result['kn'] = True
            corrects['kn'] += 1
            if q['isSimilarity']:
                sim['kn'] += 1
            else:
                unsim['kn'] += 1

        results.append(result)

        if simCount > 0 and unsimCount > 0:
            count = int(q['no'])
            log = {'no': q['no']
                , 'nb': corrects['nb'] / count, 'nb-sim': sim['nb'] / simCount, 'nb-unsim': unsim['nb'] / unsimCount
                , 'sgd': corrects['sgd'] / count, 'sgd-sim': sim['sgd'] / simCount,
                   'sgd-unsim': unsim['sgd'] / unsimCount
                , 'rf': corrects['rf'] / count, 'rf-sim': sim['rf'] / simCount, 'rf-unsim': unsim['rf'] / unsimCount
                , 'lr': corrects['lr'] / count, 'lr-sim': sim['lr'] / simCount, 'lr-unsim': unsim['lr'] / unsimCount
                , 'kn': corrects['kn'] / count, 'kn-sim': sim['kn'] / simCount, 'kn-unsim': unsim['kn'] / unsimCount}

            log = {
                'nb': [corrects['nb'] / count, corrects['nb'], count],
                'nb-sim': [sim['nb'] / simCount, sim['nb'], simCount],
                'nb-unsim': [unsim['nb'] / unsimCount, unsim['nb'], unsimCount]
            }
            print(count, end='\r')

    with open('results.json', 'w') as outfile:
        json.dump(results, outfile)

    print('=================================================')
    print(log)

    # break

# {'no': '500', 'nb': 0.362, 'sgd': 0.278, 'rf': 0.29, 'lr': 0.306, 'kn': 0.26} CountVectorizer
# {'no': '500', 'nb': 0.362, 'sgd': 0.31, 'rf': 0.262, 'lr': 0.332, 'kn': 0.26} TfidfVectorizer
# {'no': '500', 'nb': 0.37, 'sgd': 0.332, 'rf': 0.256, 'lr': 0.336, 'kn': 0.26} TfidfVectorizer with sanitize
