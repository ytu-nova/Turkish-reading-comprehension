import re
import os

import json
from difflib import SequenceMatcher

# data = ""
# lines = open("01.txt").readlines()
# for i, line in enumerate(lines):
#     data += line
#     if i %  8 in (0, 1):
#         data += "\n"

#     if i == 20:
#         break



def parse(data):
    questions = []
    pattern = re.compile(
        # r'(([0-9]+)\))([^\n]+)[\n]+([^\n]+)[\n]+([A]\)([^\n]+))[\n]+([B]\)([^\n]+))[\n]+([C]\)([^\n]+))[\n]+([D]\)([^\n]+))[\n]+([E]\)([^\n]+))',

        r'(([0-9]+)\.)([^\n]+)[\n]+([^\n]+)[\n]+([A]\)([^\n]+))[\n]+([B]\)([^\n]+))[\n]+([C]\)([^\n]+))[\n]+([D]\)([^\n]+))[\n]+([E]\)([^\n]+))',
        flags=re.IGNORECASE)

    for match in re.finditer(pattern, data):
        found = match.groups()
        question = {
            'no': found[1].strip(),
            'paragraph': found[2].strip(),
            'text': found[3].strip(),
            'answers': {'A': found[5].strip(), 'B': found[7].strip(), 'C': found[9].strip(), 'D': found[11].strip(), 'E': found[13].strip()},
            'isSimilarity': None,
            'correct': '',
        }

        nonsimilarityDeterminers = 'mamıştır|memiştir|memistir|lamaz|emez|değildir|yoktur|olmaz|bağdaşmaz|çelişir|beklenmez|yanlıştır'
        similarityDeterminers = 'abilir|ebilir|maktadır|mektedir|ıdır|idir|udur|edir|mıştır|miştir|muştur|sonuçtur|gerekir|beklenir|içerir|kaynaklanır|nitelendirilir|koyar'
        if re.search(nonsimilarityDeterminers, question['text']):
            question['isSimilarity'] = False
        elif re.search(similarityDeterminers, question['text']):
            question['isSimilarity'] = True
        else:
            # for correction
            print(question['no'], question['text'])
            exit(1)

        questions.append(question)


    i = data.find("CEVAP ANAHTARI")
    if i < 0:
        print("cevap anahtari yok")
        exit(1)

    cevapAnahtari = data[i + data[i:].find("\n"):].strip()
    print(cevapAnahtari)
    cevapAnahtari = [x.split("-") for x in cevapAnahtari.split(" ") if x != ""]
    cevapAnahtari = {int(x[0]):x[1] for x in cevapAnahtari}

    for q in questions:
        cevap = cevapAnahtari[int(q["no"])]
        assert(cevap in "ABCDE")
        q["correct"] = cevap

    return questions

questions = []

path = "./ana-dusunce/"
for fileNo in range(1, 25):
    if fileNo < 10:
        fileName = "0%d.txt" % fileNo
    else:
        fileName = "%d.txt" % fileNo
    fileName = os.path.join(path, fileName)
    data = open(fileName).read()
    print(fileName)
    questions.extend(parse(data))


path = "./yardimci-dusunce/"
for fileNo in range(1, 8):
    fileName = os.path.join(path, "0%d.txt" % fileNo)
    data = open(fileName).read()
    print(fileName)
    questions.extend(parse(data))


Q = []
LIMIT = 80
for i in range(len(questions)):
    print("%d / %d" % (i, len(questions)), end='\r')
    A = questions[i]["paragraph"][:LIMIT]

    found = False
    for j in range(i):
        B = questions[j]["paragraph"][:LIMIT]
        ratio = SequenceMatcher(None, A, B).ratio()
        if ratio > 0.8:
            found = True
            print(A)
            print(B)
            print(" ")
            break

    if not found:
        Q.append(questions[i])


print(len(questions))
print(len(Q))
questions = Q
# exit(1)
# print(len(questions))

paragraphs = []

olumlu = 0
olumsuz = 0

for q in questions:
    paragraphs.append(q["paragraph"][:LIMIT])
    if q["isSimilarity"]:
        olumlu += 1
    else:
        olumsuz += 1

print("olumlu:", olumlu)
print("olumsuz:", olumsuz)

#     print(question["no"])
#     print(question["paragraph"])
#     print(question["answers"])
#     print(question["isSimilarity"])
#     print()


ayniSorular = []
data = json.load(open('/taner/mnt/usbhdd2p1/docker/wd/python3/home/taner/ha-yayin/data.json', encoding="utf-8"))
for qNoA, q in enumerate(data):
    print(qNoA, end='\r')
    A = q["paragraph"][:LIMIT]
    for qNoB, B in enumerate(paragraphs):
        ratio = SequenceMatcher(None, A, B).ratio()
        if ratio > 0.8:
            # print(A)
            # print(B)
            # print()
            ayniSorular.append((qNoA, qNoB))
            break

print("A:", qNoA)
print("B:", len(questions))
print("ortak:", len(ayniSorular))

print("birlesim:", len(questions) + qNoA - len(ayniSorular))


with open("tumYeniSorular.json", "w") as outfile:
    json.dump(questions, outfile)


ayniSorular = set(x[1] for x in ayniSorular)
questions = [q for i, q in enumerate(questions) if i not in ayniSorular]
print(len(questions))

with open("farkliYeniSorular.json", "w") as outfile:
    json.dump(questions, outfile)
