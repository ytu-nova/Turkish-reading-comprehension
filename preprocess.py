import json
import re

questionsData = './Sorular.txt'
answersData = './Cevaplar.txt'

questions = []
data = open(questionsData, "r").read()

pattern = re.compile(
    r'(([0-9]+)\))([^\n]+)[\n]+([^\n]+)[\n]+([A]\)([^\n]+))[\n]+([B]\)([^\n]+))[\n]+([C]\)([^\n]+))[\n]+([D]\)([^\n]+))[\n]+([E]\)([^\n]+))',
    flags=re.IGNORECASE)

for match in re.finditer(pattern, data):
    found = match.groups()
    question = {
        'no': found[1],
        'paragraph': found[2],
        'text': found[3],
        'answers': {'A': found[5], 'B': found[7], 'C': found[9], 'D': found[11], 'E': found[13]},
        'isSimilarity': None,
        'correct': '',
    }
    nonsimilarityDeterminers = 'mamıştır|memiştir|memistir|lamaz|emez|değildir'
    similarityDeterminers = 'abilir|ebilir|maktadır|mektedir|ıdır|idir|udur|edir|mıştır|miştir|muştur|sonuçtur|gerekir'
    if re.search(nonsimilarityDeterminers, question['text']):
        question['isSimilarity'] = False
    elif re.search(similarityDeterminers, question['text']):
        question['isSimilarity'] = True
    else:
        # for correction
        print(question['no'], question['text'])
        break

    questions.append(question)

data = open(answersData, "r").read()
pattern = re.compile(r'(([0-9]+)\)[ ]?([ABCDE]+))', flags=re.IGNORECASE)

for match in re.finditer(pattern, data):
    found = match.groups()
    answer = {
        'no': found[1],
        'correct': found[2],
    }
    if questions[int(answer['no']) - 1]:
        questions[int(answer['no']) - 1]['correct'] = answer['correct']

with open('data.json', 'w') as outfile:
    json.dump(questions, outfile)
print("\n")
print("Total questions:", len(questions))
print("The preprocess is completed successfully")
