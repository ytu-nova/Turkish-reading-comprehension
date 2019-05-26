## ha-yayin

##### PREPROCESSING
Sorular.txt ve Cevaplar.txt dosyalarini isleyip sorulari iki sinifa ayirip JSON dosyasi olarak cikti veriyor.

```
python3 preprocess.py
```

output: data.json

##### SCIKIT LEARN
Vektore cevirdigi metinlerin farkli siniflandiricilar ile  benzerlik olasiligini hesapliyor.

```
python3 preprocess.py
```


##### SCIKIT SENTENCES

Paragraflari cumlelere ayirip bir onceki ile ayni islemi yapicak. ** daha bitmedi

```
python3 preprocess.py
```


##### BERT
BERT'in servis olarak kullanildigi pakete ihtiyaci var calismak icin.

```
python3 bert.py
```

##### CHARACTER NGRAM
karakter ngram benzerligi yontemi. cumle-paragraf ve cumle-cumle olmak uzere iki yontem var.

```
python3 char_ngram.py
```

##### WORD2VEC
cumlenin kelime vektoru ortalamasi alinip secenek ile paragraftaki cumlelerin uzakliklari karsilastiriliyor.

```
python3 word2vec_distance.py
```
