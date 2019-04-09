from logger import logger
import nltk
import csv
import re
from collections import Counter
import numpy as np
import khi2

urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-\?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)
monitor = True



interestingnpattern = re.compile(r".*anorex.*")

filename = "/home/jussi/data/erisk/training.csv"
sentencerepository = {}
author = {}
illness = {}
def readonecsvfile(filename, loglevel=False):
    """Read one file with csv lines such and return the text found in the specified slots."""
    global illness, author, onecounter, nilcounter, sentencerepository
    logger(filename, loglevel)
    with open(filename, errors="replace", newline="", encoding='utf-8') as inputtextfile:
        logger("Loading " + filename, loglevel)
        linereader = csv.reader(inputtextfile, delimiter=',', quotechar='"')
        for line in linereader:
            id = line[0] + line[1]
            author[id] = line[1]
            text = line[3] + " " + line[4]
            illness[id] = line[5]
            text = urlpatternexpression.sub("URL", text)
            text = handlepattern.sub("HANDLE", text)
            sentencerepository[id] = text
            #  logger("{} {} {}".format(id, illness[id], text), monitor)

onecounter = Counter()
nilcounter = Counter()
onef = 0
nilf = 0
readonecsvfile(filename)
for t in sentencerepository:
    words = nltk.word_tokenize(sentencerepository[t].lower())
    if illness[t] == "1":
        onecounter.update(words)
        onef += len(words)
    else:
        nilcounter.update(words)
        nilf += len(words)

khi2score = {}
debug = False
best = {}
n = 100
hap1 = []
for w in onecounter:
    if w not in nilcounter:
        hap1.append((onecounter[w], w))
    else:
        xtab = np.array([[onecounter[w],nilcounter[w]],
                            [onef - onecounter[w], nilf - nilcounter[w]]])
        logger(str(xtab), debug)
        khi2score[w] = khi2.khi2(xtab, debug, True)
best1 = sorted(khi2score, key=lambda k: khi2score[k], reverse=True)  # chk to see sign!!!
best = [(onecounter[k], khi2score[k], k) for k in best1]
hap = [f for f in hap1 if f[0] > 3]
print("{}".format(hap[:n]))
print("{}".format(best[:n]))