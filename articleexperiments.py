from nltk import word_tokenize
from nltk import sent_tokenize
import re
from logger import logger
import os
import tweetfilereader
import hyperdimensionalsemanticspace
from squintinglinguist import featurise, tokenise, window

debug = False
monitor = True
error = True
dimensionality = 2000
denseness = 10
space = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
filename = "/home/jussi/data/mini.txt"
datadirectory = "/home/jussi/data/storm/fixed/"
sentencerepository = {}
vectorrepository = {}
index = 0
space.bign = 10000




def processsentences(sents, testing=True):
    global sentencerepository, vectorrepository, index, ticker
    for s in sents:
        print(s)
        index += 1
        key = "s" + str(index)
        f = featurise(s)
        t = tokenise(s)
        vec = space.utterancevector(key, True)
        sentencerepository[key] = s
        vectorrepository[key] = vec
        logger(str(s)+"->"+str(f)+"+"+str(t), monitor)
        if ticker > 1000:
            logger(str(ticker) + " sentences processed", monitor)
            ticker = 0
        ticker += 1
#        for w in words:
#            space.addintoitem(w, vec)

files = tweetfilereader.getfilelist(datadirectory, re.compile(r".*09-01.*irma"))
ticker = 0
index = 0
for f in files:
    logger(f, monitor)
    sentences = tweetfilereader.doonetweetfile(f)
    processsentences(sentences)


# show that lexical stats work use weighting
for probe in ["hurricane", "boat", "terror"]:
    n = {}
    for v in sentencerepository:
        n[v] = space.contextsimilarity(probe, v)
        m = sorted(sentencerepository, key=lambda k: n[k], reverse=True)
        for mc in m:
            if n[mc] > 0.0001:
               print(probe, mc, n[mc], sentencerepository[mc])

#if False:
#    for v in vectorrepository:
#        print(v, sentencerepository[v], sep="\t", end="\t")
#    #    print(v, vectorrepository[v])
#        ww = nltk.word_tokenize(sentencerepository[v])
#        vec = sparsevectors.newemptyvector(dimensionality)
#    #    for www in ww:
#    #        print(www, space.indexspace[www], space.globalfrequency[www], space.frequencyweight(www), sparsevectors.sparsecosine(space.indexspace[www], vectorrepository[v]))
#        nvn = {}
#        for www in ww:
#            nvn[www] = sparsevectors.sparsecosine(space.indexspace[www], vectorrepository[v])
#            vec = sparsevectors.sparseadd(vec, sparsevectors.normalise(space.indexspace[www]), space.frequencyweight(www))
#        m = sorted(ww, key=lambda k: nvn[k], reverse=True)[:5]
#        for mc in m:
#            if nvn[mc] > 0.0001:
#                print(mc, nvn[mc], sep=":", end="\t")
#        print()

#if False:
#    for w in space.items():
#        print(w, space.globalfrequency[w], space.indexspace[w], sep="\t")
#        print("\t\t", space.contextspace[w])


# show that constructional items work the same way

# show that permuted semantic roles work "semantic grep"

# show that morphological things work (use eg finnish material) the same way

#  show that pos sequences work
