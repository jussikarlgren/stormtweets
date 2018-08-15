import math

import hyperdimensionalsemanticspace
import sparsevectors
import stringsequencespace
import nltk
import re
from logger import logger
import os

debug = False
monitor = True
error = True
dimensionality = 2000
denseness = 10
space = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
filename = "/home/jussi/data/mini.txt"
sentencerepository = {}
vectorrepository = {}
index = 0
space.bign = 10000

def getfilelist(resourcedirectory="/home/jussi/data/storm/fixed/", pattern=re.compile(r".*irma")):
    filenamelist = []
    for filenamecandidate in os.listdir(resourcedirectory):
        if pattern.match(filenamecandidate):
            logger(filenamecandidate, debug)
            filenamelist.append(filenamecandidate)
    logger(filenamelist, debug)
    return sorted(filenamelist)

def getsentencesfromlinefile(filename):
    sentences = []
    with open(filename, "r") as newsfile:
        newsline = newsfile.readline()
        nl = 0
        while newsline:
            sansapostrophe = newsline.replace("'","")
            nl += 1
            sents = nltk.sent_tokenize(sansapostrophe.lower())
            sentences = sentences + sents
            newsline = newsfile.readline()
    return sentences


def processsentences(sents):
    global sentencerepository, vectorrepository, index
    for s in sents:
        index += 1
        words = nltk.word_tokenize(s)
        vec = space.textvector(words, True)
        sentencerepository[index] = s
        vectorrepository[index] = vec
        logger(str(s), debug)
        for w in words:
            space.addintoitem(w, vec)

sentences = getsentencesfromlinefile(filename)
antals = len(sentences)
processsentences(sentences)

if False:
    for i in space.items():
        print(i,    space.globalfrequency[i],   space.bign,     space.frequencyweight(i),             sep="\t")

# show that lexical stats work use weighting
if False:
    for probe in ["jussi", "boat", "fun"]: # "["hearts", "turtle", "cat", "rabbit", "queen", "and", "off"]:
        n = {}
        for v in vectorrepository:
            n[v] = sparsevectors.sparsecosine(space.indexspace[probe], vectorrepository[v])
        m = sorted(sentencerepository, key=lambda k: n[k], reverse=True)
        for mc in m:
            if n[mc] > 0.0001:
                print(probe, mc, n[mc], sentencerepository[mc])
        print(space.contexttoindexneighbourswithweights(probe))

for v in vectorrepository:
    print(v, sentencerepository[v], sep="\t", end="\t")
#    print(v, vectorrepository[v])
    ww = nltk.word_tokenize(sentencerepository[v])
    vec = sparsevectors.newemptyvector(dimensionality)
#    for www in ww:
#        print(www, space.indexspace[www], space.globalfrequency[www], space.frequencyweight(www), sparsevectors.sparsecosine(space.indexspace[www], vectorrepository[v]))
    nvn = {}
    for www in ww:
        nvn[www] = sparsevectors.sparsecosine(space.indexspace[www], vectorrepository[v])
        vec = sparsevectors.sparseadd(vec, sparsevectors.normalise(space.indexspace[www]), space.frequencyweight(www))
    m = sorted(ww, key=lambda k: nvn[k], reverse=True)[:5]
    for mc in m:
        if nvn[mc] > 0.0001:
            print(mc, nvn[mc], sep=":", end="\t")
    print()

if False:
    for w in space.items():
        print(w, space.globalfrequency[w], space.indexspace[w], sep="\t")
        print("\t\t", space.contextspace[w])


# show that constructional items work the same way

# show that permuted semantic roles work "semantic grep"

# show that morphological things work (use eg finnish material) the same way

#  show that pos sequences work
