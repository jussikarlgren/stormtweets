import re
import os
import csv
import random

from confusionmatrix import ConfusionMatrix
from logger import logger
import hyperdimensionalsemanticspace
import squintinglinguist
import sparsevectors
from sequencelabels import SequenceLabels
# from nltk import word_tokenize, sent_tokenize
# import numpy as NP
# import pca
# ===========================================================================
debug = False
monitor = True
error = True
dimensionality = 2000
denseness = 10
ngramwindow = 3
ctxspace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
ctxspace.inputwordspace("/home/jussi/data/vectorspace/ctxspace.hyp")
docspace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
docspace.inputwordspace("/home/jussi/data/vectorspace/documentspace.hyp")
seq = SequenceLabels(dimensionality, ngramwindow)
datadirectory = "/home/jussi/data/erisk/"
outputdirectory = "/home/jussi/data/erisk/vectorspace/"
urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9.\-?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)


def tokenvector(tokenlist, initialvector=None,
                weights=True, loglevel=False):
    if initialvector is None:
        initialvector = sparsevectors.newemptyvector(dimensionality)
    for item in tokenlist:
        if not weights or str(item).startswith("JiK"):  # cxg features should not be weighted the same way lex feats are
            weight = 1
        else:
            weight = ctxspace.languagemodel.frequencyweight(item, True)
        ctxspace.observe(item, True)
        tmp = initialvector
        initialvector = sparsevectors.sparseadd(initialvector,
                                                sparsevectors.normalise(ctxspace.contextspace[item]), weight)
        if loglevel:
            logger(item + " " + str(weight) + " " + str(sparsevectors.sparsecosine(tmp, initialvector)), loglevel)
    return initialvector


def rolevector(roledict, initialvector=None, loglevel=False):
    if initialvector is None:
        initialvector = sparsevectors.newemptyvector(dimensionality)
    for role in roledict:
        for item in roledict[role]:
            ctxspace.observe(item, False, debug)
            tmp = initialvector
            initialvector = sparsevectors.sparseadd(initialvector,
                                                    sparsevectors.normalise(
                                                        ctxspace.useoperator(ctxspace.indexspace[item], role)))
            if loglevel:
                logger(role + " " + item + " " + str(sparsevectors.sparsecosine(tmp, initialvector)), loglevel)
    return initialvector


def runbatchtest(n: int=100):
    print(ticker)
    keylist = list(vectorrepositoryall.keys())[:n]
    random.shuffle(keylist)
    split = int(len(keylist) * fraction)
    train = keylist[:split]
    test = keylist[split:]
    logger("{} train vs {} test".format(len(train), len(test)))
    ones = []
    nils = []
    dummymaxconfusionmatrix = ConfusionMatrix()
    dummyrandomconfusionmatrix = ConfusionMatrix()
    centroidconfusionmatrix = ConfusionMatrix()
    poolconfusionmatrix = ConfusionMatrix()
    for trainitem in test:
        if illness[trainitem] == "1":
            ones.append(vectorrepositoryall[trainitem])
        else:
            nils.append(vectorrepositoryall[trainitem])
    onecentroid = sparsevectors.centroid(ones)
    nilcentroid = sparsevectors.centroid(nils)
    if len(nils) > len(ones):
        dummymaxguess = "0"
    else:
        dummymaxguess = "1"
    # factor = len(ones) / len(nils)
    #  no
    factor = 1 / 2
    for testitem in test:
        dummymaxconfusionmatrix.addconfusion(illness[testitem], dummymaxguess)
        if random.random() > factor:
            dummyrandomguess = "0"
        else:
            dummyrandomguess = "1"
        dummyrandomconfusionmatrix.addconfusion(illness[testitem], dummyrandomguess)
        probe = vectorrepositoryall[testitem]
        resultc = "0"
        i1 = sparsevectors.sparsecosine(probe, onecentroid)
        n1 = sparsevectors.sparsecosine(probe, nilcentroid)
        if i1 > n1:
            resultc = "1"
        centroidconfusionmatrix.addconfusion(illness[testitem], resultc)
        probeneighbours = {}
        for targetitem in train:
            probeneighbours[targetitem] = sparsevectors.sparsecosine(probe, vectorrepositoryall[targetitem])
        sortedfriends = sorted(probeneighbours, key=lambda hh: probeneighbours[hh], reverse=True)[:pooldepth]
        illity = 0
        result = "0"
        for friend in sortedfriends:
            if illness[friend] == "1":
                illity += 1
        if illity > pooldepth * factor:
            result = "1"
        nullity = pooldepth - illity
        poolconfusionmatrix.addconfusion(illness[testitem], result)
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(testitem, illness[testitem],
                                                          resultc, i1, n1,
                                                          result, illity, nullity, pooldepth))
    print("RANDOM ----------------")
    dummyrandomconfusionmatrix.evaluate()
    print("MAX ----------------")
    dummymaxconfusionmatrix.evaluate()
    print("CENTROID ----------------")
    centroidconfusionmatrix.evaluate()
    print("NEIGHBOURS --------------")
    poolconfusionmatrix.evaluate()


def processsentences(ftag: str="erisk"):
    global sentencerepository, illness, vectorrepositoryall, featurerepository, ticker
    for key in sentencerepository:
        sentence = sentencerepository[key]
        fs = squintinglinguist.featurise(sentence)
        logger(sentence, debug)
        try:
            fcxg = fs["features"]
            fpos = fs["pos"]
            fsem = fs["roles"]
            fwds = fs["words"]
            vecidx = tokenvector(fwds, None, True, debug)
            vecseq = seq.sequencevector(fpos, vecidx, debug)
            veccxg = tokenvector(fcxg, vecseq, False, debug)
            vecsem = rolevector(fsem, veccxg, debug)
            vectorrepositoryall[key] = vecsem
        except KeyError:
            pass
        logger("{}: {} -> {}".format(key, sentence, fs), debug)
        if ticker % 100 == 0:
            logger(str(ticker) + " sentences processed", monitor)
            with open("{}/{}.{}.vectors".format(outputdirectory, ftag, ticker), "w+") as outputfile:
                for item in vectorrepositoryall:
                    outputfile.write("{}\t{}\t{}\n".format(item, illness[item], vectorrepositoryall[item]))
            squintinglinguist.restartCoreNlpClient()
        if ticker % 10000 == 0:
            runbatchtest()
        ticker += 1


def readonecsvfile(filename, loglevel=False):
    """Read one file with csv lines such and return the text found in the specified slots."""
    global illness, author, sentencerepository
    logger(filename, loglevel)
    with open(filename, errors="replace", newline="", encoding='utf-8') as inputtextfile:
        logger("Loading " + filename, loglevel)
        linereader = csv.reader(inputtextfile, delimiter=',', quotechar='"')
        for line in linereader:
            key = line[0] + line[1]
            author[key] = line[1]
            text = line[3] + " " + line[4]
            illness[key] = line[5]
            text = urlpatternexpression.sub("URL", text)
            text = handlepattern.sub("HANDLE", text)
            sentencerepository[key] = text
            logger("{} {} {}".format(key, illness[key], text), monitor)


sentencerepository = {}
vectorrepositoryall = {}
featurerepository = {}
antal = 5
ticker = 0
author = {}
illness = {}
filepattern = re.compile(r"^t.*csv$")
filenamelist = []
for filenamecandidate in os.listdir(datadirectory):
    if filepattern.match(filenamecandidate):
        logger(filenamecandidate, monitor)
        filenamelist.append(os.path.join(datadirectory, filenamecandidate))
    logger(filenamelist, monitor)

logger("starting with " + str(len(filenamelist)) + " files: " + str(filenamelist), monitor)
debug = False
runtest = True
extradebug = False
fraction = 8 / 10
pooldepth = 9
tag = 1
for f in filenamelist:
    logger(f, monitor)
    readonecsvfile(f)
    processsentences(tag)
    tag += 1
