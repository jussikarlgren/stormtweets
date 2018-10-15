import re
from logger import logger
import tweetfilereader
import hyperdimensionalsemanticspace
from squintinglinguist import featurise, tokenise, window
import squintinglinguist
import sparsevectors
from sequencelabels import SequenceLabels

# ===========================================================================
debug = False
monitor = True
error = True
dimensionality = 2000
denseness = 10
ngramwindow = 3
space = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
seq = SequenceLabels(dimensionality, ngramwindow)
filename = "/home/jussi/data/mini.txt"
datadirectory = "/home/jussi/data/storm/fixed"
outputdirectory = "/home/jussi/data/storm/output"
sentencerepository = {}
vectorrepositoryidx = {}
vectorrepositoryseq = {}
vectorrepositorysem = {}
vectorrepositorycxg = {}
vectorrepositoryall = {}
featurerepository = {}
seq.restore("/home/jussi/data/storm/vectorspace/sequencemodel.hyp")


index = 0
antal = 5
files = tweetfilereader.getfilelist(datadirectory, re.compile(r".*09\-01.*"))
ticker = 0


def tokenvector(tokenlist, initialvector=None,
                weights=True, loglevel=True):
    if initialvector is None:
        initialvector = sparsevectors.newemptyvector(dimensionality)
    for item in tokenlist:
        if not weights or str(item).startswith("JiK"):  # cxg features should not be weighted the same way lex feats are
            weight = 1
        else:
            weight = space.languagemodel.frequencyweight(item, True)
        space.observe(item, True)
        tmp = initialvector
        initialvector = sparsevectors.sparseadd(initialvector, sparsevectors.normalise(space.indexspace[item]), weight)
        if loglevel:
            logger(item + " " + str(weight) + " " + str(sparsevectors.sparsecosine(tmp, initialvector)), loglevel)
#    if update:
#        for item in tokenlist:
#            for otheritem in tokenlist:
#                if otheritem == item:
#                    continue
#                updateweight = 1
#                if updateweights:
#                    updateweight = space.languagemodel.frequencyweight(item)
#                space.observecollocation(item, otheritem, updateweight)
    return initialvector


def rolevector(roledict, initialvector=None, loglevel=True):
    if initialvector is None:
        initialvector = sparsevectors.newemptyvector(dimensionality)
    for role in roledict:
        space.observe(roledict[role], False)
        tmp = initialvector
        initialvector = sparsevectors.sparseadd(initialvector,
                                sparsevectors.normalise(space.useoperator(space.indexspace[roledict[role]], role)))
        if loglevel:
            logger(role + " " + str(sparsevectors.sparsecosine(tmp, initialvector)), loglevel)
    return initialvector


def processsentences(sents, testing=True):
    global sentencerepository, vectorrepositoryidx, featurerepository, index, ticker, sequencelabels, vectorrepositoryseq
    for s in sents:
        index += 1
        key = "s" + str(index)
        if s in sentencerepository.values():
            continue
        fs = featurise(s)
        fcxg = fs["features"]
        fpos = fs["pos"]
        fsem = fs["roles"]
        fwds = fs["words"]
        vecidx = tokenvector(fwds, None, True, debug)
        vecseq = seq.sequencevector(fpos, vecidx)
        logger(sparsevectors.sparsecosine(vecseq, vecidx), monitor)
        veccxg = tokenvector(fcxg, vecidx, False, debug)
        logger(sparsevectors.sparsecosine(veccxg, vecidx), monitor)
        vecsem = rolevector(fsem, veccxg, False)
        logger(sparsevectors.sparsecosine(veccxg, vecsem), monitor)
        sentencerepository[key] = s
        vectorrepositoryidx[key] = vecidx
        vectorrepositoryseq[key] = vecseq
        vectorrepositorycxg[key] = veccxg
        vectorrepositorysem[key] = vecsem
        featurerepository[key] = fs
        logger(str(key) + ":" + str(s) + "->" + str(fs), debug)
        if ticker > 1000:
            logger(str(index) + " sentences processed", monitor)
            ticker = 0
        ticker += 1

logger("starting with " + str(files), monitor)

for f in files:
    logger(f, monitor)
    sentences = tweetfilereader.doonetweetfile(f)
    processsentences(sentences)
    space.outputwordspace(outputdirectory + "/" + str(index) + ".wordspace")
    pindex = 0
    for probe in ["i am afraid", "afraid", "i am afraid of the hurricane", "i said i was afraid the hurricane",
                  "the storm is a bitch"]:
        pindex += 1
        feats = featurise(probe)
        pkey = "p" + str(pindex)
        vecidx = tokenvector(feats["words"], None, True, debug)
        vecseq = seq.sequencevector(feats["pos"], vecidx, monitor)
        veccxg = tokenvector(feats["features"], vecseq, monitor)
        vecsem = rolevector(feats["roles"], veccxg, monitor)
        neighboursByIdx = {}
        neighboursBySeq = {}
        neighboursByCxg = {}
        neighboursBySem = {}
        for v in sentencerepository:
            d1 = space.similarity(vecidx, vectorrepositorysem[v])
            d2 = space.similarity(vecseq, vectorrepositorysem[v])
            d3 = space.similarity(veccxg, vectorrepositorysem[v])
            d4 = space.similarity(vecsem, vectorrepositorysem[v])
            if d1 > 0.1:
                neighboursByIdx[v] = d1
            if d2 > 0.1:
                neighboursBySeq[v] = d2
            if d3 > 0.1:
                neighboursByCxg[v] = d3
            if d4 > 0.1:
                neighboursBySem[v] = d4
        closestneighbours = sorted(neighboursByIdx, key=lambda k: neighboursByIdx[k], reverse=True)[:antal]
        print("---- idx " + probe)
        kk = 0
        for mc in closestneighbours:
            kk += 1
            if mc in neighboursByIdx:
                s1 = neighboursByIdx[mc]
            else:
                s1 = 0
            if mc in neighboursBySeq:
                s2 = neighboursBySeq[mc]
            else:
                s2 = 0
            if mc in neighboursByCxg:
                s3 = neighboursByCxg[mc]
            else:
                s3 = 0
            if mc in neighboursBySem:
                s4 = neighboursBySem[mc]
            else:
                s4 = 0
            print(kk,
                  str(s1), str(s2), str(s3), str(s4),
                  sentencerepository[mc],
                  sep="\t")
        closestneighbours = sorted(neighboursBySeq, key=lambda k: neighboursBySeq[k], reverse=True)[:antal]
        print("---- seq " + probe)
        kk = 0
        for mc in closestneighbours:
            kk += 1
            if mc in neighboursByIdx:
                s1 = neighboursByIdx[mc]
            else:
                s1 = 0
            if mc in neighboursBySeq:
                s2 = neighboursBySeq[mc]
            else:
                s2 = 0
            if mc in neighboursByCxg:
                s3 = neighboursByCxg[mc]
            else:
                s3 = 0
            if mc in neighboursBySem:
                s4 = neighboursBySem[mc]
            else:
                s4 = 0
            print(kk,
                  str(s1), str(s2), str(s3), str(s4),
                  sentencerepository[mc],
                  sep="\t")
        closestneighbours = sorted(neighboursByCxg, key=lambda k: neighboursByCxg[k], reverse=True)[:antal]
        print("---- cxg " + probe)
        kk = 0
        for mc in closestneighbours:
            kk += 1
            if mc in neighboursByIdx:
                s1 = neighboursByIdx[mc]
            else:
                s1 = 0
            if mc in neighboursBySeq:
                s2 = neighboursBySeq[mc]
            else:
                s2 = 0
            if mc in neighboursByCxg:
                s3 = neighboursByCxg[mc]
            else:
                s3 = 0
            if mc in neighboursBySem:
                s4 = neighboursBySem[mc]
            else:
                s4 = 0
            print(kk,
                  str(s1), str(s2), str(s3), str(s4),
                  sentencerepository[mc],
                  sep="\t")
        closestneighbours = sorted(neighboursBySem, key=lambda k: neighboursBySem[k], reverse=True)[:antal]
        print("---- sem " + probe)
        kk = 0
        for mc in closestneighbours:
            kk += 1
            if mc in neighboursByIdx:
                s1 = neighboursByIdx[mc]
            else:
                s1 = 0
            if mc in neighboursBySeq:
                s2 = neighboursBySeq[mc]
            else:
                s2 = 0
            if mc in neighboursByCxg:
                s3 = neighboursByCxg[mc]
            else:
                s3 = 0
            if mc in neighboursBySem:
                s4 = neighboursBySem[mc]
            else:
                s4 = 0
            print(kk,
                  str(s1), str(s2), str(s3), str(s4),
                  sentencerepository[mc],
                  sep="\t")
        if seq.changed:
            seq.save()
