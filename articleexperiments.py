import re
from logger import logger
import tweetfilereader
import hyperdimensionalsemanticspace
from squintinglinguist import featurise, tokenise, window
import semanticroles

debug = False
monitor = True
error = True
dimensionality = 2000
denseness = 10
space = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
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
index = 0

def processsentences(sents, testing=True):
    global sentencerepository, vectorrepositoryidx, featurerepository, index, ticker, sequencelabels, vectorrepositoryseq
    for s in sents:
        index += 1
        key = "s" + str(index)
        if s in sentencerepository.values():
            continue
        f = featurise(s)
        t = tokenise(s.lower())
        ss = semanticroles.semanticdependencyparse(s)[0]

        vecidx = space.utterancevector(key, s, f + t, None, False)
        vecseq = space.utterancevector(key, s, f + t, None, True)
        vecsem = space.utterancevector(key, s, f + t, None, True)
        veccxg = space.utterancevector(key, s, f + t, None, True)
        sentencerepository[key] = s
        vectorrepositoryidx[key] = vecidx
        vectorrepositoryseq[key] = vecseq
        featurerepository[key] = f + t
        logger(str(key) + ":" + str(s)+"->"+str(f)+"+"+str(t), debug)
        if ticker > 1000:
            logger(str(index) + " sentences processed", monitor)
            ticker = 0
        ticker += 1
#        for w in words:
#            space.addintoitem(w, vec)
antal = 5
files = tweetfilereader.getfilelist(datadirectory, re.compile(r".*09\-.*"))
ticker = 0
index = 0
for f in files:
    logger(f, monitor)
    sentences = tweetfilereader.doonetweetfile(f)
    processsentences(sentences)
    space.outputwordspace(outputdirectory + "/" + str(index) + ".wordspace")
    pindex = 0
    for probe in ["i am afraid", "afraid", "i love the hurricane", "i said i love the hurricane", "you are a bitch"]:  # "hurricane", "JiKsayverbs","hit"]:
        pindex += 1
        f = featurise(probe)
        t = tokenise(probe.lower())
        feats = f + t
        pkey = "p" + str(pindex)
        vecidx = space.utterancevector(pkey, probe, feats)
        veccxg = space.utterancevector(pkey, probe, feats, None, True)
        vecseq = space.utterancevector(pkey, probe, feats, None, True)
        vecsem = space.utterancevector(pkey, probe, feats, None, True)

        neighboursByIndex = {}
        neighboursByIndexSeq = {}
        neighboursByIndex2 = {}
        neighboursByIndexSeq2 = {}
        for v in sentencerepository:
            d = space.similarity(vecidx, vectorrepositoryidx[v])
            d2 = space.similarity(vecidx, vectorrepositoryseq[v])
            if d > 0.1:
                neighboursByIndex[v] = d
                neighboursByIndex2[v] = d2
            dp = space.similarity(vecseq, vectorrepositoryidx[v])
            dp2 = space.similarity(vecseq, vectorrepositoryseq[v])
            if dp > 0.1:
                neighboursByIndexSeq[v] = dp
                neighboursByIndexSeq2[v] = dp2
        m0 = sorted(neighboursByIndex, key=lambda k: neighboursByIndex[k], reverse=True)[:antal]
        print("---- ix " + probe)
        kk = 0
        for mc in m0:
            if neighboursByIndex[mc] > 0.1:
                kk += 1
                print(kk, str(neighboursByIndex[mc]), str(neighboursByIndex2[mc]), sentencerepository[mc], sep="\t")
                for fff in feats:
                    print(" ", fff,
                          space.similarity(space.indexspace[fff], vecidx),
                          space.similarity(space.indexspace[fff], vectorrepositoryidx[mc]),
                          space.similarity(space.indexspace[fff], vectorrepositoryseq[mc]),
                          sep="\t")
        m1 = sorted(neighboursByIndexSeq2, key=lambda k: neighboursByIndexSeq2[k], reverse=True)[:antal]
        print("---- sq " + probe)
        kk = 0
        for mc in m1:
            if neighboursByIndexSeq2[mc] > 0.1:
                kk += 1
                print(kk, str(neighboursByIndexSeq[mc]), str(neighboursByIndexSeq2[mc]), sentencerepository[mc], sep="\t")
        if space.sequencelabels.changed:
            space.sequencelabels.save()
