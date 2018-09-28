import re
from logger import logger
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
datadirectory = "/home/jussi/data/storm/fixed"
outputdirectory = "/home/jussi/data/storm/output"
sentencerepository = {}
vectorrepository = {}
featurerepository = {}
index = 0


def processsentences(sents, testing=True):
    global sentencerepository, vectorrepository, featurerepository, index, ticker
    for s in sents:
        index += 1
        key = "s" + str(index)
        if s in sentencerepository.values():
            continue
        f = featurise(s)
        t = tokenise(s.lower())
        vec = space.utterancevector(key, f + t, "nil")
        sentencerepository[key] = s
        vectorrepository[key] = vec
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
    for probe in ["afraid", "before", "later", "i love the hurricane", "i said i love the hurricane", "storm bitch"]:  # "hurricane", "JiKsayverbs","hit"]:
        pindex += 1
        f = featurise(probe)
        t = tokenise(probe.lower())
        feats = f + t
        pkey = "p" + str(pindex)
        vec = space.utterancevector(pkey, feats, "nil")
        n = {}
        s = {}
        d10 = 0
        for v in sentencerepository:
            d = space.similarity(vec, vectorrepository[v])
            if d > 0.1:
                d10 += 1
                n[v] = d
                s[v] = sentencerepository[v]
        logger(str(d10) + " sentences closer than 0.1", True)
        m = sorted(s, key=lambda k: n[k], reverse=True)[:antal]
        for mc in m:
            if n[mc] > 0.1:
                logger(mc + " " + probe + "<-" + str(n[mc]) + "->" + s[mc], True)
                for fff in feats:
                    dd = space.similarity(space.indexspace[fff], vectorrepository[mc])
                    ee = space.similarity(space.indexspace[fff], vec)
                    logger(str(fff) + " " + str(ee) + " " + str(dd), True)

# show that constructional items work the same way

# show that permuted semantic roles work "semantic grep"

# show that morphological things work (use eg finnish material) the same way

#  show that pos sequences work
