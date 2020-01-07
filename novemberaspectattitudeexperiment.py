#  experiments on attitude and aspect
#  november 2019 Jussi Karlgren jussi@lingvi.st
#  ============================================
#  Read large number of sentences
#  Sentencetokenize, Wordtokenize
#  Find a polarity lexicon (Bing Liu e.g.) to establish attitude lexicon
#  Find a list of time and manner adverbials
#  Establish list of hedges
#  Establish list of amplifiers
#  Process sentences with StanfordNLP to get tense information and modals and subclauses
#  Crosstabulate attitudinal terms with
#          adverbials, modals, tense, hedges, amplifiers, matrix vs subclause, this, that here
#  ============================================
import nltk
import stanfordnlp
import simpletextfilereader
import hyperdimensionalsemanticspace
from logger import logger
from nltk import word_tokenize, sent_tokenize
from lexicalfeatures import lexicon
#  ============================================
tag = "initialexperiment"
datadirectory = "/home/jussi/data/storm/fixed"
outputdirectory = "/home/jussi/tmp"
manyfiles = False
dimensionality = 2000
density = 10
monitor = True
error = True
debug = True
index = 0
sentencerepository = {}
outfile = outputdirectory + "/" + "attispace" + tag + ".hyp"
if manyfiles:
    files = simpletextfilereader.getfilelist(datadirectory, r".*09*.i*")
else:
    files = [datadirectory + "/" + "2017-08-25.EN.twitter.jq.harvey"]
space = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, density)
#  ============================================


def featurise(text, loglevel=False):
    returnfeatures = {}
    features = []
    words = []
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        for word in words:
            for feature in lexicon:
                if word.lower() in lexicon[feature]:
                    features.append("JiK" + feature)
        returnfeatures["features"] = features
        poses = postags(text)
        returnfeatures["pos"] = poses
    returnfeatures["words"] = words
    logger(text + "->" + str(features), loglevel)
    return returnfeatures


def processsentences(sents, index:int):
     for s in sents:
        index += 1
        key = "s" + str(index)
        if s in sentencerepository.values():
            continue
        fs = featurise(s)
        logger(s, debug)
        fcxg = fs["features"]
        fpos = fs["pos"]
        fsem = fs["roles"]
        fwds = fs["words"]
        logger(fwds, debug)
        logger(fpos, debug)
        logger(fcxg, debug)
        logger(fsem, debug)
        vecidx = tokenvector(fwds, None, True, debug)
        vecseq = seq.sequencevector(fpos, None, debug)
        vecis = sparsevectors.sparseadd(vecidx, vecseq, 1, True)
        logger("idx - comb\t" + str(sparsevectors.sparsecosine(vecidx, vecis)), debug)
        logger("seq - comb\t" + str(sparsevectors.sparsecosine(vecseq, vecis)), debug)
        veccxg = tokenvector(fcxg, vecis, False, debug)
        logger("comb - cxg\t" + str(sparsevectors.sparsecosine(vecis, veccxg)), debug)
        logger("idx - cxg\t" + str(sparsevectors.sparsecosine(vecidx, veccxg)), debug)
        logger("seq - cxg\t" + str(sparsevectors.sparsecosine(veccxg, vecseq)), debug)
        vecsem = rolevector(fsem, veccxg, debug)
        logger("idx - sem\t" + str(sparsevectors.sparsecosine(vecidx, vecsem)), debug)
        logger("seq - sem\t" + str(sparsevectors.sparsecosine(vecseq, vecsem)), debug)
        logger("comb - sem\t" + str(sparsevectors.sparsecosine(vecis, vecsem)), debug)
        logger("cxg - sem\t" + str(sparsevectors.sparsecosine(veccxg, vecsem)), debug)
        sentencerepository[key] = s
        vectorrepositoryidx[key] = vecidx
        vectorrepositoryseq[key] = vecseq
        vectorrepositorycxg[key] = veccxg
        vectorrepositorysem[key] = vecsem
        featurerepository[key] = fs
        logger(str(key) + ":" + str(s) + "->" + str(fs), debug)
        if index%1000 == 0:
            logger(str(index) + " sentences processed", monitor)


for f in files:
    logger(f, monitor)
    sentences = simpletextfilereader.doonetweetfile(f, targetterms)
    processsentences(sentences, index)
