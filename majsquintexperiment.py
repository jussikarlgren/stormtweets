import re
from logger import logger
import simpletextfilereader
import hyperdimensionalsemanticspace
from squintinglinguist import featurise
import squintinglinguist
import sparsevectors
from sequencelabels import SequenceLabels
from nltk import word_tokenize
# ===========================================================================
# config
debug = False
monitor = True
error = True
runtest = True
restorespace = False
manyfiles = False
dimensionality = 2000
denseness = 10
ngramwindow = 3
index = 0
antal = 5
datadirectory = "/home/jussi/data/storm/fixed"
outputdirectory = "/home/jussi/data/storm/output"
stormterms = {["irma", "#irma", "#harvey", "harvey", "hurricane", "#hurricane", "storm", "hurricaneharvey",
                  "harvey2017", "#harvey2017", "hurricaneirma", "irma2017", "hurricaneirma2017", "hurricanes", "flood",
                  "harveystorm", "irmastorm", "hurricaineharvey", "hurricaineirma", "hurricaneharvey2017", "disaster",
                  "fema", "post-harvey", "post-irma", "superstorm", "super-storm",
                  "stormharvey", "stormirma", "harveyhurricane", "irmahurricane", "majorhurricane", "stormprep",
                  "extremeweather", "evacuation", "flashflood" "flashfloodwatch", "harveyrelief", "houston", "texas",
                  "puertorico", "florida",
                  "#hurricaneharvey", "#hurricaneharvey2017", "#hurricaneirma", "#hurricaneirma2017"]}
# ===========================================================================
# init
tag = "0"
outfile = outputdirectory + "/" + "stormspace" + tag + ".hyp"
if manyfiles:
    files = simpletextfilereader.getfilelist(datadirectory, re.compile(r".*09*.i*"))
else:
    files = [datadirectory + "/2017-08-25.EN.twitter.jq.harvey"]
ticker = 0
space = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
if restorespace:
    space.inputwordspace("/home/jussi/data/storm/vectorspace/articlespace.hyp")
seq = SequenceLabels(dimensionality, ngramwindow)
if restorespace:
    seq.restore("/home/jussi/data/storm/vectorspace/sequencemodel.hyp")
# ===========================================================================

logger("starting with " + str(len(files)) +" files: " + str(files), monitor)
for f in files:
    logger(f, monitor)
    sentences = simpletextfilereader.doonetweetfile(f, stormterms)
    processsentences(sentences)
    space.outputwordspace(outputdirectory + "/" + str(index) + ".wordspace")
    pindex = 0
    if runtest == True:
        for probe in ["i am afraid of the hurricane", "i said i was afraid the hurricane", "getting as far away from this hurricane as possible",
                      "the storm is a bitch"]:
            pindex += 1
            feats = featurise(probe)
            pkey = "p" + str(pindex)
            vecidx = tokenvector(feats["words"], None, True)
            vecseq = seq.sequencevector(feats["pos"], None)
            veccxg = tokenvector(feats["features"], None, False)
            vecsem = rolevector(feats["roles"], None)
            vec1 = seq.sequencevector(feats["pos"], vecidx)
            vec2 = tokenvector(feats["features"], vec1, False)
            vectot = rolevector(feats["roles"], vec2)
            neighboursByIdx = {}
            neighboursBySeq = {}
            neighboursByCxg = {}
            neighboursBySem = {}
            neighboursByTot = {}
            for v in sentencerepository:
                d1 = space.similarity(vecidx, vectorrepositorysem[v])
                d2 = space.similarity(vec1, vectorrepositorysem[v])
                d3 = space.similarity(vec2, vectorrepositorysem[v])
                d4 = space.similarity(vecsem, vectorrepositorysem[v])
                d5 = space.similarity(vectot, vectorrepositorysem[v])
                if d1 > 0.1:
                    neighboursByIdx[v] = d1
                if d2 > 0.1:
                    neighboursBySeq[v] = d2
                if d3 > 0.1:
                    neighboursByCxg[v] = d3
                if d4 > 0.1:
                    neighboursBySem[v] = d4
                if d5 > 0.1:
                    neighboursByTot[v] = d5
            closestneighbours = sorted(neighboursByIdx, key=lambda k: neighboursByIdx[k], reverse=True)[:antal]
            print("---- idx " + str(index) + " " + probe)
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
                if extradebug:
                    for wf in feats["words"]:
                        print(wf,
                              space.languagemodel.frequencyweight(wf),
                              space.similarity(space.indexspace[wf], vectorrepositoryidx[mc]),
                              space.similarity(space.indexspace[wf], vectorrepositorysem[mc]),
                              sep="\t")
                    wds = word_tokenize(sentencerepository[mc])
                    for wd in wds:
                        print("\t\t\t",wd,
                              space.languagemodel.frequencyweight(wd),
                              space.similarity(space.indexspace[wd], vectorrepositoryidx[mc]),
                              space.similarity(space.indexspace[wd], vectorrepositorysem[mc]))
            closestneighbours = sorted(neighboursBySeq, key=lambda k: neighboursBySeq[k], reverse=True)[:antal]
            print("---- seq " + str(index) + " " + probe)
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
                if extradebug:
                    windowlist = seq.windows(feats["pos"])
                    print(feats["pos"])
                    for onewin in windowlist:
                        mm = seq.onesequencevector(onewin)
                        print(onewin,
                              space.similarity(mm, vectorrepositoryseq[mc]),
                              space.similarity(mm, vectorrepositorysem[mc]),
                              sep="\t")
            closestneighbours = sorted(neighboursByCxg, key=lambda k: neighboursByCxg[k], reverse=True)[:antal]
            print("---- cxg " + str(index) + " " + probe)
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
                if extradebug:
                    for wf in feats["features"]:
                        print(wf,
                              space.similarity(space.indexspace[wf], vectorrepositorycxg[mc]),
                              space.similarity(space.indexspace[wf], vectorrepositorysem[mc]),
                              sep="\t")
            closestneighbours = sorted(neighboursBySem, key=lambda k: neighboursBySem[k], reverse=True)[:antal]
            print("---- sem " + str(index) + " " + probe)
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
                if extradebug:
                    roledict = feats["roles"]
                    for role in roledict:
                        for item in roledict[role]:
                            mm = space.useoperator(space.indexspace[item], role)
                            print(role,":", item,
                                  space.similarity(mm, vectorrepositorysem[mc]),
                                  sep="\t")
            closestneighbours = sorted(neighboursByTot, key=lambda k: neighboursByTot[k], reverse=True)[:antal]
            print("---- tot " + str(index) + " " + probe)
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
                s5 = neighboursByTot[mc]
                print(kk,
                      str(s1), str(s2), str(s3), str(s4), str(s5),
                      sentencerepository[mc],
                      sep="\t")
        if seq.changed:
            seq.save()
        if space.changed:
            space.outputwordspace(outfile)

