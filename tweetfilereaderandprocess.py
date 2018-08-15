import json
import re
import os

from logger import logger
from nltk import word_tokenize, Counter, sent_tokenize, pos_tag

error = True
verbose = False
debug = False
monitor = True

presentp1verbs = Counter()
otherp1verbs = Counter()
presentotherverbs = Counter()
otherotherverbs = Counter()
presentp1adverbs = Counter()
otherp1adverbs = Counter()
presentotheradverbs = Counter()
otherotheradverbs = Counter()
p1 = Counter()
other = Counter()
hashtags = Counter()
questioncounter = Counter()
questionlocalcounter = Counter()
seen = Counter()
qpresentverbs = Counter()
qverbs = Counter()
qadverbs = Counter()

negations = ["n't", "not", "never"]

stormterms = set(["irma", "#irma", "#harvey", "harvey", "hurricane", "#hurricane", "storm", "hurricaneharvey",
                  "harvey2017", "#harvey2017", "hurricaneirma", "irma2017", "hurricaneirma2017", "hurricanes", "flood",
                  "harveystorm", "irmastorm", "hurricaineharvey", "hurricaineirma", "hurricaneharvey2017", "disaster",
                  "fema", "post-harvey", "post-irma", "superstorm", "super-storm",
                  "stormharvey", "stormirma", "harveyhurricane", "irmahurricane", "majorhurricane", "stormprep",
                  "extremeweather", "evacuation", "flashflood" "flashfloodwatch", "harveyrelief", "houston", "texas", "puertorico", "florida",
                  "#hurricaneharvey", "#hurricaneharvey2017", "#hurricaneirma", "#hurricaneirma2017"])
# explicit mention of questions or challenges: "the question is if ..." "one wonders..."

questionterms = set(["?", "question", "wonders", "wonder", "problem", "suspect", "seems"])

verbtags = ["VB", "VBN", "VBZ", "VBD", "VBP", "VBG"]

urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-\?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)



resourcedirectory = "/home/jussi/data/storm/fixed/"
filenamelist = []
pattern = re.compile(r".*irma")
for filenamecandidate in os.listdir(resourcedirectory):
    if pattern.match(filenamecandidate):
        filenamelist.append(os.path.join(resourcedirectory, filenamecandidate))



def dotweetfiles(filenamelist1, loglevel=False):
    tweetantal = 0
    questioncounter = 0
    dettatweetantal = 0
    filenamelist = sorted(filenamelist1)
    logger("Starting tweet file processing ", loglevel)
    logger(filenamelist, loglevel)
    for filename in filenamelist:
        presentp1verbs.clear()
        otherp1verbs.clear()
        otherotherverbs.clear()
        presentotherverbs.clear()
        qpresentverbs.clear()
        qverbs.clear()
        qadverbs.clear()
        presentp1adverbs.clear()
        otherp1adverbs.clear()
        otherotheradverbs.clear()
        presentotheradverbs.clear()
        dettatweetantal = 0

        questionlocalcounter = 0

        seen.clear()
        date = filename.split(".")[-5].split("/")[-1]
        with open(filename, errors="replace", encoding='utf-8') as tweetfile:
            logger("Loading " + filename, loglevel)
            try:
                data = json.load(tweetfile)
            except json.decoder.JSONDecodeError:
                logger("***" + filename, error)
                continue
            logger("Loaded", loglevel)
            for tw in data:
                try:
                    text = tw["rawText"]
                    text = urlpatternexpression.sub("URL", text)
                    text = handlepattern.sub("HANDLE", text)
                    words = word_tokenize(text.lower())
                    if set(words).isdisjoint(stormterms):
                        continue
                    if words[0] == "RT":
                        continue
 #                       logger("Discarding " + text, debug)
#                    go = False
#                    for w in word_tokenize(text.lower()):
#                        if go:
#                            hashtags[w] += 1
#                            go = False
#                        if w.startswith("#"):
#                            go = True
                    else:
                        tweetantal += 1
                        dettatweetantal += 1
                        sents = sent_tokenize(text)
                        for sentence in sents:
                            question = False
                            logger(sentence, debug)
                            words = word_tokenize(sentence)
                            poses = pos_tag(words)
 #                           sdp = semanticdependencyparse(sentence, False)
                            if seen[sentence] > 0:
                                continue
                            seen[sentence] += 1
                            if poses[0][0] == "RT":
                                continue
                            if not set(words).isdisjoint(questionterms):
                                logger("Explicit question or pondering: " + sentence, verbose)
                                questioncounter += 1
                                questionlocalcounter += 1
                                question = True
                            mesent = False
                            for item in poses:
                                if item[1] in verbtags and question:
                                    qverbs[item[0].lower()] += 1
                                if item[1] == "RB" and question:
                                    qadverbs[item[0].lower()] += 1
                                if item == ("I", "PRP"):
                                    mesent = True
                                if mesent:
                                    if item[1] == "VBP":
                                        presentp1verbs[item[0].lower()] += 1
                                    if item[1] == "VBD":
                                        otherp1verbs[item[0].lower()] += 1
                                    if item[1] == "RB":
                                        if item[0].lower() in negations:
                                            p1["negation"] += 1
                                        else:
                                            presentp1adverbs[item[0].lower()] += 1
                                else:
                                    if item[1] == "VBP":
                                        presentotherverbs[item[0].lower()] += 1
                                    if item[1] == "VBD":
                                        otherotherverbs[item[0].lower()] += 1
                                    if item[1] == "RB":
                                        if item[0].lower() in negations:
                                            other["negation"] += 1
                                        else:
                                            presentotheradverbs[item[0].lower()] += 1
                            if mesent:
                                p1["sentencecount"] += 1
                            else:
                                other["sentencecount"] += 1
                except KeyError:
                    logger("****" + str(tw), error)
#        print("hashes", hashtags)
        top = 10
        print(tweetantal, dettatweetantal)
        print(questioncounter, questionlocalcounter)
        print("p1, present", presentp1verbs.most_common(top))
        print("pX, present", presentotherverbs.most_common(top))
        print("p1, past", otherp1verbs.most_common(top))
        print("pX, past", otherotherverbs.most_common(top))
        print("Q, verbs", qverbs.most_common(top))
        print("p1, adv", presentp1adverbs.most_common(top))
        print("pX, adv", presentotheradverbs.most_common(top))
        print("Q, adv", qadverbs.most_common(top))
        print("p1neg", p1)
        print("otherneg", other)


def counttweetfiles(filenamelist1, loglevel=False):
    tweetantal = 0
    sentenceantal = 0
    questionantal = 0
    skedetweetantal = 0
    skedesentenceantal = 0
    skedequestionantal = 0
    nuvarandeskede = 0
    filenamelist = sorted(filenamelist1)
    skede = {}
    skede["irma"] = ["2017-09-06","2017-09-09"]
    skede["harvey"] = ["2017-08-25","2017-08-27"]
    skede["hurricane"] = ["2017-08-25","2017-09-09"]
    logger("Starting tweet file processing ", loglevel)
    logger(filenamelist, loglevel)
    for filename in filenamelist:
        dettatweetantal = 0
        dettasentenceantal = 0
        dettaquestionantal = 0
        seen.clear()
        date = filename.split(".")[-5].split("/")[-1]
        storm = filename.split(".")[-1]
        skedelista = skede[storm]
        if date > skedelista[nuvarandeskede]:
            nuvarandeskede += 1
            skedetweetantal = 0
            skedesentenceantal = 0
            skedequestionantal = 0
        logger(date, loglevel)
        with open(filename, errors="replace", encoding='utf-8') as tweetfile:
            logger("Loading " + filename, loglevel)
            try:
                data = json.load(tweetfile)
                logger("Loaded", loglevel)
            except json.decoder.JSONDecodeError:
                logger("***" + filename, error)
                continue
            for tw in data:
                try:
                    text = tw["rawText"]
                    text = urlpatternexpression.sub("URL", text)
                    words = word_tokenize(text.lower())
                    if set(words).isdisjoint(stormterms):
                        continue
                    if words[0] == "RT":
                        continue

                    else:
                        tweetantal += 1
                        dettatweetantal += 1
                        skedetweetantal += 1
                        sents = sent_tokenize(text)
                        for sentence in sents:
                            sentenceantal += 1
                            dettasentenceantal += 1
                            skedesentenceantal += 1
                            question = False
                            words = word_tokenize(sentence)
                            if seen[sentence] > 0:
                                continue
                            seen[sentence] += 1
                            if not set(words).isdisjoint(questionterms):
                                questionantal += 1
                                dettaquestionantal += 1
                                skedequestionantal += 1
                except KeyError:
                    logger("****" + str(tw), loglevel)
        print(date, nuvarandeskede, dettatweetantal, dettasentenceantal, dettaquestionantal, tweetantal, sentenceantal,
              questionantal, skedetweetantal, skedesentenceantal, skedequestionantal, filename, sep="\t")


counttweetfiles(filenamelist, False)