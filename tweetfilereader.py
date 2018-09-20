import json
import re
import os

from logger import logger
from nltk import word_tokenize, Counter, sent_tokenize, pos_tag

error = True
verbose = False
debug = False
monitor = True


stormterms = set(["irma", "#irma", "#harvey", "harvey", "hurricane", "#hurricane", "storm", "hurricaneharvey",
                  "harvey2017", "#harvey2017", "hurricaneirma", "irma2017", "hurricaneirma2017", "hurricanes", "flood",
                  "harveystorm", "irmastorm", "hurricaineharvey", "hurricaineirma", "hurricaneharvey2017", "disaster",
                  "fema", "post-harvey", "post-irma", "superstorm", "super-storm",
                  "stormharvey", "stormirma", "harveyhurricane", "irmahurricane", "majorhurricane", "stormprep",
                  "extremeweather", "evacuation", "flashflood" "flashfloodwatch", "harveyrelief", "houston", "texas",
                  "puertorico", "florida",
                  "#hurricaneharvey", "#hurricaneharvey2017", "#hurricaneirma", "#hurricaneirma2017"])


urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-\?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)


def getfilelist(resourcedirectory="/home/jussi/data/storm/fixed", pattern=re.compile(r".*irma")):
    filenamelist = []
    for filenamecandidate in os.listdir(resourcedirectory):
        if pattern.match(filenamecandidate):
            logger(filenamecandidate, debug)
            filenamelist.append(os.path.join(resourcedirectory,filenamecandidate))
    logger(filenamelist, debug)
    return sorted(filenamelist)


def dotweetfiles(filenamelist1, loglevel=False):
    tweetantal = 0
    sentencelist = []
    filenamelist = sorted(filenamelist1)
    logger("Starting tweet file processing ", loglevel)
    logger(str(filenamelist), loglevel)
    for filename in filenamelist:
        sl = doonetweetfile(filename, loglevel)
        sentencelist = sentencelist + sl
        tweetantal += len(sl)

def doonetweetfile(filename, loglevel=False):
    logger(filename, loglevel)
    date = filename.split(".")[-5].split("/")[-1]
    sentencelist = []
    with open(filename, errors="replace", encoding='utf-8') as tweetfile:
        logger("Loading " + filename, loglevel)
        try:
            data = json.load(tweetfile)
        except json.decoder.JSONDecodeError:
            logger("***" + filename, error)
            data = []
        logger("Loaded", loglevel)
        for tw in data:
            try:
                text = tw["rawText"]
                text = urlpatternexpression.sub("URL", text)
                text = handlepattern.sub("HANDLE", text)
                words = word_tokenize(text.lower())
                if set(words).isdisjoint(stormterms):
                    continue
                if words[0] == "rt":
                    continue
                else:
                    sents = sent_tokenize(text)
                    for sentence in sents:
                        question = False
                        logger(sentence, debug)
                        words = word_tokenize(sentence)
                        sentencelist = sentencelist + sents
            except KeyError:
                if str(tw) != "{}":  # never mind empty strings, no cause for alarm
                    logger("**** " + str(tw) + " " + str(len(sentencelist)), error)
    return sentencelist
