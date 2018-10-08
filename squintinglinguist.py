from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import re
from lexicalfeatures import lexicon
from logger import logger
from corenlp import CoreNLPClient
import semanticroles

urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-\?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)
verbtags = ["VB", "VBZ", "VBP", "VBN", "VBD", "VBG"]
adjectivetags = ["JJ", "JJR", "JJS"]
def words(text):
    words = word_tokenize(text)
    return words


def generalise(text, handlesandurls=True, nouns=True, verbs=True, adjectives=True, adverbs=False):
    accumulator = []
    if handlesandurls:
        text = urlpatternexpression.sub("U", text)
        text = handlepattern.sub("H", text)
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        poses = pos_tag(words)
        for item in poses:
            if nouns and item[1] == "NN":
                accumulator.append("N")
            elif nouns and item[1] == "NNS":
                accumulator.append("Ns")
            elif adjectives and item[1] in adjectivetags:
                accumulator.append(item[1])
            elif verbs and item[1] in verbtags:
                tag = item[1]
                if tag == "VBZ":
                    tag = "VBP"  #  neutralise for 3d present -- VBP is present
                accumulator.append(tag)
            elif adverbs and item[1] == "RB":
                accumulator.append("R")
            else:
                accumulator.append(item[0])
    return " ".join(accumulator)


# do  MD (modal) (separate out 'not' from RB)

def featurise_sentence(sentence, loglevel=False):
    features = []
    words = word_tokenize(sentence)
    for word in words:
        for feature in lexicon:
            if word.lower() in lexicon[feature]:
                features.append("JiK" + feature)
    logger(sentence + "->" + str(features), loglevel)
    return features

def tokenise(text, loglevel=False):
    return word_tokenize(text)

def window(text, window=2, direction=True):
    return False

def featurise(text, loglevel=False):
    features = []
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        for word in words:
            for feature in lexicon:
                if word.lower() in lexicon[feature]:
                    features.append("JiK" + feature)
        parsedfeatures = semanticroles.semanticdependencyparse(text)
        basicfeatures = parsedfeatures["features"]
        features += basicfeatures
    logger(text + "->" + str(features), loglevel)
    return features


def mildpositems(string, full=False):
    leaveintags = ["IN", "DT", "MD", "PRP", "PRP$", "POS", "CC", "EX", "PDT", "RP", "TO", "WP", "WP$", "WDT", "WRB"]
    words = word_tokenize(string)
    poses = pos_tag(words)
    if not full:
        returnposes = [("START", "START")]
        for p in poses:
            if p[1] in leaveintags:
                returnposes.append((p[1],p[0]))
            else:
                returnposes.append(p)
        returnposes.append(("END","END"))
    else:
        returnposes = [("START", "BEG")] + poses + [("END", "END")]
    return returnposes

