import sparsevectors
import math
import pickle

# Simplest possible logger, replace with any variant of your choice.
from logger import logger

error = True  # loglevel
debug = False  # loglevel
monitor = False  # loglevel


class SemanticSpace:
    def __init__(self, dimensionality=2000, denseness=10):
        self.indexspace = {}
        self.contextspace = {}
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.permutationcollection = {}
        self.name = {}
        self.permutationcollection["nil"] = list(range(self.dimensionality))
        self.constantdenseness = 10
        self.languagemodel = LanguageModel()

    def addoperator(self, item):
        self.permutationcollection[item] = sparsevectors.createpermutation(self.dimensionality)

    def addconstant(self, item):
        self.additem(item,
                     sparsevectors.newrandomvector(self.dimensionality,
                                                   self.dimensionality // self.constantdenseness))

    def observe(self, word, loglevel=False):
        if not self.contains(word):
            self.additem(word)
            logger(str(word) + " is new and now introduced: " + str(self.indexspace[word]), loglevel)
        self.languagemodel.observe(word)

    def additem(self, item, vector="dummy"):  # should normally be called from self.observe()
        if vector is "dummy":
            vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
        self.indexspace[item] = vector
        self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)
        self.languagemodel.additem(item)

    def addintoitem(self, item, vector, weight=1):
        if not self.contains(item):
            self.additem(item)
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item],
                                                          sparsevectors.normalise(vector),
                                                          weight)

    def observecollocation(self, item, otheritem, operator="nil"):
        if not self.contains(item):
            self.additem(item)
        if not self.contains(otheritem):
            self.additem(otheritem)
        self.addintoitem(item, sparsevectors.normalise(self.indexspace[otheritem]))
        self.addintoitem(otheritem, sparsevectors.normalise(self.indexspace[item]))

    def removeitem(self, item):
        if self.contains(item):
            del self.indexspace[item]
            del self.contextspace[item]
            self.languagemodel.removeitem[item]

    def reducewordspace(self, threshold=1):
        items = list(self.indexspace.keys())
        for item in items:
            if self.languagemodel.globalfrequency[item] <= threshold:
                self.removeitem(item)

    #================================================================
    # input output wordspace
    def outputwordspace(self, filename):
        with open(filename, 'wb') as outfile:
            for item in self.indexspace:
                try:
                    itemj = {}
                    itemj["string"] = str(item)
                    itemj["indexvector"] = self.indexspace[item]
                    itemj["contextvector"] = self.contextspace[item]
                    itemj["frequency"] = self.languagemodel.globalfrequency[item]
                    pickle.dump(itemj, outfile)
                except TypeError:
                    logger("Could not write >>" + item + "<<", error)

    def inputwordspace(self, vectorfile):
        cannedindexvectors = open(vectorfile, "rb")
        goingalong = True
        n = 0
        m = 0
        while goingalong:
            try:
                itemj = pickle.load(cannedindexvectors)
                item = itemj["string"]
                indexvector = itemj["indexvector"]
                if not self.contains(item):
                    self.additem(item, indexvector)
                    n += 1
                else:
                    self.indexspace[item] = indexvector
                    m += 1
                self.languagemodel.globalfrequency[item] = itemj["frequency"]
                self.languagemodel.bign += itemj["frequency"]  # oops should subtract previous value if any!
                self.contextspace[item] = itemj["contextvector"]
            except EOFError:
                goingalong = False
        return n, m

    # ===========================================================================
    # querying the semantic space
    def contains(self, item):
        if item in self.indexspace:
            return True
        else:
            return False

    def items(self):
        return self.indexspace.keys()

    def similarity(self, item, anotheritem):
        return self.contextsimilarity(item, anotheritem)

    def contextsimilarity(self, item, anotheritem):
        return sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[anotheritem])

    def indextocontextsimilarity(self, item, anotheritem):
        if self.contains(item):
            return sparsevectors.sparsecosine(self.indexspace[item], self.contextspace[anotheritem])
        else:
            return 0.0

    def contextneighbours(self, item, number=10, weights=False):
        n = {}
        for i in self.contextspace:
            n[i] = sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[i])
        if weights:
            r = sorted(n.items(), key=lambda k: n[k[0]], reverse=True)[:number]
        else:
            r = sorted(n, key=lambda k: n[k], reverse=True)[:number]
        return r

    def contexttoindexneighbours(self, item, number=10, weights=False, permutationname="nil"):
        permutation = self.permutationcollection[permutationname]
        n = {}
        for i in self.indexspace:
            n[i] = sparsevectors.sparsecosine(self.contextspace[item],
                   sparsevectors.permute(self.indexspace[i], permutation))
        if weights:
            r = sorted(n.items(), key=lambda k: n[k[0]], reverse=True)[:number]
        else:
            r = sorted(n, key=lambda k: n[k], reverse=True)[:number]
        return r

    def indextocontextneighbours(self, item, number=10, weights=False, permutationname="nil"):
        permutation = self.permutationcollection[permutationname]
        n = {}
        for i in self.contextspace:
            n[i] = sparsevectors.sparsecosine(sparsevectors.permute(self.indexspace[item], permutation),
                                              self.contextspace[i])
        if weights:
            r = sorted(n.items(), key=lambda k: n[k[0]], reverse=True)[:number]
        else:
            r = sorted(n, key=lambda k: n[k], reverse=True)[:number]
        return r
    # ===========================================================================
    # creating vectors for utterances
    # 1) sequential set of features
    # 2) bag of features
    # 3) weights are optional for each
    # 4) updating vectors to do with features is optional for each case
    # 5) that update might need to be weighted differently from the weight of some feature on the utterance
    def utterancevector(self, id, items, initialvector="nil", sequence=False,
                        weights=False, update=False, updateweights=False, loglevel=False):
        self.additem(id)
        if initialvector == "nil":
            initialvector = sparsevectors.newemptyvector(self.dimensionality)
        if sequence:
            logger("Sequence encoding not implemented yet.", monitor)
        else:
            for item in items:
                if weights:
                    weight = self.languagemodel.frequencyweight(item)
                else:
                    weight = 1
                self.observe(item)
                logger("hep", loglevel)
                logger(str(initialvector), loglevel)
                tmp = initialvector
#                self.addintoitem(id, self.indexspace[item], weight)
                initialvector = sparsevectors.sparseadd(initialvector,
                                    self.indexspace[item],
                                    weight)
                logger(item, loglevel)
                logger(str(self.indexspace[item]), loglevel)
                logger(str(initialvector), loglevel)
                logger(str(sparsevectors.sparsecosine(tmp,initialvector)),loglevel)
#            if update:
#                for item in items:
#                    for otheritem in items:
#                        if otheritem == item:
#                            continue
#                        updateweight = 1
#                        if updateweights:
#                            updateweight = self.languagemodel.frequencyweight(item)
#                            logger("Updateweights not implemented yet.", monitor)
#                        self.addintoitem(item, self.indexspace[otheritem], updateweight)
        self.contextspace[id] = initialvector
        return initialvector
    # ===========================================================================
    # language model
    # stats associated with observed items and the collection itself
    #
    # may (actually, should) be moved to another module at some point


class LanguageModel:
    def __init__(self):
        self.globalfrequency = {}
        self.bign = 0
        self.df = {}
        self.docs = 0

    def frequencyweight(self, word, streaming=False):
        try:
            if streaming:
                l = 500
                w = math.exp(-l * self.globalfrequency[word] / self.bign)
                #
                # 1 - math.atan(self.globalfrequency[word] - 1) / (0.5 * math.pi)  # ranges between 1 and 1/3
            else:
                w = math.log((self.docs) / (self.df[word] - 0.5))
        except KeyError:
            w = 0.5
        return w

    def additem(self, item):
        if not self.contains(item):
            self.globalfrequency[item] = 0

    def removeitem(self, item):
        self.bign -= self.globalfrequency[item]
        del self.globalfrequency[item]

    def observe(self, word):
        self.bign += 1
        if self.contains(word):
            self.globalfrequency[word] += 1
        else:
            self.globalfrequency[word] = 1

    def contains(self, item):
        if item in self.globalfrequency:
            return True
        else:
            return False

    def importstats(self, wordstatsfile):
        with open(wordstatsfile) as savedstats:
            i = 0
            for line in savedstats:
                i += 1
                try:
                    seqstats = line.rstrip().split("\t")
                    if not self.contains(seqstats[0]):
                        self.additem(seqstats[0])
                    self.globalfrequency[seqstats[0]] = int(seqstats[1])
                    self.bign += int(seqstats[1])
                except IndexError:
                    logger("***" + str(i) + " " + line.rstrip(), debug)
