import sparsevectors
import pickle
# Simplest possible logger, replace with any variant of your choice.
from logger import logger
from languagemodel import LanguageModel

error = True  # loglevel
debug = False  # loglevel
monitor = False  # loglevel


class SemanticSpace:
    def __init__(self, dimensionality=2000, denseness=10):
        self.indexspace = {}    # dict: string - sparse vector
        self.contextspace = {}  # dict: string - denser vector
        self.sequential = {}    # dict: string - boolean
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.permutationcollection = {}
        self.name = {}
        self.permutationcollection["nil"] = list(range(self.dimensionality))
        self.constantdenseness = 10
        self.languagemodel = LanguageModel()
        self.poswindow = 3
        self.changed = False

    def addoperator(self, item):
        self.permutationcollection[item] = sparsevectors.createpermutation(self.dimensionality)
        self.changed = True

    def isoperator(self, item):
        if item in self.permutationcollection:
            return True
        else:
            return False

    def useoperator(self, vector, operator):
        newvec = vector
        if operator:
            if not self.isoperator(operator):
                self.addoperator(operator)
            newvec = sparsevectors.permute(vector, self.permutationcollection[operator])
        return newvec

    def addconstant(self, item):
        self.changed = True
        self.additem(item,
                     sparsevectors.newrandomvector(self.dimensionality,
                                                   self.dimensionality // self.constantdenseness))

    def observe(self, word, update=True, loglevel=False):
        """

        :rtype: object
        """
        if not self.contains(word):
            self.additem(word)
            logger("'" + str(word) + "' is new and now introduced: " + str(self.indexspace[word]), loglevel)
        if update:
            self.languagemodel.observe(word)

    def additem(self, item, sequential="True", vector="dummy"):  # should normally be called from self.observe()
        if vector is "dummy":
            vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
        self.indexspace[item] = vector
        self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)
        self.languagemodel.additem(item)
        self.sequential = sequential
        self.changed = True

    def additemintoitem(self, item, otheritem, weight=1, operator=None):
        if not self.contains(item):
            self.additem(item)
        if not self.contains(otheritem):
            self.additem(otheritem)
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item],
                                                          self.useoperator(
                                                              sparsevectors.normalise(
                                                                  self.indexspace[otheritem]),
                                                              operator),
                                                          weight)
        self.changed = True

    def addintoitem(self, item, vector, weight=1):
        if not self.contains(item):
            self.additem(item)
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item],
                                                          sparsevectors.normalise(vector),
                                                          weight)
        self.changed = True

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
            self.languagemodel.removeitem(item)
            self.changed = True

    def reducewordspace(self, threshold=1):
        items = list(self.indexspace.keys())
        for item in items:
            if self.languagemodel.globalfrequency[item] <= threshold:
                self.removeitem(item)
                self.changed = True

    def comb(self):
        for item in self.contextspace:
            self.contexspace[item].comb()


    #================================================================
    # input output wordspace
    def outputwordspace(self, filename):
            try:
                with open(filename, 'wb') as outfile:
                    itemj = {}
                    itemj["dimensionality"] = self.dimensionality
                    itemj["densenss"] = self.denseness
                    itemj["poswindow"] = self.poswindow
                    itemj["constantdensenss"] = self.constantdenseness
                    itemj["sequential"] = self.sequential
                    itemj["indexspace"] = self.indexspace
                    itemj["contextspace"] = self.contextspace
                    itemj["permutationcollection"] = self.permutationcollection
                    itemj["languagemodel"] = self.languagemodel
                    pickle.dump(itemj, outfile)
            except:
                    logger("Could not write >>" + filename + ".toto <<", error)

    def inputwordspace(self, vectorfile):
        try:
            cannedspace = open(vectorfile, 'rb')
            itemj = pickle.load(cannedspace)
            self.dimensionality = itemj["dimensionality"]
            self.denseness = itemj["densenss"]
            self.poswindow = itemj["poswindow"]
            self.constantdenseness = itemj["constantdensenss"]
            self.sequential = itemj["sequential"]
            self.indexspace = itemj["indexspace"]
            self.contextspace = itemj["contextspace"]
            self.permutationcollection = itemj["permutationcollection"]
            self.languagemodel = itemj["languagemodel"]
        except:
            logger("Could not read from >>" + vectorfile + "<<", error)

    # ===========================================================================
    # querying the semantic space
    def contains(self, item):
        if item in self.indexspace:
            return True
        else:
            return False

    def items(self):
        return self.indexspace.keys()

    def similarity(self, vector, anothervector):
        return sparsevectors.sparsecosine(vector, anothervector)

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

