from logger import logger
#from corenlp import CoreNLPClient
import nltk
from nltk import pos_tag
from nltk import word_tokenize
import sparsevectors
import pickle

nltk.download('averaged_perceptron_tagger')

#parser = CoreNLPClient(annotators="tokenize ssplit pos lemma depparse".split())

class SequenceLabels:
    def __init__(self, dimensionality=2000, window=3,
                 sequencelabel=None, permutations={}):
        self.window = window
        self.changed = False
        self.dimensionality = dimensionality
        if sequencelabel is None:
            self.sequencelabel = sparsevectors.newrandomvector(dimensionality, dimensionality // 10)
            self.changed = True
        else:
            self.sequencelabel = sequencelabel
        self.permutations = permutations

    def windows(self, sequence):
        windowlist = []
        if self.window > 0:
           windowlist = [sequence[ii:ii + self.window] for ii in range(len(sequence) - self.window + 1)]
        return windowlist

    def onesequencevector(self, subsequence, accumulator=None):
        if accumulator == None:
            accumulator = self.sequencelabel
        if subsequence == []:
            return accumulator
        else:
            head = subsequence[0]  # type: str
            tail = subsequence[1:]
            if not head in self.permutations:
                self.permutations[head] = sparsevectors.createpermutation(self.dimensionality)
                self.changed = True
            passitdown = sparsevectors.permute(accumulator, self.permutations[head])
            return self.onesequencevector(tail, passitdown)

    def sequencevector(self, sequence):
        windowlist = self.windows(sequence)
        v = sparsevectors.newemptyvector(self.dimensionality)
        for w in windowlist:
            v = sparsevectors.sparseadd(v, sparsevectors.normalise(self.onesequencevector(w)))
        return v

    def process(self, string):
        return self.sequencevector([t[1] for t in pos_tag(word_tokenize(string))])

    #================================================================
    # save and restore sequence model
    def save(self,  filename="/home/jussi/data/storm/vectorspace/sequencemodel.hyp"):
        with open(filename, 'wb') as outfile:
            try:
                pickle.dump(self.window, outfile)
                pickle.dump(self.dimensionality, outfile)
                pickle.dump(self.sequencelabel, outfile)
                pickle.dump(self.permutations, outfile)
            except TypeError:
                logger("Could not write to file", True)

    def restore(self, modelfilename):
        modelfile = open(modelfilename, "rb")
        try:
            self.window = pickle.load(modelfile)
            self.dimensionality = pickle.load(modelfile)
            self.sequencelabel = pickle.load(modelfile)
            self.permutations = pickle.load(modelfile)
        except:
            logger("Could not load sequence model", True)