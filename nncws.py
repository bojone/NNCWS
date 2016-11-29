import numpy as np
import json
from collections import defaultdict

class NNCWS:
    
    def __init__(self):
        self.nb_gram = 7
        self.nncws_weights = np.load('nncws_weights.npy')
        self.word2id = json.load(open('word2id.json'))
        self.word2id = defaultdict(lambda: len(self.word2id)+1, self.word2id)
        self.trans_proba = {'ss':1, 'sb':1, 'bm':1, 'be':1, 'mm':1, 'me':1, 'es':1, 'eb':1}
        self.trans_proba = {i:np.log(j) for i,j in self.trans_proba.iteritems()}
    
    def core_predict(self, text):
        text_vectors = np.array([[self.nncws_weights[0][j] for j in i] for i in text])
        flatten = text_vectors.reshape((text_vectors.shape[0], text_vectors.shape[1]*text_vectors.shape[2]))
        dense = np.dot(flatten, self.nncws_weights[1]) + self.nncws_weights[2]
        relu = (dense > 0)*dense
        dense = np.dot(relu, self.nncws_weights[3])
        softmax = np.exp(dense)
        softmax_sum = softmax.sum(axis=1)
        softmax_sum = np.repeat(softmax_sum, 4).reshape((softmax_sum.shape[0], 4))
        softmax = softmax/softmax_sum
        return softmax
    
    def viterbi(self, nodes):
        paths = nodes[0]
        for l in range(1, len(nodes)):
            paths_ = paths.copy()
            paths = {}
            for i in nodes[l]:
                nows = {}
                for j in paths_:
                    if j[-1]+i in self.trans_proba:
                        nows[j+i] = paths_[j] + nodes[l][i] + self.trans_proba[j[-1]+i]
                k = np.argmax(nows.values())
                paths[nows.keys()[k]] = nows.values()[k]
        return paths.keys()[np.argmax(paths.values())]
    
    def cut_words(self, s): 
        if s:
            text_ids = [self.word2id[i] for i in s]
            text_ids = [0]*((self.nb_gram-1)/2) + text_ids + [0]*((self.nb_gram-1)/2)
            text = [text_ids[i:i+self.nb_gram] for i in range(len(text_ids) + 1 - self.nb_gram)]
            text = np.array(text)
            result = self.core_predict(text)
            nodes = [dict(zip(('s', 'b', 'm', 'e'), np.log(i))) for i in result]
            tags = self.viterbi(nodes)
            words = []
            for i in range(len(s)):
                if tags[i] in ['s', 'b']:
                    words.append(s[i])
                else:
                    words[-1] += s[i]
            return words
        else:
            return []
