import glob
import re
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import json

txt_names = glob.glob('./*/*.txt')

pure_texts = []
pure_tags = []
for name in tqdm(iter(txt_names)):
    txt = open(name).read().decode('utf-8', 'ignore').strip('\n')
    for text in txt.split('\n'):
        pure_texts.append(u'')
        pure_tags.append(u'')
        last_start = 0
        for i in re.finditer('/[a-z1-9]+ *', text):
            s = text[last_start:i.start()]
            pure_texts[-1] += s
            if len(s) > 1:
                pure_tags[-1] += 'b' + 'm'*(len(s)-2) + 'e'
            elif len(s) == 1:
                pure_tags[-1] += 's'
            last_start = i.end()

min_count = 2
word_count = Counter(''.join(pure_texts))
samples_per_epoch = sum(word_count.values())
word_count = Counter({i:j for i,j in word_count.iteritems() if j >= min_count})
word2id = defaultdict(lambda: len(word_count)+1)
id_here = 0
for i in word_count.most_common():
    id_here += 1
    word2id[i[0]] = id_here

json.dump(word2id, open('word2id.json', 'w'))
tag2id = {'s':0, 'b':1, 'm':2, 'e':3}
nb_word = len(word2id) + 2
nb_gram = 7 #n-grams模型的n，需要是奇数
word_size = 64


from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model
from keras.utils import np_utils

def data_generator(batch_size):
    while True:
        x = []
        y = []
        for i in range(len(pure_texts)):
            text_ids = [word2id[j] for j in pure_texts[i]]
            tag_ids = [tag2id[j] for j in pure_tags[i]]
            text_ids = [0]*((nb_gram-1)/2) + text_ids + [0]*((nb_gram-1)/2)
            for j in range(len(tag_ids)):
                x.append(text_ids[j: j+nb_gram])
                y.append([tag_ids[j]])
                if len(x) == batch_size:
                    yield np.array(x), np_utils.to_categorical(y, 4)
                    x = []
                    y = []
        if x:
            yield np.array(x), np_utils.to_categorical(y, 4)


ngram = Input(shape=(nb_gram,), dtype='int32')
embedded = Embedding(nb_word, word_size)(ngram)
embedded = Flatten()(embedded)
dense = Dense(64, activation='relu')(embedded)
dense = Dense(4, activation='softmax')(dense)

model = Model(input=ngram, output=dense)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 4096
nb_epoch = 30
model.fit_generator(data_generator(batch_size), samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch)
np.save('nncw_weights', model.get_weights())