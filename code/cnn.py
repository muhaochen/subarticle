
from article_to_tensor import article2tensor

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, Merge, BatchNormalization, merge
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D

from keras.optimizers import Adagrad

import numpy as np

from keras.layers import Input
from numpy import linalg as LA
import scipy

def get_longest_token_embed(a2t, title):
    tokens = a2t.tokenize_article(title)
    rst = a2t.embed_article(tokens[0])[0]
    num = -1
    for x in tokens:
        if (x.count('_') > num):
            try_embed = a2t.embed_article(x)[0]
            if try_embed[0] != 0. or try_embed[1] != 0.:
                num = x.count('_')
                rst = try_embed
    if rst[0] == rst[1] == 0.:
        max_len = 0.
        tokens = title.replace('(','').replace(')','').replace(',','').split(' ')
        for x in tokens:
            if (len(x) > max_len):
                try_embed = a2t.embed_article(x)[0]
                if try_embed[0] != 0. or try_embed[1] != 0.:
                    max_len = len(x)
                    rst = try_embed
    return rst

a2t = article2tensor()
a2t.initialize('wiki_tokenized_vec50.bin', 'entity_list.txt')

max_data = -1
limit_data = max_data > 0
raw_data = []
skip_head = True
x = None
count = 0
for line in open('../dataset/WAP_15to200_simplified.tsv'):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split('\t')
    if len(line) != 16:
        x = line
        continue
    raw_data.append(line)
    if limit_data:
        count += 1
        if count >= max_data:
            break
print x

print len(raw_data)

len_m_article = np.array([len(a2t.tokenize_article(line[1])) for line in raw_data])
len_m_title = np.array([len(line[0].split(' ')) for line in raw_data])
avg_m_article = int(np.average(len_m_article)) + 1
avg_m_title = int(np.average(len_m_title)) + 1
max_m_article = max(len_m_article)
max_m_title = max(len_m_title)
print avg_m_article, max_m_article
print avg_m_title, max_m_title

len_s_article = np.array([len(a2t.tokenize_article(line[5])) for line in raw_data])
len_s_title = np.array([len(line[4].split(' ')) for line in raw_data])
avg_s_article = int(np.average(len_s_article)) + 1
avg_s_title = int(np.average(len_s_title)) + 1
max_s_article = max(len_s_article)
max_s_title = max(len_s_title)
print avg_s_article, max_s_article
print avg_s_title, max_s_title

dim = a2t.word2vec.vector_size
article_size = 100
title_size = 14

article_tensor1 = np.array([a2t.embed_article_normalized(line[1], article_size, zero_padding=True) for line in raw_data])
title_tensor1 = np.array([a2t.embed_article_normalized(line[0], title_size, tokenize=False, zero_padding=True) for line in raw_data])
article_tensor2 = np.array([a2t.embed_article_normalized(line[5], article_size, zero_padding=True) for line in raw_data])
title_tensor2 = np.array([a2t.embed_article_normalized(line[4], title_size, tokenize=False, zero_padding=True) for line in raw_data])

title_embed1 = [get_longest_token_embed(a2t, line[0]) for line in raw_data]
title_embed2 = [get_longest_token_embed(a2t, line[4]) for line in raw_data]

title_dist = []
for i in range(len(title_embed1)):
    if title_embed1[i][0] == 0. or title_embed2[i][0] == 0.:
        title_dist.append(1.)
    else:
        title_dist.append(scipy.spatial.distance.cosine(title_embed1[i], title_embed2[i]))
title_dist = np.array(title_dist)

#nums = [[float(x[2]), float(x[3]), float(x[6]), float(x[7]), float(x[9]), float(x[10]), float(x[11]), float(x[12]), float(x[13]), float(x[14]), float(x[15])] for x in raw_data]
nums = [[float(x[9]), float(x[11]), float(x[12]), float(x[13]), float(x[14]), float(x[15])] for x in raw_data]
#nums = [[float(x[10]), float(x[11]), float(x[12]), float(x[13])] for x in raw_data]
for i in range(len(nums)):
    nums[i].append(title_dist[i])
nums = np.array(nums)

nums_max = np.amax(nums, axis=0)
nan_ap = False
print nums_max
for i in range(len(nums)):
    for j in range(len(nums[i])):
        nums[i][j] /= nums_max[j]
        #clear nan.
        if np.isnan(nums[i][j]):
            nums[i][j] = 0.
            nan_ap = True
if nan_ap:
    print "NAN detected."



nums = np.reshape(nums, (-1, 7, 1))

rating = np.array([int(x[8]) for x in raw_data])
class_label20 = np.zeros((len(rating), 2))
for i in range(len(rating)):
    class_label20[i][rating[i]] = 1.

def build_model():
    title_input1 = Input(shape=(title_size, dim), name='title1')
    x=Conv1D(filters=title_size/2, kernel_size=3)(title_input1)
    x=MaxPooling1D(2)(x)
    x=Conv1D(filters=title_size/6, kernel_size=3)(x)
    midpoint1=Flatten()(x)
    title_input2 = Input(shape=(title_size, dim), name='title2')
    x=Conv1D(filters=title_size/2, kernel_size=3)(title_input2)
    x=MaxPooling1D(2)(x)
    x=Conv1D(filters=title_size/6, kernel_size=3)(x)
    midpoint2=Flatten()(x)
    article_input1 = Input(shape=(article_size, dim), name='article1')
    x=Conv1D(filters=article_size/2, kernel_size=3)(article_input1)
    x=MaxPooling1D(2)(x)
    x=Conv1D(filters=article_size/6, kernel_size=3)(x)
    midpoint3=Flatten()(x)
    article_input2 = Input(shape=(article_size, dim), name='article2')
    x=Conv1D(filters=article_size/2, kernel_size=3)(article_input2)
    x=MaxPooling1D(2)(x)
    x=Conv1D(filters=article_size/6, kernel_size=3)(x)
    midpoint4=Flatten()(x)
    merge_text = concatenate([midpoint1,midpoint2,midpoint3,midpoint4])
    merge_text = Dense(9, activation='relu')(merge_text)
    merge_text = Dense(3, activation='relu')(merge_text)
    num_input = Input(shape=(7, 1), name='numericals')
    merge_num = Dense(8, activation='relu')(num_input)
    merge_num = Flatten()(merge_num)
    merge_all =  concatenate([merge_text, merge_num])
    x = Dense(4, activation='relu')(merge_all)
    main_output = Dense(2, activation='softmax')(x)
    merge_model = Model(inputs=[title_input1, article_input1, title_input2, article_input2, num_input], outputs=[main_output])
    return merge_model

batch_size1 = 50
adagrad = Adagrad(lr=0.01, decay=0.0)

from sklearn.model_selection import KFold, ShuffleSplit
kf = ShuffleSplit(n_splits=10)
tries = 10
cur = 0
recalls = []
accuracy = []
total = []
total_truth = []
train_test = []
for train, test in kf.split(class_label20):
    if sum([x[1] for x in class_label20[test]]) > len(class_label20[test]) * 0.8 or sum([x[1] for x in class_label20[test]]) < len(class_label20[test]) * 0.04:
        print sum([x[0] for x in class_label20[test]]), len(class_label20[test])
        continue
    train_test.append((train, test))
    cur += 1
    if cur >= tries:
        break

print len(train_test)

for train, test in train_test:
    merge_model = None
    merge_model = build_model()
    merge_model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    merge_model.fit([title_tensor1[train], article_tensor1[train], title_tensor2[train], article_tensor2[train], nums[train]], class_label20[train], batch_size=batch_size1, epochs=25)
    result1 = merge_model.evaluate([title_tensor1[test], article_tensor1[test], title_tensor2[test], article_tensor2[test], nums[test]], class_label20[test])
    pred = merge_model.predict([title_tensor1[test], article_tensor1[test], title_tensor2[test], article_tensor2[test], nums[test]])
    hit = np.zeros(2)
    gt = np.zeros(2)
    cases = np.zeros(2)
    recall = 0.
    for i in range(len(class_label20[test])):
        truth = 0
        guess = 0
        if class_label20[test][i][1] > 0.:
            truth = 1
        if pred[i][1] > pred[i][0]:
            guess = 1
        cases[guess] += 1
        gt[truth] += 1
        if (guess == truth):
            hit[guess] += 1
            if truth == 1:
                recall += 1.
    for i in range(len(hit)):
        if cases[i] > 0:
            hit[i] /= cases[i]
    #if hit[1] == 0:
        #continue
    if gt[1] > 0:
        recall /= gt[1]
    print hit
    accuracy.append(hit)
    total.append(cases)
    recalls.append(recall)
    total_truth.append(gt)

print accuracy

print np.mean(accuracy, axis=0)

total_acc = np.zeros(len(accuracy[0]))
for i in range(len(accuracy)):
    for j in range(len(accuracy[i])):
        if not np.isnan(accuracy[i][j]):
            total_acc[j] += accuracy[i][j] * total[i][j]
print np.divide(total_acc, sum(total))
print sum(total_acc) / sum(sum(total))
total_rec = 0.
for i in range(len(recalls)):
    total_rec += recalls[i] * total_truth[i][1]
total_rec /= sum([x[1] for x in total_truth])
print total_rec


