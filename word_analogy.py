import os, sys
import pickle
import numpy as np
from scipy import spatial as sp


model_path = './models/'

# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

filename = "word_analogy_dev.txt"
file = open(filename, "r")
output_file = open('model_predictions.txt', 'w')
ind_diff = [0]
for line in file :
    examples = line.split('||')[0].split(',')
    choices  = line.split('||')[1].split(',')
    avg_diff = 0.0
    for example in examples :
        word1 = example.split(':')[0].replace('"','')
        word2 = example.split(':')[1].replace('"','')
        word1_emb = embeddings[dictionary[word1]]
        word2_emb = embeddings[dictionary[word2]]
        ind_diff += word1_emb - word2_emb
    avg_diff = ind_diff / (len(example) * 1.0)

    min_index = -1
    max_index = -1
    min_diff = sys.float_info.max
    max_diff = sys.float_info.min

    for i in range(len(choices)) :
        output_file.write(choices[i].replace('\n','') + " ")
        word1 = choices[i].split(':')[0].replace('"','').replace('\n','')
        word2 = choices[i].split(':')[1].replace('"','').replace('\n','')
        word1_emb = embeddings[dictionary[word1]]
        word2_emb = embeddings[dictionary[word2]]

        emb_diff = word1_emb - word2_emb
        diff = 1.0 - sp.distance.cosine(avg_diff, emb_diff)
        if diff > max_diff :
            max_diff  = diff
            max_index = i
        if diff < min_diff :
            min_diff  = diff
            min_index = i

    output_file.write(choices[min_index].replace('\n','') + " ")
    output_file.write(choices[max_index].replace('\n','') + "\n")             

  