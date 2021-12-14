import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from utils import build_model, preprocess, predict_one_hot
from sklearn.metrics import precision_recall_fscore_support

testing_file_motivation = pd.read_csv(os.path.join("data/test/motivation", "allcharlinepairs_noids.csv"))
testing_file_emotion = pd.read_csv(os.path.join("data/test/emotion", "allcharlinepairs_noids.csv"))

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

test_input, test_labels = preprocess(testing_file_motivation, testing_file_emotion, bert_layer)

print("BERT: Testing Maslow")

model_m = build_model(bert_layer, 6, max_len=128)
model_m.load_weights('model_maslow.h5')

test_output_m = model_m.predict(test_input['motivation'])
test_output_m = np.apply_along_axis(predict_one_hot, axis=2, arr=test_output_m)

print(precision_recall_fscore_support(test_labels['maslow'], test_output_m, average='micro'))

print("BERT: Testing Reiss")

model_r = build_model(bert_layer, 20, max_len=128)
model_r.load_weights('model_reiss.h5')

test_output_r = model_r.predict(test_input['motivation'])
test_output_r = np.apply_along_axis(predict_one_hot, axis=2, arr=test_output_r)

print(precision_recall_fscore_support(test_labels['reiss'], test_output_r, average='micro'))

print("BERT: Testing Plutchik")

model_p = build_model(bert_layer, 20, max_len=128)
model_p.load_weights('model_plutchik.h5')

test_output_p = model_r.predict(test_input['motivation'])
test_output_p = np.apply_along_axis(predict_one_hot, axis=2, arr=test_output_p)

print(precision_recall_fscore_support(test_labels['reiss'], test_output_p, average='micro'))