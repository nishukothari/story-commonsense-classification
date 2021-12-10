import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from bert_utils import build_model, preprocess
from sklearn.metrics import precision_recall_fscore_support

testing_file_motivation = pd.read_csv(os.path.join("scs-baselines-master/data/test/motivation", "allcharlinepairs_noids.csv"))

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

test_input, test_labels = preprocess(testing_file_motivation, bert_layer)

model = build_model(bert_layer, max_len=128)
model.load_weights('model1.h5')

test_output = model.predict(test_input)
test_output = np.rint(test_output)

print(precision_recall_fscore_support(test_labels, test_output, average='micro'))