import pandas as pd
import numpy as np
import os
import torch
import tensorflow as tf
import tensorflow_hub as hub
from utils import build_model_bert, preprocess, predict_one_hot, build_model_bert_raw, SCSDataset, build_model_cnn, runNetwork
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

testing_file_motivation = pd.read_csv(os.path.join("data/test/motivation", "allcharlinepairs_noids.csv"))
testing_file_emotion = pd.read_csv(os.path.join("data/test/emotion", "allcharlinepairs_noids.csv"))

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

test_input, test_labels = preprocess(testing_file_motivation, testing_file_emotion, bert_layer)

bert_model_raw = build_model_bert_raw(bert_layer)
cnn_dataset_m = SCSDataset(test_input['motivation'], torch.from_numpy(test_labels['maslow']), bert_model_raw)
cnn_dataset_r = SCSDataset(test_input['motivation'], torch.from_numpy(test_labels['reiss']), bert_model_raw)
cnn_dataset_p = SCSDataset(test_input['emotion'], torch.from_numpy(test_labels['plutchik']), bert_model_raw)

print("CNN Model Testing")
print("CNN: Testing Maslow")

cnn_maslow = build_model_cnn(5)
cnn_maslow.load_state_dict(torch.load('cnn_maslow.pth'))
_, precision_maslow_cnn, recall_maslow_cnn, f1_maslow_cnn = runNetwork(False, 1, cnn_maslow, cnn_dataset_m, file_extension='cnn_maslow')

print("CNN: Testing Reiss")

cnn_reiss = build_model_cnn(19)
cnn_reiss.load_state_dict(torch.load('cnn_reiss.pth'))
_, precision_reiss_cnn, recall_reiss_cnn, f1_reiss_cnn = runNetwork(False, 1, cnn_reiss, cnn_dataset_r, file_extension='cnn_reiss')

print("CNN: Testing Plutchik")

cnn_plutchik = build_model_cnn(16)
cnn_plutchik.load_state_dict(torch.load('cnn_plutchik.pth'))
_, precision_plutchik_cnn, recall_plutchik_cnn, f1_plutchik_cnn = runNetwork(False, 1, cnn_plutchik, cnn_dataset_p, file_extension='cnn_plutchik')


print("BERT Model Testing")
print("BERT: Testing Maslow")

model_m = build_model_bert(bert_layer, 5, max_len=128)
model_m.load_weights('bert_model_maslow.h5')

test_output_m = model_m.predict(test_input['motivation'])
#test_output_m = np.rint(test_output_m)
test_output_m = np.apply_along_axis(predict_one_hot, axis=1, arr=test_output_m)

np.save("labels_bert_maslow", test_labels['maslow'])
np.save("preds_bert_maslow", test_output_m)

precision_maslow_bert, recall_maslow_bert, f1_maslow_bert, _ = precision_recall_fscore_support(test_labels['maslow'], test_output_m, average='micro')

print("BERT: Testing Reiss")

model_r = build_model_bert(bert_layer, 19, max_len=128)
model_r.load_weights('bert_model_reiss.h5')

test_output_r = model_r.predict(test_input['motivation'])
#test_output_r = np.rint(test_output_r)
test_output_r = np.apply_along_axis(predict_one_hot, axis=1, arr=test_output_r)

np.save("labels_bert_reiss", test_labels['reiss'])
np.save("preds_bert_reiss", test_output_r)

precision_reiss_bert, recall_reiss_bert, f1_reiss_bert, _ = precision_recall_fscore_support(test_labels['reiss'], test_output_r, average='micro')

print("BERT: Testing Plutchik")

model_p = build_model_bert(bert_layer, 16, max_len=128)
model_p.load_weights('bert_model_plutchik.h5')

test_output_p = model_p.predict(test_input['emotion'])
#test_output_p = np.rint(test_output_p)
test_output_p = np.apply_along_axis(predict_one_hot, axis=1, arr=test_output_p)

np.save("labels_bert_plutchik", test_labels['plutchik'])
np.save("preds_bert_plutchik", test_output_p)

precision_plutchik_bert, recall_plutchik_bert, f1_plutchik_bert, _ = precision_recall_fscore_support(test_labels['plutchik'], test_output_p, average='micro')


data = [
    ["BERT", precision_maslow_bert, recall_maslow_bert, f1_maslow_bert, precision_reiss_bert, recall_reiss_bert, f1_reiss_bert, precision_plutchik_bert, recall_plutchik_bert, f1_plutchik_bert],
    ["BERT + CNN", precision_maslow_cnn, recall_maslow_cnn, f1_maslow_cnn, precision_reiss_cnn, recall_reiss_cnn, f1_reiss_cnn, precision_plutchik_cnn, recall_plutchik_cnn, f1_plutchik_cnn]
]
  

col_names = ["Model Name", "Maslow Precision", "Maslow Recall", "Maslow F1", "Reiss Precision", "Reiss Recall", "Reiss F1", "Plutchik Precision", "Plutchik Recall", "Plutchik F1"]
  
print(tabulate(data, headers=col_names))