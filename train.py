import pandas as pd
import os
import tensorflow as tf
import tensorflow_hub as hub
import torch
from utils import CNN_Text, SCSDataset, build_model_bert, build_model_bert_raw, build_model_cnn, preprocess, runNetwork
import pickle

print("Data Preprocessing")

training_file_motivation = pd.read_csv(os.path.join("data/dev/motivation", "allcharlinepairs_noids.csv"))
training_file_emotion = pd.read_csv(os.path.join("data/dev/emotion", "allcharlinepairs_noids.csv"))

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

train_input, train_labels = preprocess(training_file_motivation, training_file_emotion, bert_layer)

bert_model_raw = build_model_bert_raw(bert_layer)
cnn_dataset_m = SCSDataset(train_input['motivation'], torch.from_numpy(train_labels['maslow']), bert_model_raw)
cnn_dataset_r = SCSDataset(train_input['motivation'], torch.from_numpy(train_labels['reiss']), bert_model_raw)
cnn_dataset_p = SCSDataset(train_input['emotion'], torch.from_numpy(train_labels['plutchik']), bert_model_raw)

print("CNN Model Training")
print("CNN: Training Maslow")

cnn_maslow = build_model_cnn(6)
runNetwork(True, 3, cnn_maslow, cnn_dataset_m)
torch.save(cnn_maslow.state_dict(), 'cnn_maslow.pth')

print("CNN: Training Reiss")

cnn_reiss = build_model_cnn(20)
runNetwork(True, 3, cnn_reiss, cnn_dataset_r)
torch.save(cnn_reiss.state_dict(), 'cnn_reiss.pth')

print("CNN: Training Plutchik")

cnn_plutchik = build_model_cnn(17)
runNetwork(True, 3, cnn_plutchik, cnn_dataset_p)
torch.save(cnn_plutchik.state_dict(), 'cnn_plutchik.pth')

print("BERT Model Training")
print("BERT: Training Maslow")

bert_model_m = build_model_bert(bert_layer, 6, max_len=128)
checkpoint_m = tf.keras.callbacks.ModelCheckpoint('bert_model_maslow.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping_m = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history_m = bert_model_m.fit(
    train_input['motivation'], train_labels['maslow'], 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint_m, earlystopping_m],
    batch_size=32,
    verbose=1
)

with open('lossMaslow', 'wb') as bert_file_maslow:
        pickle.dump(train_history_m.history, bert_file_maslow)

print("BERT: Training Reiss")

bert_model_r = build_model_bert(bert_layer, 20, max_len=128)
checkpoint_r = tf.keras.callbacks.ModelCheckpoint('bert_model_reiss.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping_r = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history_r = bert_model_r.fit(
    train_input['motivation'], train_labels['reiss'], 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint_r, earlystopping_r],
    batch_size=32,
    verbose=1
)

with open('lossReiss', 'wb') as bert_file_reiss:
        pickle.dump(train_history_r.history, bert_file_reiss)

print("BERT: Training Plutchik")

bert_model_p = build_model_bert(bert_layer, 17, max_len=128)
checkpoint_p = tf.keras.callbacks.ModelCheckpoint('bert_model_plutchik.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping_p = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history_p = bert_model_p.fit(
    train_input['emotion'], train_labels['plutchik'], 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint_p, earlystopping_p],
    batch_size=32,
    verbose=1
)

with open('lossPlutchik', 'wb') as bert_file_plutchik:
        pickle.dump(train_history_p.history, bert_file_plutchik)