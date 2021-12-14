import pandas as pd
import os
import tensorflow as tf
import tensorflow_hub as hub
from utils import build_model, preprocess
import pickle

training_file_motivation = pd.read_csv(os.path.join("data/dev/motivation", "allcharlinepairs_noids.csv"))
training_file_emotion = pd.read_csv(os.path.join("data/dev/emotion", "allcharlinepairs_noids.csv"))

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

train_input, train_labels = preprocess(training_file_motivation, training_file_emotion, bert_layer)

print("BERT: Training Maslow")

model_m = build_model(bert_layer, 6, max_len=128)
checkpoint_m = tf.keras.callbacks.ModelCheckpoint('bert_model_maslow.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping_m = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history_m = model_m.fit(
    train_input['motivation'], train_labels['maslow'], 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint_m, earlystopping_m],
    batch_size=32,
    verbose=1
)

with open('lossMaslow', 'wb') as file_pi:
        pickle.dump(train_history_m.history, file_pi)

print("BERT: Training Reiss")

model_r = build_model(bert_layer, 20, max_len=128)
checkpoint_r = tf.keras.callbacks.ModelCheckpoint('bert_model_reiss.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping_r = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history_r = model_r.fit(
    train_input['motivation'], train_labels['reiss'], 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint_r, earlystopping_r],
    batch_size=32,
    verbose=1
)

print("BERT: Training Plutchik")

model_p = build_model(bert_layer, 17, max_len=128)
checkpoint_p = tf.keras.callbacks.ModelCheckpoint('bert_model_plutchik.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping_p = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history_p = model_p.fit(
    train_input['emotion'], train_labels['plutchik'], 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint_p, earlystopping_p],
    batch_size=32,
    verbose=1
)
