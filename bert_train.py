import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from tokens import FullTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from bert_utils import build_model, preprocess

training_file_motivation = pd.read_csv(os.path.join("scs-baselines-master/data/dev/motivation", "allcharlinepairs_noids.csv"))

print("enough memory for pandas")

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

train_input, train_labels = preprocess(training_file_motivation, bert_layer)

model = build_model(bert_layer, max_len=128)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model1.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history = model.fit(
    train_input, train_labels, 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint, earlystopping],
    batch_size=16,
    verbose=1
)
