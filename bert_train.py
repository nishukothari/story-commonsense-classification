import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tokens import FullTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

def string_to_list(input_string: str) -> list:
    res = input_string.strip('][')
    res = res.replace('"', '')
    res = res.split(', ')
    
    return res


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(5, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


training_file_motivation = pd.read_csv(os.path.join("scs-baselines-master/data/dev/motivation", "allcharlinepairs_noids.csv"))
training_file_emotion = pd.read_csv(os.path.join("scs-baselines-master/data/dev/emotion", "allcharlinepairs_noids.csv"))
training_file_motivation = training_file_motivation[training_file_motivation['action'] != 'no']
training_file_motivation = training_file_motivation[training_file_motivation['motivation'] != 'none']
training_file_motivation = training_file_motivation[training_file_motivation['maslow'] != '[]']
training_file_motivation = training_file_motivation[training_file_motivation['reiss'] != '[]']
training_file_motivation = training_file_motivation[training_file_motivation['reiss'] != '["na"]']
training_file_motivation["maslow"] = training_file_motivation["maslow"].apply(string_to_list)
training_file_motivation.reset_index(drop=True, inplace=True)


module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(training_file_motivation["maslow"])
train_input = bert_encode(training_file_motivation['sentence'].values, tokenizer)

model = build_model(bert_layer, max_len=512)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history = model.fit(
    train_input, train_labels, 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint, earlystopping],
    batch_size=32,
    verbose=1
)