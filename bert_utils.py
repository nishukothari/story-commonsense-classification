import numpy as np
import tensorflow as tf
from tokens import FullTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

def my_filter(row):
    for label in row["reiss"]:
        if label == 'na':
            return False

    return True

def string_to_list(input_string: str) -> list:
    res = input_string.strip('][')
    res = res.replace('"', '')
    res = res.split(', ')
    
    return res

def build_model(bert_layer, max_len=128):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(5, activation='sigmoid')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def bert_encode(texts, tokenizer, max_len=128):
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

def preprocess(file, bert_layer):
    file = file[file['action'] != 'no']
    file = file[file['motivation'] != 'none']
    file = file[file['maslow'] != '[]']
    file = file[file['reiss'] != '[]']
    file = file[file['reiss'] != '["na"]']
    file["maslow"] = file["maslow"].apply(string_to_list)
    file["reiss"] = file["reiss"].apply(string_to_list)
    file = file[file.apply(my_filter, axis=1)]
    file.reset_index(drop=True, inplace=True)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    mlb_m = MultiLabelBinarizer()
    labels_m = mlb_m.fit_transform(file["maslow"])

    mlb_r = MultiLabelBinarizer()
    labels_r = mlb_r.fit_transform(file["reiss"])

    labels = np.concatenate((labels_m, labels_r), axis=1)
    data = bert_encode(file['sentence'].values, tokenizer)

    return data, labels