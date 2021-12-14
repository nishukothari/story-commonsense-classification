import numpy as np
import tensorflow as tf
from tokens import FullTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def string_to_list(input_string: str) -> list:
    res = input_string.strip('][')
    res = res.replace('"', '')
    res = res.split(', ')
    
    return res

def build_model_bert(bert_layer, output_dim, max_len=128):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(output_dim, activation='sigmoid')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def bert_encode(texts, tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []

    for row in texts:
        input_sequence = ["[CLS]"]
        if len(row) == 2:
            input_sequence = input_sequence + tokenizer.tokenize(row[0][0]) + tokenizer.tokenize(row[1][0]) + ["[SEP]"] + tokenizer.tokenize(row[1][0]) + ["[SEP]"]
        elif len(row) == 3:
            input_sequence = input_sequence + tokenizer.tokenize(row[0][0]) + tokenizer.tokenize(row[1][0]) + ["[SEP]"] + tokenizer.tokenize(row[2][0]) + ["[SEP]"]

        if len(input_sequence) > max_len:
            input_sequence = input_sequence[:max_len-1]
            input_sequence = input_sequence + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def combine(df):
    all_labels = df.values.sum()
    most_freq = mode(all_labels)
    if most_freq == 'na' or most_freq == '':
        most_freq = 'none'
    return [most_freq]

def combine_sentence(df):
    full_sentence = []
    full_sentence.append([df['char']])
    full_sentence.append([df['sentence']])
    
    if not pd.isnull(df['context']):
        split_context = df['context'].split("|")
        i = len(split_context) - 1
        while i >= 0:
            full_sentence.append([split_context[i]])
            i -= 1

    return full_sentence

def preprocess(file_motivation, file_emotion, bert_layer):
    file_motivation["maslow"] = file_motivation["maslow"].apply(string_to_list)
    file_motivation["reiss"] = file_motivation["reiss"].apply(string_to_list)
    file_motivation['full_sentence'] = file_motivation.apply(combine_sentence, axis=1)
    groups_motivation = file_motivation.groupby(['storyid', 'linenum', 'char']).aggregate({
        'maslow': lambda x: combine(x),
        'reiss':  lambda x: combine(x),
        'full_sentence': 'first'
    }).reset_index()

    file_emotion["plutchik"] = file_emotion["plutchik"].apply(string_to_list)
    file_emotion['full_sentence'] = file_emotion.apply(combine_sentence, axis=1)
    groups_emotion = file_emotion.groupby(['storyid', 'linenum', 'char']).aggregate({
        'plutchik': lambda x: combine(x),
        'full_sentence': 'first'
    }).reset_index()

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    mlb_m = MultiLabelBinarizer()
    mlb_m.fit([
        ["love"], 
        ["physiological"], 
        ["stability"], 
        ["esteem"], 
        ["spiritual growth"], 
        ["none"]
    ])
    labels_m = mlb_m.transform(groups_motivation["maslow"])

    mlb_r = MultiLabelBinarizer()
    mlb_r.fit([
        ["status"],
        ["idealism"],
        ["power"],
        ["family"],
        ["food"],
        ["indep"],
        ["belonging"],
        ["competition"],
        ["honor"],
        ["romance"],
        ["savings"],
        ["contact"],
        ["health"],
        ["serenity"],
        ["curiosity"],
        ["approval"],
        ["rest"],
        ["tranquility"],
        ["order"],
        ["none"]
    ])
    labels_r = mlb_r.transform(groups_motivation["reiss"])

    mlb_p = MultiLabelBinarizer()
    mlb_p.fit([
        ["disgust:2"],
        ["disgust:3"],
        ["surprise:3"],
        ["surprise:2"],
        ["fear:2"],
        ["fear:3"],
        ["anger:2"],
        ["anger:3"],
        ["trust:3"],
        ["trust:2"],
        ["anticipation:2"],
        ["anticipation:3"],
        ["sadness:3"],
        ["sadness:2"],
        ["joy:2"],
        ["joy:3"],
        ["none"]
    ])
    labels_p =  mlb_p.transform(groups_emotion["plutchik"])

    inputs_m = bert_encode(groups_motivation['full_sentence'].values, tokenizer)
    inputs_e = bert_encode(groups_emotion['full_sentence'].values, tokenizer)
    
    inputs = {
        'motivation': inputs_m,
        'emotion': inputs_e
    }
    labels = {
        'maslow': labels_m,
        'reiss': labels_r,
        'plutchik': labels_p
    }

    return inputs, labels

def build_model_bert_raw(bert_layer, max_len=128):
  input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
  segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

  _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

  model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=sequence_output)
  model.compile(tf.keras.optimizers.Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
  
  return model

def build_model_cnn(num_classes):
    cnn = args = {}
    args['embed_num'] = 128
    args['embed_dim'] = 768
    args['class_num'] = num_classes
    args['kernel_num'] = 100
    args['kernel_sizes'] = [3, 4, 5]
    args['dropout'] = 0.5
    args['static'] = False
    cnn = CNN_Text(
        args
    )
    return cnn

def predict_one_hot(x):
    d = x.reshape(-1, x.shape[-1])
    d2 = np.zeros_like(d)
    d2[np.arange(len(d2)), d.argmax(1)] = 1
    d2 = d2.reshape(x.shape)
    return d2

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        print(args)
        self.args = args
        
        V = args['embed_num']
        D = args['embed_dim']
        C = args['class_num']
        Ci = 1
        Co = args['kernel_num']
        Ks = args['kernel_sizes']

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args['dropout'])
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.args['static']:
            self.embed.weight.requires_grad = False

    def forward(self, x):    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

class SCSDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def get_dataloader(dataset, batch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    return dataloader

def runNetwork(train, num_epochs, net, dataset, batch=32):

    criterion = nn.CrossEntropyLoss()
    if train:
        net.train()
    else:
        net.eval()

    hist_loss = []
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    total_count = 0
    for i in range(num_epochs):
        loader = get_dataloader(dataset, batch)
        for embeddings, label in tqdm(loader):        
            prediction = net.forward(embeddings)
            loss = criterion(prediction, label)
            prediction = prediction.detach().numpy()
            prediction = np.apply_along_axis(predict_one_hot, axis=2, arr=prediction)

            hist_loss.append(loss.item())

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(label, prediction)

            total_count += 1

        print('Epoch {}: Loss: {:.4f}'.format(i, hist_loss[total_count - 1]))

    return hist_loss, precision, recall, f1