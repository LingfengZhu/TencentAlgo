import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Tools for building a model
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
import keras.backend as K
from keras.optimizers import *
from keras.utils import to_categorical
import tensorflow as tf

# Tools for assessing the quality of model prediction
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


from tqdm import tqdm
import gc

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                 start_mem - end_mem) / start_mem))
    return df

# 读取并合并数据集
train_dir = "train_preliminary/"
test_dir = "test/"
click_train = pd.read_csv(train_dir + "click_log.csv")
click_train = reduce_mem_usage(click_train)
ad_train = pd.read_csv(train_dir + "ad.csv")
ad_train = reduce_mem_usage(ad_train)
click_log = click_train.merge(ad_train, how="left", on="creative_id")
# click_log["type"] = "train"

click_test = pd.read_csv(test_dir + "click_log.csv")
ad_test = pd.read_csv(test_dir + "ad.csv")
click_test = reduce_mem_usage(click_test)
ad_test = reduce_mem_usage(ad_test)
click_log_test = click_test.merge(ad_test, how="left", on="creative_id")
# click_log_test['type'] = "test"

click_log_sort = click_log.sort_values("time")
click_log_test_sort = click_log_test.sort_values("time")

click_all = click_log_sort.append(click_log_test_sort)
print(click_all.shape)

click_all["product_id"] = click_all["product_id"].apply(lambda x : 0 if x == '\\N' else x)
click_all["industry"] = click_all["product_id"].apply(lambda x : 0 if x == '\\N' else x)
click_all["product_id"] = click_all["product_id"].astype(np.int32)
click_all["industry"] = click_all["industry"].astype(np.int32)
click_all = reduce_mem_usage(click_all)

train_len = click_log['user_id'].nunique()

sentence1 = click_all.groupby('user_id')['creative_id'].agg(lambda x: list(x.astype(str))).tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence1)
sequences1 = tokenizer.texts_to_sequences(sentence1)

maxlen = 100
train_pad1 = pad_sequences(sequences1, maxlen, truncating='post', padding='post')

# 将训练好的creative_id词向量，保存到字典里
embedding_dict1={}
with open("ljx_ad/feature_v300/w2v_creative_id.txt", 'r') as f:
    for i, line in enumerate(f):
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict1[word]=vectors
f.close()

embedding1 = []
#将字典中键和值循环取出添加到列表中
for i in embedding_dict1.keys():
    if len(embedding_dict1[i]) > 1:
        embedding1.append(embedding_dict1[i])
embedding1 = np.array(embedding1)

x1 = train_pad1[:train_len]
x1_test = train_pad1[train_len:]

# 标签
user = pd.read_csv("train_preliminary/user.csv")
gender = user['gender'] - 1


# 将训练好的ad_id词向量，保存到字典里
embedding_dict2 = {}
with open("ljx_ad/feature_v300/w2v_ad_id.txt", 'r') as f:
    for i, line in enumerate(f):
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:],'float32')
        embedding_dict2[word]=vectors
f.close()

embedding2 = []
for i in embedding_dict2.keys():
    if len(embedding_dict2[i]) > 1:
        embedding2.append(embedding_dict2[i])
embedding2 = np.array(embedding2)

sentence2 = click_all.groupby('user_id')['ad_id'].agg(lambda x: list(x.astype(str))).tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence2)
sequences2 = tokenizer.texts_to_sequences(sentence2)

train_pad2 = pad_sequences(sequences2, maxlen, truncating='post', padding='post')

x2 = train_pad2[:train_len]
x2_test = train_pad2[train_len:]


# 将训练好的ad_id词向量，保存到字典里
embedding_dict3 = {}
with open("ljx_ad/feature_v300/w2v_advertiser_id.txt", 'r') as f:
    for i, line in enumerate(f):
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:],'float32')
        embedding_dict3[word]=vectors
f.close()

embedding3 = []
for i in embedding_dict3.keys():
    if len(embedding_dict3[i]) > 1:
        embedding3.append(embedding_dict3[i])
embedding3 = np.array(embedding3)

sentence3 = click_all.groupby('user_id')['advertiser_id'].agg(lambda x: list(x.astype(str))).tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence3)
sequences3 = tokenizer.texts_to_sequences(sentence3)

train_pad3 = pad_sequences(sequences3, maxlen, truncating='post', padding='post')

x3 = train_pad3[:train_len]
x3_test = train_pad3[train_len:]


def Model_gender(embedding1, embedding2, embedding3):
    K.clear_session()

    seq1 = Input(shape=(maxlen,))
    seq2 = Input(shape=(maxlen,))
    seq3 = Input(shape=(maxlen,))

    emb_layer_1 = Embedding(
        input_dim=embedding1.shape[0],
        output_dim=embedding1.shape[1],
        input_length=maxlen,
        weights=[embedding1],
        trainable=False
    )

    emb_layer_2 = Embedding(
        input_dim=embedding2.shape[0],
        output_dim=embedding2.shape[1],
        input_length=maxlen,
        weights=[embedding2],
        trainable=False
    )

    emb_layer_3 = Embedding(
        input_dim=embedding3.shape[0],
        output_dim=embedding3.shape[1],
        input_length=maxlen,
        weights=[embedding3],
        trainable=False
    )


    x1 = emb_layer_1(seq1)
    x2 = emb_layer_2(seq2)
    x3 = emb_layer_3(seq3)


    sdrop = SpatialDropout1D(rate=0.2)

    x1 = sdrop(x1)
    x2 = sdrop(x2)
    x3 = sdrop(x3)


    x1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Bidirectional(LSTM(128, recurrent_dropout=0.1))(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(64, activation="relu")(x1)

    x2 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Bidirectional(LSTM(128, recurrent_dropout=0.1))(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(64, activation="relu")(x2)

    x3 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Bidirectional(LSTM(128, recurrent_dropout=0.1))(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Dense(64, activation="relu")(x3)


    x = concatenate([x1, x2, x3])
    x = Activation(activation="relu")(BatchNormalization()(Dense(32)(x)))
    pred = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[seq1, seq2, seq3], outputs=pred)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    return model


# gc.collect()
#
# skf = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
# sub_gender = np.zeros((x1_test.shape[0],1))
# # oof_pred = np.zeros((x1.shape[0],))
# score = []
# count = 0
# if not os.path.exists("model"):
#     os.mkdir("model")
#
# for i, (train_index, test_index) in enumerate(skf.split(x1, gender)):
#     print("FOLD | ", count + 1)
#     print("###" * 35)
#     gc.collect()
#     filepath = "model/nn_v1.h5"
#     checkpoint = ModelCheckpoint(
#         filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
#     reduce_lr = ReduceLROnPlateau(
#         monitor='val_acc', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
#     earlystopping = EarlyStopping(
#         monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='max')
#     callbacks = [checkpoint, reduce_lr, earlystopping]
#
#     model_gender = Model_gender(embedding1, embedding2, embedding3, embedding4, embedding5, embedding6)
#     if count == 0:
#         model_gender.summary()
#     x1_tr, x1_va = x1[train_index], x1[test_index]
#     x2_tr, x2_va = x2[train_index], x2[test_index]
#     x3_tr, x3_va = x3[train_index], x3[test_index]
#     x4_tr, x4_va = x4[train_index], x4[test_index]
#     x5_tr, x5_va = x5[train_index], x5[test_index]
#     x6_tr, x6_va = x6[train_index], x6[test_index]
#
#     y_tr, y_va = gender[train_index], gender[test_index]
#
#     hist = model_gender.fit([x1_tr, x2_tr, x3_tr, x4_tr, x5_tr, x6_tr],
#                          y_tr, batch_size=128, epochs=15,
#                          validation_data=([x1_va, x2_va, x3_va, x4_va, x5_va, x6_va], y_va),
#                          callbacks=callbacks, verbose=1, shuffle=True)
#
#     #     model_age.load_weights(filepath)
#     # oof_pred[test_index] = model_age.predict([x1_va, x2_va, x3_va], batch_size=2048, verbose=1)
#     sub_gender += model_gender.predict([x1_test, x2_test, x3_test, x4_test, x5_test, x6_test], batch_size=128, verbose=1) / skf.n_splits
#     #     score.append(np.max(hist.history['val_acc']))
#     count += 1
#
# 保存结果
prediction = pd.DataFrame()
prediction['user_id'] = click_log_test['user_id'].sort_values().unique()
#
# prediction['predicted_gender'] = sub_gender
# prediction.loc[prediction['predicted_gender'] >= 0.5, 'predicted_gender'] = 2
# prediction.loc[prediction['predicted_gender'] < 0.5, 'predicted_gender'] = 1
#
# 十分类 ，训练年龄
y_age = user['age'] - 1
age = pd.get_dummies(y_age).values


def Model_age(embedding1, embedding2, embedding3):
    K.clear_session()

    seq1 = Input(shape=(maxlen,))
    seq2 = Input(shape=(maxlen,))
    seq3 = Input(shape=(maxlen,))

    emb_layer_1 = Embedding(
        input_dim=embedding1.shape[0],
        output_dim=embedding1.shape[1],
        input_length=maxlen,
        weights=[embedding1],
        trainable=False
    )

    emb_layer_2 = Embedding(
        input_dim=embedding2.shape[0],
        output_dim=embedding2.shape[1],
        input_length=maxlen,
        weights=[embedding2],
        trainable=False
    )

    emb_layer_3 = Embedding(
        input_dim=embedding3.shape[0],
        output_dim=embedding3.shape[1],
        input_length=maxlen,
        weights=[embedding3],
        trainable=False
    )


    x1 = emb_layer_1(seq1)
    x2 = emb_layer_2(seq2)
    x3 = emb_layer_3(seq3)


    sdrop = SpatialDropout1D(rate=0.2)

    x1 = sdrop(x1)
    x2 = sdrop(x2)
    x3 = sdrop(x3)


    x1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Bidirectional(LSTM(128, recurrent_dropout=0.1))(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(64, activation="relu")(x1)

    x2 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Bidirectional(LSTM(128, recurrent_dropout=0.1))(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(64, activation="relu")(x2)

    x3 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Bidirectional(LSTM(128, recurrent_dropout=0.1))(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Dense(64, activation="relu")(x3)

    x = concatenate([x1, x2, x3])
    x = Activation(activation="relu")(BatchNormalization()(Dense(32)(x)))
    pred = Dense(10, activation='softmax')(x)
    model = Model(inputs=[seq1, seq2, seq3], outputs=pred)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    return model


gc.collect()

skf = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
sub_age = np.zeros((x1_test.shape[0], 10))
oof_pred = np.zeros((x1.shape[0], 10))
score = []
count = 0
if not os.path.exists("model"):
    os.mkdir("model")

for i, (train_index, test_index) in enumerate(skf.split(x1, y_age)):
    print("FOLD | ", count + 1)
    print("###" * 35)
    gc.collect()
    filepath = "model/nn_v1.h5"
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='max')
    callbacks = [checkpoint, reduce_lr, earlystopping]

    model_age = Model_age(embedding1, embedding2, embedding3)
    if count == 0:
        model_age.summary()
    x1_tr, x1_va = x1[train_index], x1[test_index]
    x2_tr, x2_va = x2[train_index], x2[test_index]
    x3_tr, x3_va = x3[train_index], x3[test_index]

    y_tr, y_va = age[train_index], age[test_index]

    hist = model_age.fit([x1_tr, x2_tr, x3_tr],
                         y_tr, batch_size=128, epochs=15,
                         validation_data=([x1_va, x2_va, x3_va], y_va),
                         callbacks=callbacks, verbose=1, shuffle=True)

    #     model_age.load_weights(filepath)
    oof_pred[test_index] = model_age.predict([x1_va, x2_va, x3_va], batch_size=128, verbose=1)
    sub_age += model_age.predict([x1_test, x2_test, x3_test], batch_size=128, verbose=1) / skf.n_splits
    #     score.append(np.max(hist.history['val_acc']))
    count += 1

# 对测试集测试
prediction['predicted_age'] = np.argmax(sub_age, axis=1)
prediction['predicted_age'] = prediction['predicted_age'] + 1


# 保存文件
prediction.to_csv("3input.csv", index=False)

