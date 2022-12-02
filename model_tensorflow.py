import csv
import numpy as np
import pandas as pd
from datetime import datetime

train_data = pd.read_csv("data/train.csv")

# new features
train_data["url_count"] = train_data["urls"].apply(lambda s: s[1:-1].count("\'")/2)
train_data["text_len"] = train_data["text"].apply(lambda s: len(s))
train_data["hashtags_count"] = train_data["hashtags"].apply(lambda s: s[1:-1].count("\'")/2)
train_data["day"] = train_data["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t/1000).day)
train_data["hour"] = train_data["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t/1000).hour)

# indicators of keywords
train_data["Macron"] =  train_data["text"].apply(lambda s: ("macron" in s.lower().split()))
train_data["Zemmour"] =  train_data["text"].apply(lambda s: ("zemmour" in s.lower().split()))
train_data["Melenchon"] =  train_data["text"].apply(lambda s: ("melenchon" in s.replace("é","e").lower().split()))
train_data["rt"] =  train_data["text"].apply(lambda s: ("rt" in s.lower().split()))

# sentiment analysis - time consuming
# from nltk.sentiment import SentimentIntensityAnalyzer
# sia = SentimentIntensityAnalyzer()
# print("sentiment analysis...")
# train_data["compound"] =  train_data["text"].apply(lambda s: sia.polarity_scores(s)['compound'])

print(train_data.corr())

# select useful columns
train_data_filtered = train_data.loc[:, ["retweets_count","favorites_count","followers_count","statuses_count","friends_count",
                                 "hashtags_count","hour","verified","url_count","text_len","rt","Macron","Zemmour","Melenchon"]]

from verstack.stratified_continuous_split import scsplit

X_train, X_eval, y_train, y_eval = scsplit(train_data_filtered, train_data_filtered['retweets_count'], stratify=train_data_filtered['retweets_count'], test_size=0.3)
X_train = X_train.drop(['retweets_count'], axis=1)
X_eval = X_eval.drop(['retweets_count'], axis=1)

# Standardize the data
normal_columns = ["favorites_count","followers_count","statuses_count","friends_count","text_len"]
mu, sigma = X_train[normal_columns].mean(axis=0), X_train[normal_columns].std(axis=0)
X_train.loc[:, normal_columns] = (X_train[normal_columns] - mu) / sigma
X_eval.loc[:, normal_columns] = (X_eval[normal_columns] - mu) / sigma

# MLP
import tensorflow as tf
from tensorflow.keras.constraints import MaxNorm

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(1),
])


# model.compile(optimizer='adam', loss='mae')

import tensorflow_addons as tfa

lr = 1e-3
wd = 1e-5
optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)

model.compile(optimizer=optimizer, loss='mae')

# initial_lr = 0.1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_lr,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)

# initial_wd = 1e-4 * initial_lr
# wd_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_wd,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)

# optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule)
# model.compile(optimizer=optimizer, loss='mae')

model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.float32), epochs=1000, batch_size=1000,
         validation_data=(X_eval.values.astype(np.float32), y_eval.values.astype(np.float32)), shuffle=True)

model.evaluate(X_eval.values.astype(np.float32),  y_eval.values.astype(np.float32), verbose=2)

# evaluation

eval_data = pd.read_csv("evaluation.csv")

eval_data["url_count"] = eval_data["urls"].apply(lambda s: s[1:-1].count("\'")/2)
eval_data["text_len"] = eval_data["text"].apply(lambda s: len(s))
eval_data["hashtags_count"] = eval_data["hashtags"].apply(lambda s: s[1:-1].count("\'")/2)
eval_data["day"] = eval_data["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t/1000).day)
eval_data["hour"] = eval_data["timestamp"].apply(lambda t: datetime.utcfromtimestamp(t/1000).hour)
eval_data["Macron"] =  eval_data["text"].apply(lambda s: ("macron" in s.lower().split()))
eval_data["Zemmour"] =  eval_data["text"].apply(lambda s: ("zemmour" in s.lower().split()))
eval_data["Melenchon"] =  eval_data["text"].apply(lambda s: ("melenchon" in s.lower().split()))
eval_data["rt"] =  eval_data["text"].apply(lambda s: ("rt" in s.lower().split()))

# print("sentiment analysis...")
# eval_data["compound"] =  eval_data["text"].apply(lambda s: sia.polarity_scores(s)['compound'])

eval_data = eval_data.loc[:, ["favorites_count","followers_count","statuses_count","friends_count",
                                 "hashtags_count","hour","verified","url_count","text_len","rt","Macron","Zemmour","Melenchon"]]

# normalize
eval_data.loc[:, normal_columns] = (eval_data.loc[:, normal_columns] - mu) / sigma

print(eval_data)

pred = model.predict(eval_data.values.astype(np.float32))

print(pred)

# output normalization
for i,p in enumerate(pred):
    if p<0: pred[i] = 0

eval_data = pd.read_csv("evaluation.csv")
with open("predictions.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "retweets_count"])
    for index, prediction in enumerate(pred):
        writer.writerow([str(eval_data['TweetID'].iloc[index]) , str(int(prediction))])

