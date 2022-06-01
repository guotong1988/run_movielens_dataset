
# EDIT HERE TO CHANGE MODEL
from model_transformer import RecModel

f = open('./data/train.tsv', encoding="utf-8", mode="r")

data_train = []
data_dev = []

for i, line in enumerate(f):
    if i == 0:
        continue
    splits = [float(one) for one in line.strip().split(",")]
    if i % 50 == 0:
        data_dev.append(splits)
    else:
        data_train.append(splits)

all_feature_names = ['C'] * 30
max_id_list = []
for name in all_feature_names:
    max_id = 1129
    max_id_list.append(max_id)


import tensorflow as tf

with tf.variable_scope("", reuse=tf.AUTO_REUSE):
    model_train = RecModel(
        feature_name_list=all_feature_names,
        max_ids=max_id_list,
        is_training=True)
with tf.variable_scope("", reuse=True):
    model_dev = RecModel(
        feature_name_list=all_feature_names,
        max_ids=max_id_list,
        is_training=False)

model_train.train(data_train, data_dev, model_dev)

