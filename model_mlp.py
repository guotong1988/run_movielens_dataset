
import numpy as np
import time
import os

emb_dim = 32
from sklearn import metrics
from common_func import *



class RecModel(object):

    def __init__(self,
                 feature_name_list,
                 max_ids,
                 is_training):

        self.batch_size = 1024
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

        self.input_label = tf.placeholder(shape=[None], dtype=tf.float32)

        self.input_ids_batch = tf.placeholder(shape=[None, len(max_ids)], dtype=tf.int32)

        feature_list = []

        for index in range(len(max_ids)):
            embed_feature = to_emb_int_id(share_feature_name=feature_name_list[index],
                                          input_ids_batch=self.input_ids_batch[:, index:index + 1],
                                          emb_matrix_size=max_ids[index] + 1,
                                          emb_hidden_dim=emb_dim)
            feature_list.extend(embed_feature)

        with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE):
            last_layer = ensemble_layer(feature_list)

        self.logits = tf.layers.dense(inputs=last_layer,
                                      units=1,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      name='logits')
        self.loss = loss_function(labels=tf.expand_dims(self.input_label, -1), logits=self.logits)

        self.score = tf.sigmoid(self.logits)
        if not is_training: return
        self.train_op = tf.train.AdamOptimizer(0.00005).minimize(self.loss, global_step=self.global_step)


    def build_feat(self, batch):
        y_batch = np.array(batch)[:, 0]
        X_batch = np.array(batch)[:, 1:]

        feed_dict = {
            self.input_label: y_batch,
            self.input_ids_batch: X_batch
        }
        return feed_dict


    def train(self, data_train, data_dev, dev_model):
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            best_auc = 0
            epoch_no_improve = 0

            for epoch in range(100):
                print("---------- epoch %d ---------" % epoch)
                self.train_one_epoch(sess, data_train)

                print("\nEvaluation:")
                avg_auc = dev_model.run_evaluate(sess, data_dev)
                if avg_auc > best_auc:
                    print("{0}:{1},new best score!!!".format(avg_auc, best_auc))
                    epoch_no_improve = 0
                    best_auc = avg_auc
                    step = tf.train.global_step(sess, self.global_step)
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                else:
                    epoch_no_improve += 1
                    if epoch_no_improve >= 5:
                        break

    def run_evaluate(self, sess, data_dev):
        batches = batches_func(data_dev, self.batch_size, False)
        batch_count = 0.0
        total_loss = 0.0
        score_list = []
        score_list_binary = []
        label_list_binary = []

        for one_batch in batches:
            batch_count += 1
            feed_dict_dev = self.build_feat(batch=one_batch)

            loss, score, label = sess.run([self.loss, self.score, self.input_label], feed_dict_dev)

            total_loss += loss
            score_list.extend(score)
            for one in score:
                if one > 0.8:
                    score_list_binary.append(1)
                else:
                    score_list_binary.append(0)
            for one in label:
                if one > 0.8:
                    label_list_binary.append(1)
                else:
                    label_list_binary.append(0)
        f1 = metrics.f1_score(label_list_binary, score_list_binary)
        avg_auc = metrics.roc_auc_score(label_list_binary, score_list)

        avg_loss = total_loss / batch_count
        print("loss", avg_loss, "auc", avg_auc, "f1", f1)
        return avg_auc


    def train_one_epoch(self, sess, data_train):
        batches = batches_func(data_train, self.batch_size)
        total_loss = 0.0
        one_epoch_step = 0
        for batch in batches:
            one_epoch_step += 1
            feed_dict = self.build_feat(batch)
            _, loss = sess.run(
                [self.train_op, self.loss],
                feed_dict)
            total_loss += loss
            if one_epoch_step % 2000 == 0:
                print(one_epoch_step)
                avg_loss = total_loss / one_epoch_step
                print("loss {:g}".format(avg_loss))
