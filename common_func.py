import tensorflow as tf

init = tf.initializers.random_uniform(minval=-0.01, maxval=0.01)
kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
bias_initializer = tf.constant_initializer(0.0)


def batches_func(data, batch_size):
    data_size = len(data)
    number_batches = int((data_size - 1) / batch_size) + 1
    for batch_num in range(number_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]


def loss_function(logits, labels):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy)


def prelu(_x, scope=''):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=[_x.get_shape()[-1]],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def ensemble_layer(feature_list):
    features = tf.concat(feature_list, 1)  # [batch_size, feature_num * hidden_size]
    mid_layer = tf.layers.dense(inputs=features,
                                units=256,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                name='fc1')

    mid_layer = prelu(mid_layer, "fc1")
    mid_layer = tf.layers.dense(inputs=mid_layer,
                                units=32,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                name='deep_layer')
    return mid_layer  # [batch_size, final_hidden_size]


def to_emb_int_id(share_feature_name, input_ids_batch, emb_matrix_size, emb_hidden_dim=32):
    feature_size = input_ids_batch.get_shape()[1]
    embed_list = []
    emb_matrix = tf.get_variable(name=share_feature_name,
                                 shape=[emb_matrix_size + 1, emb_hidden_dim],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    for feature_idx in range(0, feature_size):
        one_id = input_ids_batch[:, feature_idx]
        feature_emb = tf.nn.embedding_lookup(emb_matrix, one_id)
        embed_list.append(feature_emb)
    return embed_list