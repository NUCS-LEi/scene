import tensorflow as tf


class Model(object):

    @staticmethod
    def inference(x, drop_rate):
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout

        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden9 = dense

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('scene1'):
            dense = tf.layers.dense(hidden10, units=80)
            scene1 = dense

        with tf.variable_scope('scene2'):
            dense = tf.layers.dense(hidden10, units=80)
            scene2 = dense

        with tf.variable_scope('scene3'):
            dense = tf.layers.dense(hidden10, units=80)
            scene3 = dense

        label_logits = tf.stack([scene1, scene2, scene3], axis=1)
        return label_logits

    @staticmethod
    def loss(label_logits, label_batch):
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label_batch, logits=label_logits[:, 0]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label_batch, logits=label_logits[:, 1]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label_batch, logits=label_logits[:, 2]))
        loss = digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy
        return loss