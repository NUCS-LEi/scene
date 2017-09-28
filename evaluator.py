import tensorflow as tf
from donkey import Donkey
from model import Model


class Evaluator(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def evaluate(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = num_examples // batch_size
        needs_include_length = False

        with tf.Graph().as_default():
            image_batch, lable_batch = Donkey.build_batch(path_to_tfrecords_file,
                                                                 num_examples=num_examples,
                                                                 batch_size=batch_size,
                                                                 shuffled=False)
            label_logits = Model.inference(image_batch, drop_rate=0.0)
            label_predictions1 = tf.argmax(label_logits[:,0], axis=1)
            label_predictions2 = tf.argmax(label_logits[:,1], axis=1)
            label_predictions3 = tf.argmax(label_logits[:,2], axis=1)
            accuracy1, update_accuracy1 = tf.metrics.accuracy(
                labels=lable_batch,
                predictions=label_predictions1
            )
            accuracy2, update_accuracy2 = tf.metrics.accuracy(
                labels=lable_batch,
                predictions=label_predictions2
            )
            accuracy3, update_accuracy3 = tf.metrics.accuracy(
                labels=lable_batch,
                predictions=label_predictions3
            )
            accuracy = accuracy1 + accuracy2 + accuracy3
            update_accuracy = update_accuracy1 + update_accuracy2 + update_accuracy3
        

            tf.summary.image('image', image_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                 tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)

                for _ in range(num_batches):
                    sess.run(update_accuracy)

                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val
