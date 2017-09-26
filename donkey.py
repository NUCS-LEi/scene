import tensorflow as tf


class Donkey(object):
    @staticmethod
    def _preprocess(image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.multiply(tf.subtract(image, 0.5), 2)
        image = tf.reshape(image, [64, 64, 3])
        image = tf.random_crop(image, [54, 54, 3])
        return image

    @staticmethod
    def _read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })

        image = Donkey._preprocess(tf.decode_raw(features['image'], tf.uint8))
        label = tf.cast(features['label'], tf.int32)
        return image, label

    @staticmethod
    def build_batch(path_to_tfrecords_file, num_examples, batch_size, shuffled):
        assert tf.gfile.Exists(path_to_tfrecords_file), '%s not found' % path_to_tfrecords_file

        filename_queue = tf.train.string_input_producer([path_to_tfrecords_file], num_epochs=None)
        image, label = Donkey._read_and_decode(filename_queue)

        min_queue_examples = int(0.4 * num_examples)
        if shuffled:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                                     batch_size=batch_size,
                                                                     num_threads=2,
                                                                     capacity=min_queue_examples + 3 * batch_size,
                                                                     min_after_dequeue=min_queue_examples)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                             batch_size=batch_size,
                                                             num_threads=2,
                                                             capacity=min_queue_examples + 3 * batch_size)
        return image_batch, label_batch