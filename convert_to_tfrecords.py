import os
import numpy as np
import json
import random
from PIL import Image
import tensorflow as tf
from meta import Meta

tf.app.flags.DEFINE_string('data_dir', '.\data',
                           'Directory to SVHN (format 1) folders and write the converted files')
FLAGS = tf.app.flags.FLAGS


class ExampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0

    @staticmethod
    def _get_attrs(digit_struct_mat_file, index):
        """
        Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
        attrs = {}
        f = digit_struct_mat_file
        item = f['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = f[item][key]
            values = [f[attr.value[i].item()].value[0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            attrs[key] = values
        return attrs

    @staticmethod
    def _preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):
        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
#         image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        image = image.resize([64, 64])
        return image

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self, label):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        index = path_to_image_file.split('\\')[-1]
        self._example_pointer += 1

        image = np.array(Image.open(path_to_image_file).resize([64, 64])).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'label': ExampleReader._int64_feature(label[index])
        }))
        return example


def convert_to_tfrecords(path_to_dataset_dir_and_digit_struct_mat_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))

    for path_to_dataset_dir, path_to_digit_struct_mat_file in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.jpg'))
        total_files = len(path_to_image_files)
        print('%d files found in %s' % (total_files, path_to_dataset_dir))

        with open(path_to_digit_struct_mat_file) as digit_struct_mat_file:    
            data = json.load(digit_struct_mat_file)
            label = {}
            for item in data:
                label[item['image_id']] = int(item['label_id'])
            example_reader = ExampleReader(path_to_image_files)
            for index, path_to_image_file in enumerate(path_to_image_files):
                print('(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file))

                example = example_reader.read_and_convert(label)
                if example is None:
                    break

                idx = choose_writer_callback(path_to_tfrecords_files)
                writers[idx].write(example.SerializeToString())
                num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print('Saving meta file to %s...' % path_to_tfrecords_meta_file)
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)


def main(_):
    path_to_train_dir = os.path.join(FLAGS.data_dir, 'train')
    path_to_test_dir = os.path.join(FLAGS.data_dir, 'test')
    path_to_train_image_dir = os.path.join(path_to_train_dir, 'images')
    path_to_test_image_dir = os.path.join(path_to_test_dir, 'images')
    path_to_train_digit_json_file = os.path.join(path_to_train_dir, 'scene.json')
    path_to_test_digit_json_file = os.path.join(path_to_test_dir, 'scene.json')

    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')

    for path_to_file in [path_to_train_tfrecords_file, path_to_val_tfrecords_file, path_to_test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    print('Processing training and validation data...')
    [num_train_examples, num_val_examples] = convert_to_tfrecords([(path_to_train_image_dir, path_to_train_digit_json_file)],
                                                                  [path_to_train_tfrecords_file, path_to_val_tfrecords_file],
                                                                  lambda paths: 0 if random.random() > 0.1 else 1)
    print('Processing test data...')
    [num_test_examples] = convert_to_tfrecords([(path_to_test_image_dir, path_to_test_digit_json_file)],
                                               [path_to_test_tfrecords_file],
                                               lambda paths: 0)

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file)

    print('Done')


if __name__ == '__main__':
    tf.app.run(main=main)