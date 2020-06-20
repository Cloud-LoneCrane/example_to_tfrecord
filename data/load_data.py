import tensorflow as tf
import numpy as np


def _parse_serialized_images_images(serialized_example):
    """
    将输入的序列化的input和label都是image的example解析并返回
    :param serialized_example:
    :return:
    """
    # 定义一个解析序列的features
    expected_features = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "shape": tf.io.FixedLenFeature(shape=[3], dtype=tf.int64)
    }
    # 反序列化
    parsed_features = tf.parse_single_example(serialized_example, features=expected_features)
    shape = parsed_features["shape"]
    img = tf.cast(tf.reshape(tf.decode_raw(parsed_features["image"], tf.float64), shape=shape), dtype=tf.float32)
    musk = tf.cast(tf.reshape(tf.decode_raw(parsed_features["mask"], tf.float64), shape=shape), dtype=tf.float32)
    size = [512, 512, 1]
    return tf.image.resize_images(img, size=size), tf.image.resize_images(musk, size=size)


def _parse_serialized_images_values(serialized_example):
    """
    将输入的序列化的input是图片，label是单值的example解析并返回
    :param serialized_example:
    :return:
    """
    # 定义一个解析序列的features
    expected_features = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
        "shape": tf.io.FixedLenFeature(shape=[3], dtype=tf.int64)
    }
    # 反序列化
    parsed_features = tf.parse_single_example(serialized_example, features=expected_features)
    shape = parsed_features["shape"]
    img = tf.cast(tf.reshape(tf.decode_raw(parsed_features["image"], tf.float32), shape=shape), dtype=tf.float32)
    label = parsed_features["label"]

    size = [512, 512, 1]
    return tf.image.resize_images(img, size=size), label


def tfrecord_filenames_to_dataset(file_names, BATCH_SIZE=1, parse_function=_parse_serialized_images_images):
    """
    :param file_names:
    :param BATCH_SIZE:
    :param parse_function:
    :return:
    """
    # 2.构建文件dataset
    dataset_file = tf.data.Dataset.list_files(file_names)

    # 3.构建全部文件内容的dataset_filecontent
    dataset_filecontent = dataset_file.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
        cycle_length=3  # 读取文件的并行数
    )

    # 3.构建样本的dataset
    dataset = dataset_filecontent.map(parse_function,  # 负责将example解析并反序列化处理的函数
                                      num_parallel_calls=6  # 处理样本的并行线程数量
                                      )
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.repeat().batch(BATCH_SIZE)
    return dataset


parse_image_image = _parse_serialized_images_images
parse_image_value = _parse_serialized_images_values