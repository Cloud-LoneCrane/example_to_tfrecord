import tensorflow as tf
import random
import numpy as np


def serialize_sample_image_with_image(image, mask):
    """
    序列化input和label都是图片的样本，如图像分割和图片深度预测
    :param image:
    :param mask:
    :return:
    """
    # image, mask.shape=[30, 512, 512]
    # 1.创建用于Feature对象的list
    image_bytelist = tf.train.BytesList(value=[image.tobytes()])
    mask_bytelist = tf.train.BytesList(value=[mask.tobytes()])
    shape_int64list = tf.train.Int64List(value=list(image.shape))

    # 2.构建Features对象
    features = tf.train.Features(
        feature={
            "image": tf.train.Feature(bytes_list=image_bytelist),
            "mask": tf.train.Feature(bytes_list=mask_bytelist),
            "shape": tf.train.Feature(int64_list=shape_int64list)
        }
    )

    # 3.features组建Example
    example = tf.train.Example(features=features)

    # 4.将Example序列化
    return example.SerializeToString()


def serialize_sample_image_with_value(image, value):
    """
    序列化input是image, label是单值的样本, 如图片类别预测
    :param image:
    :param value:
    :return:
    """
    # 1.创建用于Feature对象的list
    image_bytelist = tf.train.BytesList(value=[image.tobytes()])
    value_bytelist = tf.train.Int64List(value=[value])
    shape_int64list = tf.train.Int64List(value=list(image.shape))

    # 2.构建Features对象
    features = tf.train.Features(
        feature={
            "image": tf.train.Feature(bytes_list=image_bytelist),
            "value": tf.train.Feature(int64_list=value_bytelist),
            "shape": tf.train.Feature(int64_list=shape_int64list)
        }
    )

    # 3.features组建Example
    example = tf.train.Example(features=features)

    # 4.将Example序列化
    return example.SerializeToString()


def save_serialize_to_tfrecord(serialize_sample_list, full_name):
    """
    将传入的serialize_sample_list 写入到名为full_name的tfrecord文件中
    :param serialize_sample_list:
    :param full_name:
    :return:
    """
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(full_name, options=options) as writer:
        for serialize_sample in serialize_sample_list:
            # 每次写入一个样本
            # 5.将Example写入到tfrecord文件
            writer.write(serialize_sample)


def shuffler_array_inputs_labels(inputs, labels):
    """
    打乱传入的inputs和labels
    :param inputs: array_inputs
    :param labels: array_labels
    :return: inputs, labels
    """
    permutation = np.random.permutation(inputs.shape[0])
    inputs = inputs[permutation]
    labels = labels[permutation]

    return inputs, labels


def shuffler_list_inputs_labels(inputs, labels):
    """
    打乱传入的inputs和labels
    :param inputs: list_inputs
    :param labels: list_labels
    :return: inputs, labels
    """
    state = random.getstate()
    random.shuffle(inputs)
    random.setstate(state)
    random.shuffle(labels)

    return inputs, labels


def save_data_hist_fig(data, png_name, bins=100):
    """
    将传入的data进行直方图统计并将结果保存到png_name指定的名字中
    :param data:
    :param png_name:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.hist(data.ravel(), bins=bins)
    plt.savefig(png_name)
    plt.clf()
    return None


if __name__ == '__main__':
    # a = np.array(range(10))
    # b = np.array(range(10))
    #
    # a, b = shuffler_array_inputs_labels(a, b)
    # print(a, b)
    #
    # c = list(a)
    # d = list(b)
    # c, d = shuffler_list_inputs_labels(c, d)
    # print(c, d)

    data = np.random.random((32, 32))*100
    save_data_hist_fig(data, "test.png")