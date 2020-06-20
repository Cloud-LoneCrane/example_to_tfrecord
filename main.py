from utils import serialize_sample_image_with_image, save_data_hist_fig, shuffler_array_inputs_labels, \
    save_serialize_to_tfrecord, save_example_to_tfrecord
from data import tfrecord_filenames_to_dataset
import os
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt

image_dir = r"D:\data\COVID19\merged\image"
mask_dir = r"D:\data\COVID19\merged\mask"
train_test_split = 0.85

num_examples_one_tfrecord = 100

save_tfrecord_dir = ""


def _get_raw_example(dir):
    mix_filenames = os.listdir(dir)

    image_file_names = [name for name in mix_filenames if name.startswith("tr_im")]
    mask_file_name = [name for name in mix_filenames if name.startswith("tr_mask")]

    x = []
    y = []
    counter = 0
    for image_name, mask_name in zip(image_file_names, mask_file_name):
        images = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir, image_name)))
        masks = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir, mask_name)))
        # print(image_name, images.shape, end="")

        if not os.path.exists("hist"):
            os.mkdir("hist")
        # 归一化前进行数据分布的直方图可视化
        save_png_name = "hist/raw-{}.png"
        # for i in range(images.shape[0]):
        #     save_data_hist_fig(images[i], save_png_name.format(i))
        # save_data_hist_fig(images, "hist/raw_merged.png")

        import matplotlib.pyplot as plt
        plt.hist(images.ravel(), bins=100)
        # plt.hist(np.multiply(images, masks).ravel(), bins=100)
        plt.savefig("hist/raw_merged.png")
        plt.clf()

        # 对images进行归一化处理[0, 1]
        images = images*1.0
        MIN, MAX = np.min(images), np.max(images)
        print(image_name, MIN, MAX, images.shape, np.min(masks), np.max(masks))

        images = (images-MIN)/(MAX-MIN)
        print(image_name, MIN, MAX, images.shape, np.min(masks), np.max(masks))
        # 归一化前进行数据分布的直方图可视化
        save_png_name = "hist/after-{}.png"
        # for i in range(images.shape[0]):
        #     save_data_hist_fig(images[i], save_png_name.format(i))
        # save_data_hist_fig(images, "hist/after_merged.png")

        # images = np.power(images, 0.4)
        plt.hist(images.ravel(), bins=100)
        # plt.hist(np.multiply(images, masks).ravel(), bins=100)
        plt.savefig("hist/after_merged.png")
        plt.clf()

        return None

    #     # 先确定当前image的图片数量
    #     num = len(masks)
    #     hw = len(masks[0, :])
    #     if hw == 630:
    #         images = images[:, 59:512 + 59, 59:512 + 59]
    #         masks = masks[:, 59:512 + 59, 59:512 + 59]
    #
    #     images = images.reshape([num, 512, 512, 1])
    #     masks = masks.reshape([num, 512, 512, 1])
    #
    #     del images
    #     del masks
    #
    # # # 对x进行归一化处理, （已处理）
    # # from sklearn.preprocessing import StandardScaler
    # # scaler = StandardScaler()
    # # x = scaler.fit_transform(x)
    #
    # # 保证测试集和训练集无重叠，对增强前的数据切分出train和test
    # split = 0.85
    # split = int(np.floor(counter * split))
    # train_images = x[:split]
    # train_masks = y[:split]
    # test_images = x[split:]
    # test_masks = y[split:]
    #
    # # 删除不用的变量节约内存
    # del x
    # del y
    # del image_file_names
    # del mask_file_name
    #
    # # 转成numpy数组
    # # train_images = np.array(train_images, dtype=np.float16)
    # # print(train_images.shape)
    # # 数据量太大，直接转换成numpy数组内存报错
    #
    # # 方案2:
    # # 按照样本进行处理：
    # # 一个样本：[step, h, w, 1]-->数据增强-->增强后-->[8, step, h, w, 1]-->写入到tfrecord文件
    # # 此处要控制多少个原始的样本写入到一个tfrecord文件，另外这些增强后的样本要先进行一个随机处理之后再写入到tfrecord
    #
    # return train_images, train_masks, test_images, test_masks


def show_raw_nii(example_filename):
    import matplotlib
    matplotlib.use('TkAgg')
    import nibabel as nib
    from nibabel.viewers import OrthoSlicer3D

    img = nib.load(example_filename)
    width, height, queue = img.dataobj.shape
    print(width, height, queue)
    OrthoSlicer3D(img.dataobj).show()

    return None


def get_raw_example():
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    filenames = os.listdir(image_dir)

    images = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_dir, filenames[0])))
    images = images.reshape(images.shape + (1,))
    masks = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, filenames[0])))
    masks = masks.reshape(masks.shape + (1,))
    hw = images.shape[1]
    if hw == 630:
        images = images[:, 59:512 + 59, 59:512 + 59]
        masks = masks[:, 59:512 + 59, 59:512 + 59]
        # 归一化
        MIN, MAX = np.min(images), np.max(images)
        images = (images - MIN) / (MAX - MIN)

    for name in filenames[1:]:
        imgs = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_dir, name)))
        imgs = imgs.reshape(imgs.shape + (1,))
        mks = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, name)))
        mks = mks.reshape(mks.shape + (1,))
        # 归一化
        MIN, MAX = np.min(imgs), np.max(imgs)
        imgs = (imgs - MIN) / (MAX - MIN)

        hw = imgs.shape[1]
        if hw == 630:
            images = np.concatenate((images, imgs[:, 59:512 + 59, 59:512 + 59]), axis=0)
            masks = np.concatenate((masks, mks[:, 59:512 + 59, 59:512 + 59]), axis=0)
        else:
            images = np.concatenate((images, imgs), axis=0)
            masks = np.concatenate((masks, mks), axis=0)

    # print(images.shape, masks.shape)    # 929,512,512,1
    del imgs
    del mks
    raw_shape = images.shape
    # 对x进行归一化处理, （已处理）
    images = scaler.fit_transform(images.reshape(raw_shape[0], -1)).reshape(raw_shape)
    # print(images.shape)     # (929, 512, 512, 1)

    # shuffler
    images, masks = shuffler_array_inputs_labels(images, masks)
    split = 0.85
    split_index = int(np.floor(images.shape[0] * split))
    train_images = images[:split_index]
    train_masks = masks[:split_index]
    test_images = images[split_index:]
    test_masks = masks[split_index:]

    del images
    del masks

    # print(train_images.shape, test_images.shape)    # (789, 512, 512, 1) (140, 512, 512, 1)

    return train_images, train_masks, test_images, test_masks


def save():
    train_images, train_masks, test_images, test_masks = get_raw_example()
    save_example_to_tfrecord(images=train_images, labels=train_masks, save_tfrecord_dir=save_tfrecord_dir,
                             num_examples_one_tfrecord=num_examples_one_tfrecord,
                             name_prefix="train")
    save_example_to_tfrecord(images=test_images, labels=test_masks, save_tfrecord_dir=save_tfrecord_dir,
                             num_examples_one_tfrecord=num_examples_one_tfrecord,
                             name_prefix="test")

    return None


def read():
    filenames = os.listdir(save_tfrecord_dir)
    test_names = [os.path.join(save_tfrecord_dir, name) for name in filenames if name.startswith("test")]
    dataset = tfrecord_filenames_to_dataset(test_names, BATCH_SIZE=1)
    image, mask = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for i in range(5):
            img, label = sess.run([image, mask])
            print(img.shape, label.shape)
    return None


if __name__ == '__main__':
    read()
