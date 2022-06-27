import os
import click

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def load_data(data_dir="data/COVID-19_Radiography_Dataset"):
    image_data_path = "data/COVID-19_Radiography_Dataset/image_data.npy"
    click.secho("Loading data...", fg="green")
    if os.path.exists(image_data_path):
        image_list = np.load(image_data_path, allow_pickle=True)
        labels = {"Normal": 0, "COVID": 1}
        label_list = []
        for label, value in labels.items():
            for _ in os.listdir(data_dir + "/" + label + "/images"):
                label_list.append(value)
        if image_list.shape[0] != len(label_list):
            raise ValueError
    else:
        labels = {"Normal": 0, "COVID": 1}
        image_path_list, label_list = [], []
        for label, value in labels.items():
            for file in os.listdir(data_dir + "/" + label + "/images"):
                image_path_list.append(data_dir + "/{}/images/{}".format(label, file))
                label_list.append(value)
        image_list = np.array(list(map(lambda x: read_image(x), image_path_list)))
        np.save(image_data_path, image_list)

    return image_list, label_list


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (70, 70)) / 255.0
    return np.asarray(image)


def train_val_test_split(image_list, label_list, ratio=(0.2, 0.2), test_only=False):
    x = np.asarray(image_list).astype("float32")
    y = to_categorical(label_list, num_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=ratio[0], random_state=42
    )
    if test_only:
        return x_test, y_test

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=ratio[1], random_state=42
    )
    return x_train, x_val, x_test, y_train, y_val, y_test
