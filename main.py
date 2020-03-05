# Load std packages
import os
import zipfile
import pandas as pd
import numpy as np
# import PIL.Image
import time
import seaborn as sns
import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook
# %matplotlib inline
import cv2
from tqdm.auto import tqdm

# custom ones
from utils import load, preprocess_img, plot_result  # @todo : add other paquet once the training is over
from preprocessing.preprocessing import one_hot_y, split_train_test
from models.model import history_save  # model_v4,
from include import RUN_LOCAL

if __name__ == '__main__':
    #global RUN_LOCAL  # Add var RUN_LOCAL
    #RUN_LOCAL = True
    # load data
    DATA_FOLDER = '/Users/tianqi/PycharmProjects/kaggle_bengaliai_cv19/data/bengaliai-cv19'  # @todo : A remplir
    train_df, test_df, class_map_df, sample_submission_df, img_id, img = load(DATA_FOLDER)
    # preprocess
    train_x_img = preprocess_img(img, h=137, w=236, data_aug=None, new_size=64)
    # get y
    y_gr, y_vd, y_cd = one_hot_y(y=train_df)
    # split
    x_training, x_test, y_gr_training, y_gr_test, \
    y_vd_training, y_vd_test, y_cd_training, y_cd_test \
        = split_train_test(train_x_img, y_gr, y_vd, y_cd, split=0.8)
    # train

    if RUN_LOCAL == 1: # make a custom model
        import tensorflow as tf
        # fix plot_model #@todo : add it in model.py
        from tensorflow.keras.utils import plot_model

        def model_v4(input_h, input_w, summary=False, plot_summary=False):
            inputs = tf.keras.Input(shape=(input_h, input_w, 1))
            model = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu',
                                           input_shape=(input_h, input_w, 1))(inputs)
            model = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.BatchNormalization(momentum=0.15)(model)
            model = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(model)
            model = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Dropout(rate=0.2)(model)

            model = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.BatchNormalization(momentum=0.15)(model)
            model = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(model)
            model = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.BatchNormalization(momentum=0.15)(model)
            model = tf.keras.layers.Dropout(rate=0.3)(model)

            model = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.BatchNormalization(momentum=0.15)(model)
            model = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(model)
            model = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.BatchNormalization(momentum=0.15)(model)
            model = tf.keras.layers.Dropout(rate=0.3)(model)

            model = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.BatchNormalization(momentum=0.15)(model)
            model = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(model)
            model = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
            model = tf.keras.layers.BatchNormalization(momentum=0.15)(model)
            model = tf.keras.layers.Dropout(rate=0.2)(model)
            model = tf.keras.layers.Flatten()(model)
            model = tf.keras.layers.Dense(1024, activation="relu")(model)
            model = tf.keras.layers.Dropout(rate=0.3)(model)
            x = tf.keras.layers.Dense(512, activation="relu")(model)

            # Add a final sigmoid layer for classification
            head_gr = tf.keras.layers.Dense(10, activation='softmax', name='hgr')(x)
            head_vd = tf.keras.layers.Dense(11, activation='softmax', name='hvd')(x)
            head_cd = tf.keras.layers.Dense(7, activation='softmax', name='hcd')(x)

            model = tf.keras.Model(inputs, outputs=[head_gr, head_vd, head_cd])

            model.compile(optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            if summary == True:
                model.summary()
            if plot_summary == True:
                plot_model(model, to_file='model.png')

            return model


    vv = model_v4(input_h=64, input_w=64, plot_summary=False)
    # Get smaller dataset for training
    x_training, y_gr_training, y_vd_training, y_cd_training  = x_training[:1000], y_gr_training[:1000], y_vd_training[:1000], y_cd_training[:1000]
    history = vv.fit(x_training, [y_gr_training, y_vd_training, y_cd_training], epochs=3, batch_size=500, \
                     validation_data=(x_test, [y_gr_test, y_vd_test, y_cd_test]))

    plot_result(history)
    history_save(history, filename='toto')





