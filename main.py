# Load std packages
import os
import zipfile
import pandas as pd
import numpy as np
#import PIL.Image
import time
import seaborn as sns
import matplotlib.pyplot as plt
#from tqdm import tqdm_notebook
#%matplotlib inline
import cv2
from tqdm.auto import tqdm

# custom ones
from utils import load, plot_result #@todo : add other paquet once the training is over
from preprocessing.preprocessing import greyscale_to_rgb, one_hot_y, split_train_test, resize, normalize_img
from models.model import model_toto, model_toto2, model_toto3, model_v4, history_save

if __name__ == '__main__':
    #load data
    #DATA_FOLDER = '/Users/tianqi/PycharmProjects/kaggle_bengaliai_cv19/kaggle/bengaliai-cv19/'  # @todo : A remplir
    DATA_FOLDER = '/Users/Alain/GitHub/kaggle_bengaliai_cv19/data/' #@todo : A remplir
    train_df, test_df, class_map_df, sample_submission_df, train_img_df = load(DATA_FOLDER)
    #preprocess
    train_img_df = train_img_df[:1000].drop(['image_id'], axis=1) #@todo : to modify when training when whold dataset
    #train_x_img = greyscale_to_rgb(train_img_df, h=137, w=236)
    # resize x
    train_x_img = resize(train_img_df, 137, 236, size=64, need_progress_bar=True)
    # normalize x
    train_x_img = normalize_img(train_x_img, 64, 64)
    #get y
    y_gr, y_vd, y_cd = one_hot_y(y=train_df)
    # split
    x_training, x_test, y_gr_training, y_gr_test, \
    y_vd_training, y_vd_test, y_cd_training, y_cd_test \
        = split_train_test(train_x_img, y_gr, y_vd, y_cd, split=0.8)
    #train
    #toto = model_toto3()
    #history = toto.fit(x_training, [y_gr_training, y_vd_training, y_cd_training], epochs=20, batch_size=50,  validation_data=(x_test, [y_gr_test, y_vd_test, y_cd_test]))
    vv = model_v4(input_h=64, input_w=64, plot_summary=True)
    history = vv.fit(x_training, [y_gr_training, y_vd_training, y_cd_training], epochs=100, batch_size=50, \
                     validation_data=(x_test, [y_gr_test, y_vd_test, y_cd_test]))

    plot_result(history)
    history_save(history, filename= 'toto')




