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

# custom ones
from utils import load #@todo : add other paquet once the training is over
from preprocessing.preprocessing import greyscale_to_rgb, one_hot_y, split_train_test
from models.model import model_toto

if __name__ == '__main__':
    #load data
    DATA_FOLDER = '/Users/tianqi/PycharmProjects/kaggle_bengaliai_cv19/kaggle/bengaliai-cv19/' #@todo : A remplir
    train_df, test_df, class_map_df, sample_submission_df, train_img_df = load(DATA_FOLDER)
    #preprocess
    train_img_df = train_img_df[:10000].drop(['image_id'], axis=1)
    train_x_img = greyscale_to_rgb(train_img_df, h=137, w=236)
    y_gr, y_vd, y_cd = one_hot_y(y=train_df)
    # split
    x_training, x_test, y_gr_training, y_gr_test, \
    y_vd_training, y_vd_test, y_cd_training, y_cd_test \
        = split_train_test(train_x_img, y_gr, y_vd, y_cd, split=0.8)
    #train
    toto = model_toto(summary = False, plot_summary = False)
    history = toto.fit(x_training, [y_gr_training, y_vd_training, y_cd_training], epochs=20, batch_size=50,
                        validation_data=(x_test, [y_gr_test, y_vd_test, y_cd_test]))

    plot_result(history)



