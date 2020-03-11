#load data
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from preprocessing.preprocessing import resize


def load(DATA_FOLDER, paquets = None):
    DATA_FOLDER = DATA_FOLDER
    train_df = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
    class_map_df = pd.read_csv(os.path.join(DATA_FOLDER, 'class_map.csv'))
    sample_submission_df = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv'))

    #load paquets
    if paquets is not None:
        img_id = []
        img = []
        for i in paquets :
            df = pd.read_parquet(DATA_FOLDER + '/%s'%i , engine='pyarrow')
            img_id.append(df['image_id'].values)
            img.append(df.drop('image_id', axis=1).values.astype(np.uint8))
        img_id = np.array(img_id)
        img = np.array(img)
    else : # local mode, take paquet 0
        df = pd.read_parquet(os.path.join(DATA_FOLDER,'train_image_data_0.parquet'), engine='pyarrow')
        img_id = df['image_id'].values
        img = df.drop('image_id', axis=1).values.astype(np.uint8)

    return train_df, test_df, class_map_df, sample_submission_df, img_id, img

def data_aug():
    pass
def preprocess_img(img, h, w, new_size = 64):

    s = new_size
    tmp_h = h
    tmp_w = w

    img_new = img.copy()

    if new_size is not None :
        tmp_h = h
        tmp_w = w
        img_new = resize(img_new, tmp_h, tmp_w, size=s)
        tmp_h = s
        tmp_w = s

    # normalize
    img_new = img_new / 255.0
    # reshape
    img_new = img_new.reshape(-1, tmp_h, tmp_w, 1)

    print('** Transformation on image **')
    print('size of image :', img.shape)
    print('original image height and width : ', h, ', ', w)
    print('Data augmentation : ', data_aug)
    print('New size : ', new_size )
    print('Normalize image')

    return img_new

def sub_plot_res(axes, x, y1, y2, l1='Training gr_accuracy', l2='Dev gr_accuracy', title='Accuracy for gr'):
    axes.plot(x, y1, label=l1)
    axes.plot(x, y2, label=l2)
    axes.set_title(title)
    axes.legend(loc='best')


def plot_result(history):
    try :
        # train result
        acc_gr = history.history['hgr_acc']
        acc_vd = history.history['hvd_acc']
        acc_cd = history.history['hcd_acc']
        # validation/dev result
        val_acc_gr = history.history['val_hgr_acc']
        val_acc_vd = history.history['val_hvd_acc']
        val_acc_cd = history.history['val_hcd_acc']

        # train loss
        loss_gr = history.history['hgr_loss']
        loss_vd = history.history['hvd_loss']
        loss_cd = history.history['hcd_loss']

        # val loss
        val_loss_gr = history.history['val_hgr_loss']
        val_loss_vd = history.history['val_hvd_loss']
        val_loss_cd = history.history['val_hcd_loss']
    except KeyError:
        # train result
        acc_gr = history.history['hgr_accuracy']
        acc_vd = history.history['hvd_accuracy']
        acc_cd = history.history['hcd_accuracy']
        # validation/dev result
        val_acc_gr = history.history['val_hgr_accuracy']
        val_acc_vd = history.history['val_hvd_accuracy']
        val_acc_cd = history.history['val_hcd_accuracy']

        # train loss
        loss_gr = history.history['hgr_loss']
        loss_vd = history.history['hvd_loss']
        loss_cd = history.history['hcd_loss']

        # val loss
        val_loss_gr = history.history['val_hgr_loss']
        val_loss_vd = history.history['val_hvd_loss']
        val_loss_cd = history.history['val_hcd_loss']

    epochs = range(len(acc_gr))

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    # axes[0,0].plot(epochs, acc_gr, label='Training gr_accuracy')
    # axes[0,0].plot(epochs, val_acc_gr, label='Training gr_accuracy')
    # axes[0,0].set_title('Accuracy for gr')
    # axes[0,0].legend(loc='best')
    sub_plot_res(axes[0, 0], epochs, acc_gr, val_acc_gr)
    sub_plot_res(axes[1, 0], epochs, acc_vd, val_acc_vd,
                 l1='Training vd_accuracy', l2='Dev vd_accuracy', title='Accuracy for vd')
    sub_plot_res(axes[2, 0], epochs, acc_cd, val_acc_cd,
                 l1='Training cd_accuracy', l2='Dev cd_accuracy', title='Accuracy for cd')

    sub_plot_res(axes[0, 1], epochs, loss_gr, val_loss_gr,
                 l1='Training gr loss', l2='Dev gr loss', title='Loss for gr')
    sub_plot_res(axes[1, 1], epochs, loss_vd, val_loss_vd,
                 l1='Training vd loss', l2='Dev vd loss', title='Loss for vd')
    sub_plot_res(axes[2, 1], epochs, loss_cd, val_loss_cd,
                 l1='Training cd loss', l2='Dev cd loss', title='Loss for cd')

