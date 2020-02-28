#load data
import os
import pandas as pd
import pip

def get_installed_packages():
    installed_packages = pip.get_installed_distributions()
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                      for i in installed_packages])
    print(installed_packages_list)

def load(DATA_FOLDER):
    DATA_FOLDER = DATA_FOLDER
    train_df = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
    class_map_df = pd.read_csv(os.path.join(DATA_FOLDER, 'class_map.csv'))
    sample_submission_df = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv'))

    # Load all train data image
    df_list = []
    for i in range(1): # For now just some small data
        tmp_name = 'train_image_data_' + str(i) + '.parquet'
        df_i =  pd.read_parquet(os.path.join(DATA_FOLDER,tmp_name))
        print(f'shape of {i} parquet : {df_i.shape}')
        df_list.append(df_i)

    train_img_df = pd.concat(df_list)


    return train_df, test_df, class_map_df, sample_submission_df, train_img_df


def sub_plot_res(axes, x, y1, y2, l1='Training gr_accuracy', l2='Dev gr_accuracy', title='Accuracy for gr'):
    axes.plot(x, y1, label=l1)
    axes.plot(x, y2, label=l2)
    axes.set_title(title)
    axes.legend(loc='best')


def plot_result(history):
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

