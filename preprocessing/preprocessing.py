import numpy as np
import pandas as pd

def greyscale_to_rgb(df,h=137, w =236):# prepare X ## Need to replace df_i by train_img_df when working
    #img_df = df.drop(['image_id'], axis = 1)
    img_df = df.to_numpy()
    img_df = img_df/255.0

    # convert greyscale to rgb scale
    rgb_batch = np.repeat(img_df, 3, -1)    # since we're using transfer learning, hence need to convert to rgb

    x = rgb_batch.reshape(-1,h,w,3) #
    print(x.shape)
    return x

def one_hot_y (y):
  # expect y to have 3 cols : gr, vd, cd
  # gr = y[:,0]
  y_gr = pd.get_dummies(y['grapheme_root']).values
  y_cd = pd.get_dummies(y['consonant_diacritic']).values
  y_vd = pd.get_dummies(y['vowel_diacritic']).values
  return y_gr, y_vd, y_cd

def split_train_test(x, y_gr, y_vd, y_cd, split = 0.8): #split should be greater than 0.7
  indices = np.random.permutation(x.shape[0])
  tmp = int(x.shape[0] * split)
  training_idx, test_idx = indices[:tmp], indices[tmp:]
  x_training, x_test = x[training_idx], x[test_idx]
  y_gr_training, y_gr_test = y_gr[training_idx], y_gr[test_idx]
  y_vd_training, y_vd_test = y_vd[training_idx], y_vd[test_idx]
  y_cd_training, y_cd_test = y_cd[training_idx], y_cd[test_idx]
  return x_training, x_test,   y_gr_training, y_gr_test ,   y_vd_training, y_vd_test,   y_cd_training, y_cd_test

