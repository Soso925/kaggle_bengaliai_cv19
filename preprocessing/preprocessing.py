import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

def greyscale_to_rgb(df,h=137, w =236):# prepare X

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

def normalize_img(df, h, w):# prepare X
    img_df = df.to_numpy()
    x = img_df/255.0
    x = x.reshape(-1, h, w, 1)
    print(x.shape)
    return x

# resize
# courtesy of kaggle notebooks
def resize(df, input_h, input_w, size=64,  need_progress_bar=True):
    resized = {}
    resize_size=size
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image=df.loc[df.index[i]].values.reshape(input_h,input_w)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image=df.loc[df.index[i]].values.reshape(input_h,input_w)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized