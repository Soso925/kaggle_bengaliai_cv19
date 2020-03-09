# custom ones
from utils import load, preprocess_img, plot_result  # @todo : add other paquet once the training is over
from preprocessing.preprocessing import one_hot_y, split_train_test
from models.model import history_save, model_v4

if __name__ == '__main__':

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

    vv = model_v4(input_h=64, input_w=64, plot_summary=False)
    # Get smaller dataset for training
    x_training, y_gr_training, y_vd_training, y_cd_training  = x_training[:100], y_gr_training[:100], y_vd_training[:100], y_cd_training[:100]
    history = vv.fit(x_training, [y_gr_training, y_vd_training, y_cd_training], epochs=3, batch_size=50, \
                     validation_data=(x_test, [y_gr_test, y_vd_test, y_cd_test]))

    plot_result(history)
    history_save(history, filename='toto')





