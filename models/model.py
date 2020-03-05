import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pickle
import json
from include import RUN_LOCAL

def model_save(model, filename, weight = False):
    #model is generally huge. Save the model wisely
    if weight == True :
        model.save( filename + ".h5")
        print('saved the model for you in .h5 format' )

    else :
        model_json = model.to_json()
        with open(filename + ".json", "w") as json_file:
            json_file.write(model_json)
        print('saved the model for you in .json format')

def history_save(fit, filename):
    #history is always true
    with open(filename, 'wb') as file_pi:
        pickle.dump(fit.history, file_pi)
    print('saved the history for you in pickle format')

def model_toto(summary = False, plot_summary = False) :
    inception = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                                                            input_shape=(137, 236, 3), pooling='max', classes=512)
    for layer in inception.layers:
        layer.trainable = False

    last_layer = inception.get_layer('mixed2')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    head_gr = tf.keras.layers.Dense(168, activation='softmax', name='hgr')(x)
    head_vd = tf.keras.layers.Dense(11, activation='softmax', name='hvd')(x)
    head_cd = tf.keras.layers.Dense(7, activation='softmax', name='hcd')(x)

    model = tf.keras.Model(inception.input, outputs=[head_gr, head_vd, head_cd])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if summary == True :
        model.summary()
    if plot_summary == True :
        plot_model(model, to_file='model.png')
    return model

def model_toto2(summary = False, plot_summary = False) :
    inception = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                                                            input_shape=(137, 236, 3), pooling='max', classes=512)
    for layer in inception.layers:
        layer.trainable = False

    last_layer = inception.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    head_gr = tf.keras.layers.Dense(168, activation='softmax', name='hgr')(x)
    head_vd = tf.keras.layers.Dense(11, activation='softmax', name='hvd')(x)
    head_cd = tf.keras.layers.Dense(7, activation='softmax', name='hcd')(x)

    model = tf.keras.Model(inception.input, outputs=[head_gr, head_vd, head_cd])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if summary == True :
        model.summary()
    if plot_summary == True :
        plot_model(model, to_file='model.png')
    return model

def model_toto3(summary = False, plot_summary = False) :
    inception = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights=None, input_tensor=None,
                                                            input_shape=(137, 236, 3), pooling='max', classes=512)

    last_layer = inception.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    head_gr = tf.keras.layers.Dense(168, activation='softmax', name='hgr')(x)
    head_vd = tf.keras.layers.Dense(11, activation='softmax', name='hvd')(x)
    head_cd = tf.keras.layers.Dense(7, activation='softmax', name='hcd')(x)

    model = tf.keras.Model(inception.input, outputs=[head_gr, head_vd, head_cd])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if summary == True :
        model.summary()
    if plot_summary == True :
        plot_model(model, to_file='model.png')
    return model


def model_v4(input_h, input_w, summary=False, plot_summary=False):
    global RUN_LOCAL
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
    if RUN_LOCAL !=1 :
        head_gr = tf.keras.layers.Dense(168, activation='softmax', name='hgr')(x)
    else :
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
