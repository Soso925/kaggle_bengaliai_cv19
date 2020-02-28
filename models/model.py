import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model



def model_toto (summary = False, plot_summary = False) :
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

def model_toto2 (summary = False, plot_summary = False) :
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

def model_toto3 (summary = False, plot_summary = False) :
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

