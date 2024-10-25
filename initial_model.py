import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, Softmax, Add, Flatten, Activation, Dropout, AveragePooling1D
from keras.models import Model
from keras import optimizers, losses, activations
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

def build_model(hp):
    inp = Input(shape=(127, 1))
    
    # Convolutional Block 1
    x = Conv1D(
        filters=hp.Int('filters1', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('kernel_size1', values=[3, 5]),
        activation='relu',
        padding='same'
    )(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Convolutional Block 2
    x = Conv1D(
        filters=hp.Int('filters2', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size2', values=[3, 5]),
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Convolutional Block 3
    x = Conv1D(
        filters=hp.Int('filters3', min_value=64, max_value=256, step=64),
        kernel_size=hp.Choice('kernel_size3', values=[3, 5]),
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Dense Layer
    x = Flatten()(x)
    x = Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    )(x)
    
    # Output Layer
    output = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
