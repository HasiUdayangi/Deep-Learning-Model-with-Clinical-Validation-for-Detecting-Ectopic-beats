import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, Softmax, Add, Flatten, Activation, Dropout, AveragePooling1D
from keras.models import Model
from keras import optimizers, losses, activations
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

def network():
    inp = Input(shape=(127, 1))
    
    # First Convolutional Block
    conv1_1 = Conv1D(16, 5, padding='same')(inp)
    conv1_1 = BatchNormalization()(conv1_1)
    A1_1 = Activation("relu")(conv1_1)
    conv1_2 = Conv1D(16, 5, padding='same')(A1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    A1_2 = Activation("relu")(conv1_2)
    pool1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(A1_2)
    avg1 = AveragePooling1D(pool_size=2, strides=2, padding='same')(pool1)
    
    # Second Convolutional Block
    conv2_1 = Conv1D(32, 3, padding='same')(avg1)
    conv2_1 = BatchNormalization()(conv2_1)
    A2_1 = Activation("relu")(conv2_1)
    conv2_2 = Conv1D(32, 3, padding='same')(A2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    A2_2 = Activation("relu")(conv2_2)
    pool2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(A2_2)
    avg2 = AveragePooling1D(pool_size=2, strides=2, padding='same')(pool2)
    
    # Third Convolutional Block
    conv3_1 = Conv1D(64, 3, padding='same')(avg2)
    conv3_1 = BatchNormalization()(conv3_1)
    A3_1 = Activation("relu")(conv3_1)
    conv3_2 = Conv1D(64, 3, padding='same')(A3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    A3_2 = Activation("relu")(conv3_2)
    pool3 = MaxPooling1D(pool_size=2, strides=3, padding='same')(A3_2)
    avg3 = AveragePooling1D(pool_size=2, strides=2, padding='same')(pool3)
    
    # Dense Layers
    flatten = Flatten()(avg3)
    dense_end1 = Dense(64, activation='relu')(flatten)
    dense_end1 = BatchNormalization()(dense_end1)
    dense_end2 = Dense(32, activation='relu')(dense_end1)
    main_output = Dense(3, activation='softmax')(dense_end2)
    
    model = Model(inputs=inp, outputs=main_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model
