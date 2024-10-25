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
    x = Dense(
        units=hp.Int('dense_units2', min_value=32, max_value=128, step=32),
        activation='relu'
    )(x)
    
    # Output Layer
    output = Dense(3, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=output)
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Model training
MAX_EPOCHS = 100
batch_size=32


stopping = keras.callbacks.EarlyStopping(patience=20)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=0.001 * 0.001)
checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)


model.fit(
    train_x, train_y,
    batch_size=batch_size,
    epochs=MAX_EPOCHS,
    validation_data=(val_x, val_y),
    callbacks=[checkpointer, reduce_lr, stopping])

best_model_path = get_filename_for_saving(save_dir)
best_model = keras.models.load_model(best_model_path)


# Hyper parameter tunning
patience = 5
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
model_checkpoint = ModelCheckpoint('RESULTS/model_.keras', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

# Setup Bayesian Optimization tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='hyperparam_opt',
    project_name='ectopic_detection_v8'
)


tuner.search(t_x, t_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping, model_checkpoint])


# Load the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save model
best_model.save('RESULTS/model_.keras')

# Get the best hyperparameters and print them
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:", best_hps.values)

metric_results = calculate_metrics_with_ci(val_y, val_pred_classes, num_classes=3, n_bootstrap=1000)

# Printing results
for metric, result in metric_results.items():
    print(f"{metric}: Mean = {result['mean']}, 95% CI = {result['95% CI']}")



best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:", best_hps.values)

metric_results = calculate_metrics_with_ci(val_y, val_pred_classes, num_classes=3, n_bootstrap=1000)

# Printing results
for metric, result in metric_results.items():
    print(f"{metric}: Mean = {result['mean']}, 95% CI = {result['95% CI']}")
