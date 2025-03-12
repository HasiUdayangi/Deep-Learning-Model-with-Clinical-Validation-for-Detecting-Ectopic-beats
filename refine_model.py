
"""
Refine Pre-Trained Ectopic Beat Detection Model with Hyperparameter Tuning

The script:
  - Loads and preprocesses the dataset.
  - Loads the pre-trained model.
  - Freezes the model and unfreezes the last N layers (where N is a tunable hyperparameter).
  - Compiles the model with a tunable learning rate.
  - Uses RandomSearch to find the best hyperparameters based on validation accuracy.
  - Trains and evaluates the refined model.
  - Saves the best refined model.
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import kerastuner as kt  # Ensure keras-tuner is installed: pip install keras-tuner


# Load feature data and labels
x_GCUH = pd.read_csv('X.csv')
y_GCUH = pd.read_csv('y.csv')

# Replace string labels with numeric values: 'NOTEB'->0, 'VEB'->1, 'SVEB'->2
y_GCUH = y_GCUH.replace(['NOTEB', 'VEB', 'SVEB'], [0, 1, 2])

# Convert data to numpy arrays
X = x_GCUH.values
y = y_GCUH.values.flatten()

# One-hot encode labels (3 classes)
y_cat = to_categorical(y, num_classes=3)

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)


def build_refined_model(hp):
    # Path to the pre-trained model
    model_path = ''
    model = load_model(model_path)
    
    # Total number of layers in the model
    num_layers = len(model.layers)
    
    # Hyperparameter: number of layers to unfreeze for fine-tuning (from 1 up to num_layers)
    num_unfreeze = hp.Int('num_unfreeze', min_value=1, max_value=num_layers, step=1, default=1)
    
    # Freeze all layers first
    for layer in model.layers:
        layer.trainable = False
    # Unfreeze the last 'num_unfreeze' layers
    for layer in model.layers[-num_unfreeze:]:
        layer.trainable = True
    
    # Hyperparameter: learning rate
    lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG', default=1e-4)
    optimizer = Adam(learning_rate=lr)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


tuner = kt.RandomSearch(
    build_refined_model,
    objective='val_accuracy',
    max_trials=10,           
    executions_per_trial=2,
    directory='refine_tuning',
    project_name='ectopic_refinement'
)

print("Starting hyperparameter tuning...")
tuner.search(x_train, y_train,
             epochs=100,            # Fewer epochs may be used for tuning
             validation_data=(x_val, y_val))

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters found:")
print(f"  num_unfreeze: {best_hps.get('num_unfreeze')}")
print(f"  learning_rate: {best_hps.get('learning_rate')}")


model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train,
                    epochs=100,            # Increase epochs for final training
                    validation_data=(x_val, y_val),
                    batch_size=32)

val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Save the best refined model
model.save('')
print("Refined model saved as ''.")
