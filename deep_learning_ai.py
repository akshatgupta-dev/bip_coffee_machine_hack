
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Simulated dataset
data = pd.DataFrame({
    'Power Low Sum': [170861.1997, 10749.59985, 16431.60022, 453241.1995, 
                      160844.7998, 160614.2009, 8809.199707, 158447.9995, 427644.397],
    'Cup Type': ['double_small', 'single_small', 'single_small', 'double_large', 
                 'single_large', 'single_large', 'single_small', 'double_small', 'double_large']
})

# Encode labels
label_encoder = LabelEncoder()
data['Cup Type'] = label_encoder.fit_transform(data['Cup Type'])

# Split features and labels
X = data[['Power Low Sum']].values
y = data['Cup Type'].values      

# Randomly split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model with more neurons and layers
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Increased neurons
    layers.Dense(64, activation='relu'),  # Added another hidden layer
    layers.Dense(32, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')  # Output layer
])

# Compile the model with a smaller learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model for more epochs to ensure memorization
model.fit(X_train, y_train, epochs=1000, batch_size=2, verbose=0)

# Shuffle the test data for randomness
indices = np.arange(X_test.shape[0])
np.random.shuffle(indices)
X_test = X_test[indices]
y_test = y_test[indices]

# Evaluate on the shuffled test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {(test_acc * 100):.2f}%')

# Make predictions on the shuffled test set
predictions = model.predict(X_test)

# Convert predictions to label indices
predicted_classes = np.argmax(predictions, axis=1)

# Print true labels and predicted labels to verify correctness
print(f"True labels: {y_test}")
print(f"Predicted labels: {predicted_classes}")
