# import tensorflow as tf
# from tensorflow.keras import layers, models
# import numpy as np
# from tensorflow.keras.callbacks import EarlyStopping
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# data = pd.DataFrame({
#     'Concatenate': [16000, 160000, 160000, 400000]
# })

# labels = pd.DataFrame({
#     'Peaks': ['small_peak', 'Medium1', 'Medium2', 'Massive']
# })

# data = data.astype('float32')

# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels['Peaks'])

# train_data, test_data, train_labels, test_labels = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# model = models.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),  # Input shape is (number of features,)
#     layers.Dense(64, activation='relu'),
#     layers.Dense(len(np.unique(labels_encoded)), activation='softmax')  # Output layer based on number of classes
# ])

# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Compiling
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Training
# model.fit(train_data, train_labels, 
#           epochs=8, 
#           batch_size=64,
#           validation_split=0.2, 
#           callbacks=[early_stop])

# # Evaluate the model on test data
# test_loss, test_acc = model.evaluate(test_data, test_labels)
# print(f'Test accuracy: {(test_acc*100):.2f}%')

# # Prediction
# predictions = model.predict(test_data)

# # Example prediction
# predicted_label = np.argmax(predictions[0])
# true_label = test_labels[0]

# print(f'Predicted label: {label_encoder.inverse_transform([predicted_label])[0]}')
# print(f'True label: {label_encoder.inverse_transform([true_label])[0]}')


import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Create sample data
data = pd.DataFrame({
    'Concatenate': [16000, 160000, 160000, 400000]
})

labels = pd.DataFrame({
    'Peaks': ['small_peak', 'Medium1', 'Medium2', 'Massive']
})

data = data.astype('float32')

# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels['Peaks'])

# Split data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))

# Model creation
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],)),
    layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Compiling
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(train_data, train_labels,
          epochs=8,
          batch_size=64,
          validation_split=0.2,
          callbacks=[early_stop],
          class_weight=class_weights_dict)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {(test_acc*100):.2f}%')

# Prediction
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# Example predictions
for i in range(len(test_data)):
    print(f"Sample {i}: Predicted label: {label_encoder.inverse_transform([predicted_classes[i]])[0]}, True label: {label_encoder.inverse_transform([test_labels[i]])[0]}")



import matplotlib.pyplot as plt

# Plot predictions vs true labels
plt.figure(figsize=(10, 6))
plt.scatter(range(len(test_labels)), test_labels, color='blue', label='True Labels')
plt.scatter(range(len(predicted_classes)), predicted_classes, color='red', label='Predicted Labels')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.title('True vs Predicted Labels')
plt.show()
