#!/usr/bin/env python
# coding: utf-8

# In[13]:


import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
import h5py
from tensorflow.keras.utils import to_categorical


# In[14]:


# Load the .h5 file data
h5f = h5py.File("C:\\Users\\PMLS\\Downloads\\Autonomous_Vehicles_SVHN_single_grey1 (1).h5", 'r')
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]


# In[15]:


def img_lab(n):
    plt.figure(figsize=(n, 1))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(X_train[i], cmap='gray')
        plt.axis('off')
    plt.show()
    print('label for each of the above image:%s' % (y_train[0:n]))

img_lab(10)


# In[16]:


# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 1024, 1)
X_test = X_test.reshape(X_test.shape[0], 1024, 1)

# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[17]:


# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
print('The number of classes in this dataset are:', num_classes)


# In[18]:


# Define the CNN model
def cnn_model():
    model = Sequential()
    # First Convolutional Layer
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(1024, 1)))
    model.add(MaxPooling1D(pool_size=2))
    
    # Second Convolutional Layer
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Third Convolutional Layer
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Flatten the output before dense layers
    model.add(Flatten())
    
    # Fully Connected Dense Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    # Output Layer (for 10 classes)
    model.add(Dense(10, activation='softmax'))  
    return model

# Instantiate the model
model = cnn_model()


# In[19]:


# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
training_history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)


# In[20]:


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Loss:', scores[0])
print('Accuracy:', scores[1])


# In[21]:


# Plotting training and validation accuracy/loss
accuracy = training_history.history['accuracy']
val_accuracy = training_history.history['val_accuracy']
loss = training_history.history['loss']
val_loss = training_history.history['val_loss']
epochs = range(len(accuracy))

plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




