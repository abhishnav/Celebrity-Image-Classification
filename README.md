**CELEBRITY IMAGE CLASSIFIER**

**Summary of the Chosen Model, Training Process, and Critical Findings**

**Chosen Model:**

- Convolutional Neural Network (CNN) for image classification.
- Architecture includes convolutional layers, max-pooling layers, a flatten layer, and dense layers.
- Utilizes softmax activation for multi-class classification.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
```

**Training Process:**

- Dataset includes cropped images of celebrities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
- Preprocessing involves resizing and normalization.
- Model compiled using Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
- Training for 30 epochs, batch size of 128, and 10% validation split.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.1)
```

**Critical Findings:**

- Model achieved an accuracy of approximately 92.5% on the test set.
- Classification report provides precision, recall, and F1-score for each class, offering insights into performance on individual categories.
- The `make_prediction` function enables predictions on new images, enhancing practical utility.

Overall, the chosen CNN model demonstrates strong performance in classifying celebrity images, and its practical utility is enhanced through the `make_prediction` function for new images.
