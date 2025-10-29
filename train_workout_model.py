# train_workout_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Step 1: Load the dataset
data = pd.read_csv("workout_data_large.csv")  # Use the larger, balanced CSV

# Step 2: Encode goal labels
goal_encoder = LabelEncoder()
data['goal_encoded'] = goal_encoder.fit_transform(data['goal'])

# Step 3: One-hot encode workout_type
workout_type_encoder = OneHotEncoder(sparse_output=False)
workout_type_encoded = workout_type_encoder.fit_transform(data[['workout_type']])

# Step 4: Combine features
X = np.hstack([data[['avg_calories']].values, workout_type_encoded])
y = data['goal_encoded'].values

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # input dimension = avg_calories + one-hot workout types
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(goal_encoder.classes_), activation='softmax')  # number of goals
])

# Step 7: Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 8: Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Step 9: Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")

# Step 10: Save model and encoders
model.save("workout_model.h5")
np.save("goal_classes.npy", goal_encoder.classes_)
np.save("workout_type_classes.npy", workout_type_encoder.categories_[0])

print("✅ Model and encoders saved successfully!")
