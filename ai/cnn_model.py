import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def build_cnn(input_shape):
    """
    Builds a 1D Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Shape of the input data (number of features, 1).

    Returns:
        tf.keras.models.Sequential: CNN model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),  # 1D Convolutional layer with 32 filters, kernel size 3, and ReLU activation
        tf.keras.layers.MaxPooling1D(pool_size=2),  # 1D Max Pooling layer with pool size 2
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),  # 1D Convolutional layer with 64 filters, kernel size 3, and ReLU activation
        tf.keras.layers.MaxPooling1D(pool_size=2),  # 1D Max Pooling layer with pool size 2
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        # 1D Convolutional layer with 64 filters, kernel size 3, and ReLU activation
        tf.keras.layers.MaxPooling1D(pool_size=2),  # 1D Max Pooling layer with pool size 2
        tf.keras.layers.Flatten(),  # Flatten the output from the convolutional layers
        tf.keras.layers.Dense(units=128, activation='relu'),  # Dense (fully connected) layer with 128 units and ReLU activation
        tf.keras.layers.Dropout(0.5),  # Dropout layer with a dropout rate of 0.5
        tf.keras.layers.Dense(units=3, activation='softmax')  # Output layer with 3 units (buy, sell, hold) and softmax activation
    ])

    model.compile(optimizer='adam',  # Compile the model with the Adam optimizer
                  loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy loss
                  metrics=['accuracy'])  # Track accuracy as a metric
    return model

def train_cnn(model, data):
    """
    Trains a CNN model on the provided data.

    Args:
        model (tf.keras.models.Sequential): CNN model to train.
        data (pandas.DataFrame): DataFrame containing features and a 'target' column.

    Returns:
        tuple: (loss, accuracy) of the model evaluated on the test set.
    """
    # Prepare the data for the CNN
    X = data.drop('Target', axis=1).values  # Extract features from the DataFrame
    y = data['Target'].values  # Extract target values from the DataFrame

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data with 20% for testing

    # Reshape the data for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshape training data for 1D convolution
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Reshape testing data for 1D convolution

    # Train the model
    model.fit(X_train, y_train,steps_per_epoch=20, epochs=500, batch_size=64, verbose=1)  # Train the model for 10 epochs with batch size 32

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)  # Evaluate the model on the test set
    print(f"Loss: {loss}")  # Print the loss
    print(f"Accuracy: {accuracy}")  # Print the accuracy

    return (loss, accuracy)  # Return the loss and accuracy

# Example Usage:
# Assuming 'data' is your pandas DataFrame
# input_shape = (data.shape[1] - 1, 1) # Calculate input shape
# cnn_model = build_cnn(input_shape)
# evaluation_metrics = train_cnn(cnn_model, data)

def main():
    # Define an example input shape
    input_shape = (10, 1)  # 10 features, 1 channel

    # Build the CNN model
    cnn_model = build_cnn(input_shape)

    # Print a summary of the model to verify its structure
    cnn_model.summary()

    # Verify that the model was built successfully
    print("CNN model built successfully!")

if __name__ == "__main__":
    main()