import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def build_model(input_shape=(53,), num_classes=10):
    """
    Build and compile the ANN model matching FYP_ANN_3 architecture.
    """
    # Input layer: 53 binary feature inputs
    input_layer = Input(shape=input_shape, name="input_layer")

    # First hidden layer: 16 neurons, ReLU activation, 0.2 Dropout
    x = Dense(16, activation='relu', name="dense_1")(input_layer)
    x = Dropout(0.2, name="dropout_1")(x)

    # Second hidden layer: 16 neurons, ReLU activation, 0.2 Dropout
    x = Dense(16, activation='relu', name="dense_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)

    # Third hidden layer: 16 neurons, ReLU activation, 0.2 Dropout
    x = Dense(16, activation='relu', name="dense_3")(x)
    x = Dropout(0.2, name="dropout_3")(x)

    # Output layer: 10 disease classes with softmax activation
    output_layer = Dense(num_classes, activation='softmax', name="output_layer")(x)

    # Build model using Functional API
    model = Model(inputs=input_layer, outputs=output_layer, name="FYP_ANN_3")

    # Compile model with Adam optimizer and sparse_categorical_crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
