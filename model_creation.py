import tensorflow as tf

def create_deep_neural_network(X_train):
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
            tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model