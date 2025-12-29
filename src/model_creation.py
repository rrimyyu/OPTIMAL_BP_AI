import tensorflow as tf
from typing import Iterable, Tuple, Optional

def create_deep_neural_network(
    n_features: int,
    l2: float = 0.01,
    dropout: float = 0.5,
    optimizer: str = "adam",
):
    """
    Fully-connected neural network for binary classification.

    Parameters
    ----------
    n_features : int
        Number of input features.
    l2 : float
        L2 regularization strength.
    dropout : float
        Dropout rate.
    optimizer : str
        Keras optimizer name.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                input_shape=(n_features,),
            ),
            tf.keras.layers.Dropout(dropout),

            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2),
            ),
            tf.keras.layers.Dropout(dropout),

            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2),
            ),

            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model