import tensorflow as tf
from typing import Iterable, Tuple, Optional


def create_deep_neural_network(
    X_train,
    compile_model: bool = True,
    seed: Optional[int] = None,
):

    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            32, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            input_shape=(X_train.shape[1],)
        ),
        tf.keras.layers.Dense(
            16, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.Dense(
            4, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    if compile_model:
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy']
        )

    return model


def create_deep_neural_network_stable(
    X_train,
    hidden_units: Tuple[int, ...] = (64, 32, 16),
    l2: float = 1e-4,
    dropout: float = 0.2,
    use_batchnorm: bool = True,
    learning_rate: float = 1e-3,
    seed: Optional[int] = 42,
    compile_model: bool = True,
):
    """
    Build a compact, configurable DNN for binary classification.

    Args
    ----
    X_train: pd.DataFrame or np.ndarray
        Training features; only used to infer input dimension.
    hidden_units: tuple
        Number of units per hidden layer.
    l2: float
        L2 regularization strength applied to Dense kernels.
    dropout: float
        Dropout rate applied after activations (0 disables if set to 0).
    use_batchnorm: bool
        Whether to insert BatchNorm before activations.
    learning_rate: float
        Optimizer learning rate.
    seed: int | None
        Random seed for reproducibility (None to disable).
    compile_model: bool
        If True, compile with Adam + binary cross-entropy + metrics.

    Returns
    -------
    tf.keras.Model
        A compiled (or uncompiled) Keras model ready to fit.
    """

    input_dim = X_train.shape[1]
    reg = tf.keras.regularizers.l2(l2)

    layers = [tf.keras.layers.Input(shape=(input_dim,))]

    # Hidden blocks: Dense -> (BatchNorm) -> ReLU -> (Dropout)
    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(
            units,
            activation=None,                  # BN -> Activation pattern
            kernel_regularizer=reg,
            kernel_initializer="he_normal",
            use_bias=not use_batchnorm
        ))
        if use_batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))
        if dropout and dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))

    # Output
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers, name="binary_dnn")

    if compile_model:
        metrics = [
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=metrics,
        )

    return model


def default_callbacks(
    out_dir: str = "results/models",
    monitor: str = "val_auc_roc",
    patience: int = 20,
    min_delta: float = 1e-4
):
    """
    Suggested callbacks for stable training.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=patience, restore_best_weights=True, min_delta=min_delta
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.5, patience=max(5, patience // 3), min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{out_dir}/best_binary_dnn.keras",
            monitor=monitor, mode="max", save_best_only=True
        ),
    ]
