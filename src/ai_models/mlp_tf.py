"""TensorFlow/Keras MLP engine — unified for CoAX and CoXAM.

Same two-hidden-layer architecture as the PyTorch MLPEngine.
Uses tf.keras so it is compatible with:
  - SHAP DeepExplainer  (via .keras_model property)
  - tf-explain / integrated-gradients (via GradientTape on .keras_model)
  - LIME  (via predict_proba callable)

cognitive_agent='coax'  enables forward_logits (GradientTape logit access).
cognitive_agent='coxam' standard probability output only.

Weight files (.weights.h5) are read from / written to:
    src/ai_models/<cognitive_agent>/mlp/<file_name>
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent


class _BaseEngine(ABC):
    gradient_based: bool = True

    @abstractmethod
    def predict(self, X_dense) -> np.ndarray: ...
    @abstractmethod
    def train(self, X, y, **kw): ...
    @abstractmethod
    def save(self, file_name: str | None = None): ...
    @abstractmethod
    def load(self, file_name: str): ...


class TFMLPEngine(_BaseEngine):
    """
    Parameters
    ----------
    cognitive_agent : 'coax' | 'coxam'
        'coax'  — builds a separate logit-output sub-model exposed as
                  .logit_model for gradient-based XAI (Integrated Gradients).
                  Weights from coax/mlp/.
        'coxam' — probability output only; weights from coxam/mlp/.
    input_dim, num_classes, hidden_dimension, dropout_rate :
        Architecture hyperparameters matching the PyTorch version.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dimension: int = 50, dropout_rate: float = 0.0,
                 cognitive_agent: str = 'coxam', **_):
        import tensorflow as tf

        self.gradient_based = True
        self._agent = cognitive_agent
        self._weight_dir = _ROOT / cognitive_agent / 'mlp'

        inp = tf.keras.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(hidden_dimension, activation='relu',
                                   kernel_initializer='glorot_uniform')(inp)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(hidden_dimension, activation='relu',
                                   kernel_initializer='glorot_uniform')(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        logits = tf.keras.layers.Dense(num_classes,
                                        kernel_initializer='glorot_uniform',
                                        name='logits')(x)
        probs = tf.keras.layers.Softmax(name='probs')(logits)

        self.model = tf.keras.Model(inputs=inp, outputs=probs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        # coax-only: sub-model that outputs raw logits for GradientTape XAI
        if cognitive_agent == 'coax':
            self.logit_model = tf.keras.Model(inputs=inp, outputs=logits)

    # coax-only: raw logits for Integrated Gradients / GradientTape attribution
    def forward_logits(self, X_dense):
        if self._agent != 'coax':
            raise AttributeError(
                "forward_logits is only available for cognitive_agent='coax'"
            )
        X = np.atleast_2d(X_dense).astype(np.float32)
        return self.logit_model(X)

    def predict(self, X_dense) -> np.ndarray:
        return self.model.predict(
            np.atleast_2d(X_dense).astype(np.float32), verbose=0
        )

    def predict_proba(self, X_dense) -> np.ndarray:
        return self.predict(X_dense)

    def train(self, X, y, X_dev=None, y_dev=None, epochs=300, batch_size=1000):
        import tensorflow as tf

        callbacks = []
        val_data = None
        ckpt = str(self._weight_dir / '_best.weights.h5')

        if X_dev is not None and y_dev is not None:
            val_data = (X_dev.astype(np.float32), y_dev)
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                ckpt, monitor='val_accuracy',
                save_best_only=True, save_weights_only=True, verbose=0,
            ))

        history = self.model.fit(
            X.astype(np.float32), y,
            validation_data=val_data,
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1,
        )

        from pathlib import Path
        if callbacks and Path(ckpt).exists():
            self.model.load_weights(ckpt)
            best = max(history.history.get('val_accuracy', [0]))
            print(f"Reverted to best val_accuracy {best:.4f}")

        return history

    def evaluate(self, X, y) -> float:
        preds = self.predict(X)
        return float(np.mean(np.argmax(preds, axis=1) == y))

    def save(self, file_name: str | None = None):
        name = file_name or 'tf_model_weights.weights.h5'
        path = self._weight_dir / name
        self.model.save_weights(str(path))
        print(f"[tf-mlp/{self._agent}] saved → {path}")

    def load(self, file_name: str):
        path = self._weight_dir / file_name
        self.model.load_weights(str(path))
        print(f"[tf-mlp/{self._agent}] loaded ← {path}")

    @property
    def keras_model(self):
        """Expose tf.keras.Model for SHAP DeepExplainer and tf-explain."""
        return self.model
