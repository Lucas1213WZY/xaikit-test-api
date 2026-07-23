"""MLP model implementations — PyTorch and TensorFlow/Keras.

cognitive_agent='coax'  enables forward_logits / forward_logits_or_probs
                        (required by Captum-based gradient attribution).
cognitive_agent='coxam' omits those gradient hooks (not needed for CoXAM).

Weight files are read from / written to:
    assets/model_weights/mlp/<cognitive_agent>/<file_name>
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..paths import model_weights_dir

_BOOST_DEFAULTS = {'coax': 10, 'coxam': 50}


# ---------------------------------------------------------------------------
# Thin abstract base (shared by PyTorch and TF engines)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dimension=50,
                 dropout_rate=0, device_id=-1, cognitive_agent='coxam', **_):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device_id = device_id
        self._coax = (cognitive_agent == 'coax')

        h = hidden_dimension
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(h, h),         nn.ReLU(), nn.Dropout(p=dropout_rate),
        )
        self.final_layer = nn.Linear(h, num_classes)

        # coax keeps softmax as a named attribute (needed by Captum hooks)
        if self._coax:
            self.softmax = nn.Softmax(dim=1)

        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss()

        if device_id >= 0:
            self.cuda(device=device_id)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    # coax-only: raw logits for gradient-based XAI
    def forward_logits(self, X):
        if not self._coax:
            raise AttributeError("forward_logits is only available for cognitive_agent='coax'")
        return self.final_layer(self.feature_extractor(X))

    def forward_with_log(self, X):
        h = self.feature_extractor(X)
        return self.log_softmax_layer(self.final_layer(h))

    def forward(self, X):
        h = self.feature_extractor(X)
        logits = self.final_layer(h)
        if self._coax:
            return self.softmax(logits)
        return nn.functional.softmax(logits, dim=1)


class _TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32 if y.ndim == 2 else torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ---------------------------------------------------------------------------
# Engine (wraps MLPModel, handles train / eval / io)
# ---------------------------------------------------------------------------

class MLPEngine(_BaseEngine):
    """
    Parameters
    ----------
    cognitive_agent : 'coax' | 'coxam'
        'coax'  — enables forward_logits / forward_logits_or_probs; weights
                  are loaded from  src/ai_models/coax/mlp/.
        'coxam' — standard forward pass only; weights from coxam/mlp/.
    input_dim, num_classes, hidden_dimension, dropout_rate, device_id :
        Passed through to MLPModel.
    """

    def __init__(self, *args, cognitive_agent='coxam', **kwargs):
        self.gradient_based = True
        self._agent = cognitive_agent
        self._weight_dir = model_weights_dir('mlp', cognitive_agent)
        self.model = MLPModel(*args, cognitive_agent=cognitive_agent, **kwargs)

    # coax-only: raw logits for Captum attribution adapter
    def forward_logits_or_probs(self, X_dense):
        if self._agent != 'coax':
            raise AttributeError(
                "forward_logits_or_probs is only available for cognitive_agent='coax'"
            )
        return self.model.forward_logits(X_dense)

    def predict(self, X_dense) -> np.ndarray:
        self.model.eval()
        X = torch.tensor(np.atleast_2d(X_dense), dtype=torch.float32)
        with torch.no_grad():
            return self.model(X).numpy()

    def train(self, X, y, X_dev=None, y_dev=None, batch_size=1000, epochs=300):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loader = DataLoader(_TorchDataset(X, y), batch_size=batch_size, shuffle=True)

        best_acc, best_state = 0.0, None
        for epoch in range(epochs):
            self.model.train()
            total_loss, n = 0.0, 0
            for data, targets in loader:
                if self.model.device_id >= 0:
                    data, targets = data.cuda(), targets.cuda()
                optimizer.zero_grad()
                loss = self.model.nll_loss(self.model.forward_with_log(data), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); n += 1
            print(f"Epoch {epoch + 1}  loss={total_loss / n:.4f}")

            if X_dev is not None:
                acc = self.evaluate(X_dev, y_dev)
                print(f"  dev accuracy={acc:.4f}")
                if acc > best_acc:
                    best_acc, best_state = acc, self.model.state_dict()

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Reverted to best dev accuracy {best_acc:.4f}")

    def evaluate(self, X, y) -> float:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        if self.model.device_id >= 0:
            X_t, y_t = X_t.cuda(), y_t.cuda()
        with torch.no_grad():
            return (self.model(X_t).argmax(dim=1) == y_t).float().mean().item()

    def save(self, file_name: str | None = None):
        self._weight_dir.mkdir(parents=True, exist_ok=True)
        path = self._weight_dir / (file_name or 'model_weights.pth')
        torch.save(self.model.state_dict(), path)
        print(f"[pytorch-mlp/{self._agent}] saved → {path}")

    def load(self, file_name: str):
        path = self._weight_dir / file_name
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f"[pytorch-mlp/{self._agent}] loaded ← {path}")


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
        self._weight_dir = model_weights_dir('mlp', cognitive_agent)

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
        self._weight_dir.mkdir(parents=True, exist_ok=True)
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
        self._weight_dir.mkdir(parents=True, exist_ok=True)
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
