"""Two-layer MLP classifier implemented with NumPy only.

Architecture:
    input -> linear1 -> relu -> linear2 -> softmax cross-entropy loss
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class MLPParams:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclass
class ForwardCache:
    x: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    logits: np.ndarray
    probs: np.ndarray


@dataclass
class TrainHistory:
    train_loss: list[float]
    train_acc: list[float]
    val_loss: list[float]
    val_acc: list[float]


def make_dataset(
    n_samples: int = 1200,
    noise: float = 0.22,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a normalized two-class toy dataset."""
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float64)
    x_val = scaler.transform(x_val).astype(np.float64)
    return x_train, y_train.astype(np.int64), x_val, y_val.astype(np.int64)


def init_params(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int = 42,
) -> MLPParams:
    """He-style initialization for the ReLU hidden layer."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))
    b2 = np.zeros(output_dim)
    return MLPParams(w1=w1, b1=b1, w2=w2, b2=b2)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward(x: np.ndarray, params: MLPParams) -> ForwardCache:
    z1 = x @ params.w1 + params.b1
    a1 = relu(z1)
    logits = a1 @ params.w2 + params.b2
    probs = softmax(logits)
    return ForwardCache(x=x, z1=z1, a1=a1, logits=logits, probs=probs)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    batch_indices = np.arange(y.shape[0])
    correct_probs = probs[batch_indices, y]
    return float(-np.mean(np.log(correct_probs + eps)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y))


def backward(cache: ForwardCache, y: np.ndarray, params: MLPParams) -> MLPParams:
    """Backpropagate softmax cross-entropy through linear2, ReLU, linear1."""
    batch_size = y.shape[0]

    dlogits = cache.probs.copy()
    dlogits[np.arange(batch_size), y] -= 1.0
    dlogits /= batch_size

    dw2 = cache.a1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)

    da1 = dlogits @ params.w2.T
    dz1 = da1 * (cache.z1 > 0.0)

    dw1 = cache.x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return MLPParams(w1=dw1, b1=db1, w2=dw2, b2=db2)


def update(params: MLPParams, grads: MLPParams, lr: float) -> None:
    params.w1 -= lr * grads.w1
    params.b1 -= lr * grads.b1
    params.w2 -= lr * grads.w2
    params.b2 -= lr * grads.b2


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.permutation(x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def evaluate(x: np.ndarray, y: np.ndarray, params: MLPParams) -> tuple[float, float]:
    cache = forward(x, params)
    return cross_entropy_loss(cache.probs, y), accuracy(cache.probs, y)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 400,
    batch_size: int = 64,
    lr: float = 0.08,
    seed: int = 42,
    log_every: int = 50,
) -> tuple[MLPParams, TrainHistory]:
    params = init_params(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=int(np.max(y_train)) + 1,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    history = TrainHistory(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    for epoch in range(1, epochs + 1):
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            cache = forward(xb, params)
            grads = backward(cache, yb, params)
            update(params, grads, lr)

        train_loss, train_acc = evaluate(x_train, y_train, params)
        val_loss, val_acc = evaluate(x_val, y_val, params)
        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    return params, history


def gradient_check(seed: int = 7) -> float:
    """Numerically check selected gradients on a tiny synthetic batch."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(5, 3))
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    params = init_params(input_dim=3, hidden_dim=4, output_dim=3, seed=seed)
    cache = forward(x, params)
    grads = backward(cache, y, params)

    eps = 1e-5
    max_abs_error = 0.0
    checks = [
        ("w1", (0, 0)),
        ("w1", (2, 3)),
        ("b1", (1,)),
        ("w2", (0, 1)),
        ("w2", (3, 2)),
        ("b2", (2,)),
    ]

    for name, idx in checks:
        arr = getattr(params, name)
        grad_arr = getattr(grads, name)
        original = arr[idx]

        arr[idx] = original + eps
        loss_plus = cross_entropy_loss(forward(x, params).probs, y)

        arr[idx] = original - eps
        loss_minus = cross_entropy_loss(forward(x, params).probs, y)

        arr[idx] = original
        numerical = (loss_plus - loss_minus) / (2 * eps)
        analytical = grad_arr[idx]
        max_abs_error = max(max_abs_error, abs(numerical - analytical))

    return max_abs_error


def main() -> None:
    x_train, y_train, x_val, y_val = make_dataset()
    print("dataset:", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    print("gradient_check_max_abs_error:", f"{gradient_check():.2e}")

    _, history = train(x_train, y_train, x_val, y_val)
    print(
        "final:",
        f"train_acc={history.train_acc[-1]:.3f}",
        f"val_acc={history.val_acc[-1]:.3f}",
    )


if __name__ == "__main__":
    main()
