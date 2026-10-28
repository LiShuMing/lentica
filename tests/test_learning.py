import numpy as np

from conftest import load_module


mlp = load_module("learning_mlp_test", "learning/gpu-llm/experiments/chapter01/two_layer_mlp_numpy.py")


def test_numpy_backprop_matches_finite_difference():
    x = np.array([[0.3, -0.2], [0.4, 0.5]])
    y = np.array([0, 1])
    params = mlp.init_params(2, 3, 2, seed=1)
    grads = mlp.backward(mlp.forward(x, params), y, params)
    epsilon = 1e-6
    for name in ("w1", "b1", "w2", "b2"):
        values = getattr(params, name)
        expected = getattr(grads, name)
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + epsilon
            plus = mlp.cross_entropy_loss(mlp.forward(x, params).probs, y)
            values[index] = original - epsilon
            minus = mlp.cross_entropy_loss(mlp.forward(x, params).probs, y)
            values[index] = original
            np.testing.assert_allclose(expected[index], (plus - minus) / (2 * epsilon), atol=1e-6)


def test_numpy_update_reduces_loss_on_tiny_fixture():
    x = np.array([[0.3, -0.2], [0.4, 0.5]])
    y = np.array([0, 1])
    params = mlp.init_params(2, 3, 2, seed=1)
    before = mlp.cross_entropy_loss(mlp.forward(x, params).probs, y)
    mlp.update(params, mlp.backward(mlp.forward(x, params), y, params), lr=0.01)
    assert mlp.cross_entropy_loss(mlp.forward(x, params).probs, y) < before
