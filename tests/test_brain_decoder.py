"""Test the python functions from src/train_brain_decoder.py."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "./src/")

from src.train_brain_decoder import BrainCNN, adjust_dimensions, normalize

testdata1 = [
    (
        np.arange(0, 10),
        4.5,
        2.8722813232690143,
        np.array(
            [
                -1.5666989,
                -1.2185436,
                -0.8703883,
                -0.522233,
                -0.1740777,
                0.1740777,
                0.522233,
                0.8703883,
                1.2185436,
                1.5666989,
            ]
        ),
    ),
    (
        np.linspace(5, 15, 6),
        10.0,
        3.415650255319866,
        np.array(
            [
                -1.46385011,
                -0.87831007,
                -0.29277002,
                0.29277002,
                0.87831007,
                1.46385011,
            ]
        ),
    ),
]


@pytest.mark.parametrize("data, mean, std, res", testdata1)
def test_normalize(data, mean, std, res) -> None:
    """Test it the data is normalized correctly."""
    result = normalize(data=data)
    norm_data = np.round(result[0], 7)
    assert np.allclose(norm_data, res)
    assert np.allclose(result[1], mean)
    assert np.allclose(result[2], std)


testdata2 = [
    jnp.ones((32, 1, 44, 1125)),
    jnp.zeros((160, 1, 44, 1125)),
    jnp.zeros((287, 1, 44, 1125)),
]


@pytest.mark.parametrize("input", testdata2)
def test_cnn(input) -> None:
    """Test CNN module.

    Test for number of output features.
    Test if net has Conv and Dense layer(s).
    Test if cnn works on input dimensions.

    Args:
        input (jnp.ndarray): The input for cnn.
    """
    key = jax.random.PRNGKey(42)  # type: ignore
    cnn = BrainCNN()
    variables = cnn.init(key, input)
    keys = [k for k, _ in variables["params"].items()]
    keys_strip = [s.split("_")[0] for s in keys]

    assert "Conv" in keys_strip
    assert "Dense" in keys_strip
    assert np.allclose(variables["params"][keys[-1]]["bias"].shape[0], 4)

    out = cnn.apply(variables, input)
    assert np.allclose(out.shape, (input.shape[0], 4))


def test_adjust_dims() -> None:
    """Test if adjust_dimensions returns correct shape."""
    assert adjust_dimensions(jnp.ones((1, 2, 3))).shape == (1, 1, 3, 2)
    assert adjust_dimensions(jnp.ones((4, 5, 6))).shape == (4, 1, 6, 5)
