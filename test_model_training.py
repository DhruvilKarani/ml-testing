import tensorflow as tf
import tempfile
import os
import numpy as np
import pytest
from model import SimpleMLPClassifier
print('TensorFlow version: {}'.format(tf.__version__))

BATCH_SIZE = 16

def create_model():
    model = SimpleMLPClassifier(784, 128, 1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@pytest.fixture(scope="session")
def model():
    """Create a model instance."""
    model = create_model()
    return model


@pytest.fixture(scope="session")
def copy_model():
    model = create_model()
    return model


@pytest.fixture(scope="session")
def batch():
    """Create a batch of data."""
    x_train = tf.random.normal((BATCH_SIZE, 784))
    y_train = tf.random.uniform((BATCH_SIZE, 1), minval=0, maxval=2, dtype=tf.int32)
    return (x_train, y_train)


def test_model_forward_pass(model, batch):
    """Test if a model can perform a forward pass."""
    model(batch[0])


def test_model_output_shape(model, batch):
    """Test if the model output has the correct shape."""
    output = model(batch[0])
    assert output.shape == (BATCH_SIZE, 1)


def test_model_overfit_single_batch(model, batch):
    """Test if a model can overfit a single batch and produce near zero final loss."""
    model.fit(batch[0], batch[1], epochs=100, batch_size=BATCH_SIZE)
    result = model.evaluate(batch[0], batch[1])
    assert result[0] < 0.0001, f"Model does not overfit a single batch. Final loss is {result[0]}"



def test_if_weights_match_after_loading(model, copy_model, batch):
    """Test if the model weights match after saving and loading."""
    model.fit(batch[0], batch[1], epochs=1, batch_size=BATCH_SIZE)
    weights_copy = {w.name: np.copy(w.numpy()) for w in model.trainable_weights}

    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_weights(os.path.join(tmpdirname, "model-ckpt"))
        copy_model.evaluate(batch[0], batch[1])
        copy_model.load_weights(os.path.join(tmpdirname, "model-ckpt"))
        weights_loaded = {w.name: np.copy(w.numpy()) for w in model.trainable_weights}

    for name in weights_loaded.keys():
        assert np.allclose(weights_copy[name], weights_loaded[name]), f"Weights do not match after saving and loading. Layer {name} is different."
