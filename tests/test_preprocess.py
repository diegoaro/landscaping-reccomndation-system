# test_preprocess.py
import pytest
from src.preprocess import preprocess_image

def test_preprocess_image():
    img = preprocess_image("path/to/test/image.jpg")
    assert img.shape == (256, 256, 3)  # Check if image is resized properly
