import pytest
import numpy as np
from src.feature_extraction import extract_features, extract_object_features


def test_extract_features():
    """Test feature extraction from an image."""

    # Load a dummy image
    dummy_image = np.random.rand(256, 256, 3)  # Random image for testing
    dummy_objects = [(50, 50, 100, 100)]  # Dummy object for object-based feature extraction

    features = extract_features(dummy_image, dummy_objects)

    assert len(features) > 0, "Feature extraction failed, no features extracted"
    assert isinstance(features, np.ndarray), "Features should be a NumPy array"


def test_extract_object_features():
    """Test object feature extraction."""

    # Create a dummy object image
    dummy_object_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Draw a circle to simulate an object
    cv2.circle(dummy_object_image, (50, 50), 40, (255, 255, 255), -1)

    object_features = extract_object_features(dummy_object_image)

    assert len(object_features) == 3, "Object features should return 3 elements: area, perimeter, circularity"
    assert all([f >= 0 for f in object_features]), "Object feature values should be non-negative"
