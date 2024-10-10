import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

# Consider defining these parameters in a configuration file for easy modification
RADIUS = 8  # Radius for texture analysis
N_POINTS = 8  # Number of points for texture analysis
SERVICE_RECOMMENDATIONS = {
    "bare_patch": ["Grading & Overseeding", "Landscape Construction"],
    "large_object": ["Rock Installations", "Custom Landscape Design"],
    "weed_growth": ["Weed Control"],
    "poor_drainage": ["Drainage Solutions", "Lawn Grading"],
    # Extend with more mappings based on your needs
}


def extract_features(image_path, detected_objects):
    """
    Extracts features from a preprocessed image and detected objects.

    Args:
        image_path (str): Path to the image file.
        detected_objects (list): List of object class IDs or names detected in the image.

    Returns:
        dict: A dictionary containing extracted features.
    """

    try:
        # Load the preprocessed image (assuming preprocessing is done elsewhere)
        img = cv2.imread(image_path)

        # Error handling for invalid image format
        if img is None:
            raise ValueError(f"Image at {image_path} cannot be loaded.")

        # Convert to grayscale (suitable for texture analysis)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract texture features (e.g., contrast, homogeneity)
        features = {}
        for i in range(len(gray)):
            for j in range(len(gray[0])):
                # Calculate texture properties around current pixel
                glcm = greycomatrix(gray[i:i+RADIUS, j:j+RADIUS], distances=[1], angles=[0], levels=256, mode='L')
                contrast = greycoprops(glcm, 'contrast')[0, 0]
                homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]

                # Average texture features across a local region (adjust window size as needed)
                window_size = 5
                contrast_avg = np.mean(contrast[i - window_size // 2:i + window_size // 2 + 1, j - window_size // 2:j + window_size // 2 + 1])
                homogeneity_avg = np.mean(homogeneity[i - window_size // 2:i + window_size // 2 + 1, j - window_size // 2:j + window_size // 2 + 1])

                # Add features to dictionary with appropriate keys
                features.update({f"texture_contrast_{i}_{j}": contrast_avg, f"texture_homogeneity_{i}_{j}": homogeneity_avg})

        # Extract object-based features (if object detection is used)
        object_features = {}
        for obj_id in detected_objects:
            # Map object ID to meaningful feature based on your object detection setup
            object_feature_name = SERVICE_RECOMMENDATIONS.get(obj_id, None)
            if object_feature_name:
                object_features[object_feature_name] = 1  # Presence indicator

        # Combine all features
        features.update(object_features)

        return features

    except Exception as e:
        print(f"Error extracting features from image {image_path}: {e}")
        return None  # Indicate failure