import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Preprocesses an image by resizing, converting color space, normalizing, and (optionally) detecting objects.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array (RGB, normalized between 0-1).

    Raises:
        ValueError: If the image cannot be loaded.
    """

    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at {image_path} cannot be loaded.")

        # Resize the image while maintaining aspect ratio
        # Consider using content-aware resize for better preservation of details
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # Convert color space (adjust based on your model's requirement)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to the range [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def detect_objects(img, model_path="yolov8n.onnx"):
    """
    Detects objects in the image using a pre-trained YOLO model (optional).

    Args:
        img (numpy.ndarray): The preprocessed image as a NumPy array.
        model_path (str, optional): Path to the pre-trained object detection model. Defaults to "yolov8n.onnx".

    Returns:
        list: List of detected object class IDs (if object detection is enabled) or an empty list otherwise.

    Raises:
        Exception: If an error occurs during object detection.
    """

    # Check if object detection is enabled and a model path is provided
    if model_path is not None:
        try:
            model = cv2.dnn.readNet(model_path)
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (640, 640), swapRB=True, crop=False)
            model.setInput(blob)
            detections = model.forward()

            features = []
            for detection in detections[0]:
                confidence = float(detection[5])
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    class_id = int(detection[6])
                    features.append(class_id)  # Store class ID for further use

            return features

        except Exception as e:
            print(f"Error in object detection: {e}")
            return []

    # Return an empty list if object detection is disabled
    return []


def batch_process_images(image_dir, model_path="yolov8n.onnx"):
    """
    Batch processes all images in a directory, performs preprocessing, and optionally detects objects.

    Args:
        image_dir (str): Path to the directory containing images.
        model_path (str, optional): Path to the pre-trained object detection model. Defaults to "yolov8n.onnx".

    Returns:
        dict: A dictionary where keys are image filenames and values are lists of detected object class IDs (or empty lists if object detection is disabled).
    """

    processed_images = {}
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        img = preprocess_image(image_path)
        if img is not None:
            features = detect_objects(img, model_path)
            processed_images[image_file] = features
    return processed_images


# Example usage
if __name__ == "__main__":
    image_dir = "path/to/images"  # Replace with your image directory
    processed_images = batch_process_images(image_dir)
    print(f"Processed images: {processed_images}")