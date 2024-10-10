import os
from glob import glob

def generate_features_csv(image_folder, csv_path, default_score=0):
    """
    Generates a CSV file containing image paths and initial lawn scores.

    Args:
        image_folder (str): Path to the folder containing images.
        csv_path (str): Path to save the generated CSV file.
        default_score (float, optional): Default score to assign to images if no model prediction is available. Defaults to 0.
    """

    # Check if image folder exists
    if not os.path.exists(image_folder):
        raise ValueError(f"Image folder '{image_folder}' does not exist.")

    # Get list of image paths
    image_paths = glob(os.path.join(image_folder, "*.jpg"), recursive=True)  # Adjust pattern for supported formats
    image_paths.extend(glob(os.path.join(image_folder, "*.png"), recursive=True))

    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        writer.writerow(["Image Path", "Initial Score"])

        # Write image paths and initial scores
        for image_path in image_paths:
            writer.writerow([image_path, default_score])

    print(f"Generated CSV file: {csv_path}")

if __name__ == "__main__":
    image_folder = "data/images"
    csv_path = "data/features.csv"
    default_score = 0.5  # Adjust default score as needed

    generate_features_csv(image_folder, csv_path, default_score)