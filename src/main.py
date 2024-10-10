import feature_extraction
import generate_features_csv
import model_trainer
import preprocess
import recommendation
import scrape_reddit


def main():
    # Define paths and parameters
    image_folder = "data/images"
    csv_path = "data/features.csv"
    model_path = "data/models/lawn_score_model.pkl"
    subreddit_name = "landscaping"
    num_lines = 10

    # Generate features CSV
    generate_features_csv.generate_csv(image_folder, csv_path)

    # Load and preprocess images
    processed_images = preprocess.batch_process_images(image_folder)

    # Extract features
    features_data = []
    for image_file, features in processed_images.items():
        extracted_features = feature_extraction.extract_features(image_file, features)
        features_data.append(extracted_features)

    # Train the model
    X, y = [], []
    for data in features_data:
        X.append(list(data.values()))
        y.append(data['lawn_score'])
    model = model_trainer.train_model(X, y)

    # Make predictions and recommendations
    for data in features_data:
        predicted_score = model.predict([list(data.values())])[0]
        recommendations = recommendation.recommend_services(predicted_score, data['detected_objects'])
        print(f"Image: {image_file}, Predicted Score: {predicted_score:.2f}, Recommendations: {recommendations}")

    # Scrape Reddit for additional tips
    reddit_tips = scrape_reddit.scrape_reddit(subreddit_name, num_lines)
    print("Reddit Tips:")
    for tip in reddit_tips:
        print(tip)


if __name__ == "__main__":
    main()