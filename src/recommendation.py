import logging

def recommend_services(lawn_score, features, score_thresholds=None):
    """Generate recommendations based on lawn score and detected features."""

    # Default score thresholds
    if score_thresholds is None:
        score_thresholds = {
            'low': 50,
            'medium': 75
        }

    recommendations = []

    # Score-based recommendations
    if lawn_score < score_thresholds['low']:
        recommendations.extend(["Fertilization", "Weed Control & Mulching"])
    elif lawn_score < score_thresholds['medium']:
        recommendations.extend(["Overseeding", "Tree & Shrub Care"])
    else:
        recommendations.extend(["Lawn Maintenance Tips", "Custom Landscape Design"])

    # Feature-based recommendations
    feature_recommendations = {
        "bare_patch": ["Grading & Overseeding", "Landscape Construction"],
        "large_object": ["Rock Installations", "Custom Landscape Design"],
        # Dynamically add more mappings
        "weed_growth": ["Weed Control"],
        "poor_drainage": ["Drainage Solutions", "Lawn Grading"],
        # Extend with more features as necessary
    }

    for feature in features:
        if feature in feature_recommendations:
            recommendations.extend(feature_recommendations[feature])

    # Remove duplicates and log recommendations
    unique_recommendations = list(set(recommendations))
    logging.info(f"Generated recommendations: {unique_recommendations}")

    return unique_recommendations


# Example usage
if __name__ == "__main__":
    lawn_score = 60
    detected_features = ["bare_patch", "large_object"]
    recommendations = recommend_services(lawn_score, detected_features)
    print(f"Recommendations: {recommendations}")