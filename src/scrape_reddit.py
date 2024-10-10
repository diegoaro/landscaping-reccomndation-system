import praw
import os
import csv
import json
import requests
import time
from datetime import datetime, timedelta
from PIL import Image  # Import the Pillow library

FILLER_WORDS_FILE = 'filler_words.txt'
LANDSCAPING_TERMS_FILE = 'landscaping_terms.txt'

# Initialize global word lists
filler_words = []
landscaping_terms = []

def load_word_list(file_path):
    """Load words from a specified text file into a list."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading word list from {file_path}: {e}")
        return []

def save_word_list(word_list, file_path):
    """Save the word list back to the specified text file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for word in word_list:
                f.write(f"{word}\n")
    except Exception as e:
        print(f"Error saving word list to {file_path}: {e}")

def update_word_lists(relevant_words, body_words):
    """Update the global word lists based on relevant words from the current entry."""
    global filler_words, landscaping_terms
    filler_words = list(set(filler_words) - set(relevant_words))
    landscaping_terms = list(set(landscaping_terms) | set(body_words))

def filter_words(body, filler_words, landscaping_terms):
    """Filter the body text to extract relevant words."""
    words = body.lower().split()
    relevant_words = [word for word in words if word in landscaping_terms and word not in filler_words]
    return relevant_words

def sentiment_score(body):
    """Dummy sentiment analysis function (replace with a real model as needed)."""
    positive_words = ['good', 'great', 'excellent', 'love', 'happy', 'fantastic', 'enjoy']
    negative_words = ['bad', 'terrible', 'hate', 'sad', 'angry', 'awful']

    score = 0
    words = body.lower().split()
    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1

    return score

def scrape_reddit(subreddit_name='landscaping', num_lines=10, output_format='print'):
    """Scrape a specific number of lines from Reddit posts in the given subreddit."""
    global filler_words, landscaping_terms  # Declare global variables

    try:
        # Load dynamic word lists
        filler_words = load_word_list(FILLER_WORDS_FILE)
        landscaping_terms = load_word_list(LANDSCAPING_TERMS_FILE)

        # Reddit API credentials
        reddit = praw.Reddit(
            username='Free-Fishing-37',
            password='Dr08202002@',
            client_id='_9rrU-jCBr6qeEhENpUxdQ',
            client_secret='jDkyonZOUjM-q1RsPifWP8413Rvlgg',
            user_agent='landscaping-tips-scraper'
        )

        subreddit = reddit.subreddit(subreddit_name)

        tips = []
        today = datetime.now()
        year_ago = today - timedelta(days=365)  # Date one year ago
        image_dir = 'data/images'  # Directory to save images

        # Create the directory if it doesn't exist
        os.makedirs(image_dir, exist_ok=True)

        # Fetch new posts from the subreddit
        for days_back in range(0, 366, 15):  # 15-day increments
            target_date = year_ago + timedelta(days=days_back)
            print(f"Scraping posts from: {target_date.strftime('%Y-%m-%d')}")

            # Get new submissions
            for submission in subreddit.new(limit=None):
                if submission.created_utc < target_date.timestamp():
                    continue  # Skip submissions older than the target date
                if submission.created_utc >= (target_date + timedelta(days=15)).timestamp():
                    break  # Stop processing when we exceed the 15-day window

                if submission.stickied:
                    continue  # Skip stickied posts

                title = submission.title  # Get the submission title
                body = submission.selftext  # Get the submission body
                image_filename = None  # Initialize image filename

                # Check if the post has a valid URL and is an image link
                if submission.url and ('jpg' in submission.url or 'jpeg' in submission.url or 'png' in submission.url):
                    url = submission.url
                    image_filename = url.split("/")[-1]  # Get the last part of the URL

                    if not image_filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_filename += ".jpg"  # Default to .jpg if no extension

                    # Download the image to the specified directory
                    image_path = os.path.join(image_dir, image_filename)
                    r = requests.get(url)

                    # Ensure the request was successful before saving
                    if r.status_code == 200:
                        with open(image_path, "wb") as f:
                            f.write(r.content)

                        # Resize the image to half its original dimensions
                        with Image.open(image_path) as img:
                            new_size = (img.width // 2, img.height // 2)  # Halve the dimensions
                            img = img.resize(new_size, Image.ANTIALIAS)  # Resize with high-quality resampling
                            img.save(image_path)  # Save the resized image

                    else:
                        print(f"Failed to download image from {url}, status code: {r.status_code}")

                # Calculate relevant words and sentiment score
                relevant_words = filter_words(body, filler_words, landscaping_terms)
                score = sentiment_score(body)

                # Update dynamic word lists based on current entry
                update_word_lists(relevant_words, set(body.lower().split()))

                tips.append((title, body, image_filename, relevant_words, score))  # Append relevant info

        # Save tips to the specified output format
        if output_format == 'csv':
            with open('features.csv', mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Title", "Body", "Image Filename", "Relevant Words", "Sentiment Score"])  # Write header
                for title, body, image_filename, relevant_words, score in tips:
                    writer.writerow([title, body, image_filename, ', '.join(relevant_words), score])  # Write all info
        elif output_format == 'json':
            # Create a list of dictionaries for JSON output
            tips_json = [{"title": title, "body": body, "image_filename": image_filename,
                           "relevant_words": relevant_words, "sentiment_score": score} for
                          title, body, image_filename, relevant_words, score in tips]
            with open('reddit_tips.json', mode='w', encoding='utf-8') as file:
                json.dump(tips_json, file, indent=4)
        else:
            for idx, (title, body, image_filename, relevant_words, score) in enumerate(tips):
                print(f"Tip {idx + 1}: {title}\nBody: {body}\nImage: {image_filename}\n"
                      f"Relevant Words: {relevant_words}\nSentiment Score: {score}\n")

        # Update and save word lists to text files
        save_word_list(filler_words, FILLER_WORDS_FILE)
        save_word_list(landscaping_terms, LANDSCAPING_TERMS_FILE)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_reddit(subreddit_name='landscaping', num_lines=500, output_format='csv')  # Save to CSV
