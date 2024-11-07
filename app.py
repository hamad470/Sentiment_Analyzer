
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import logging
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define paths to model files
tfidf_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
model_path = os.path.join(os.path.dirname(__file__), "svm_sentiment.pkl")


# Load the trained model, vectorizer, and selector with error handling
try:
    tfidf = pickle.load(open(tfidf_path, "rb"))
    logging.info("Vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading vectorizer: {e}")
    tfidf = None

try:
    model = pickle.load(open(model_path, "rb"))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None



# Download necessary NLTK resources (should be done once)
nltk.data.path.append('./nltk_data')

def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, download_dir='./nltk_data')

# Check and download NLTK resources
ensure_nltk_resources()

# Define the transform_text function
def transform_text(text):
    try:
        text = text.lower()
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.isalnum()]
        filtered_words = [word for word in filtered_words if word not in stopwords.words('english')]
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        return " ".join(stemmed_words)
    except Exception as e:
        logging.error(f"Error in transform_text: {e}")
        return ""

# Define home route to render index.html
@app.route('/')
def home():
    return render_template('index.html')

# Define predict route for POST requests
# Define predict route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get message data from JSON request
        data = request.get_json()
        if not data or "message" not in data:
            logging.warning("Invalid input: 'message' key is missing.")
            print("Invalid input: 'message' key is missing.")
            return jsonify({"error": "Invalid input, 'message' key is missing."}), 400

        msg = data.get("message", "")
        print("mesage :",msg)

        # Preprocess the text
        transformed_msg = transform_text(msg)
        if not transformed_msg:
            logging.error("Error during text transformation.")
            print("Error during text transformation.")
            return jsonify({"error": "Error during text transformation."}), 500

        # Check if model, tfidf, and selector are loaded
        if tfidf is None:
            logging.error("Model or vectorizer or selector is not loaded.")
            print("Model or vectorizer or selector is not loaded.")
            return jsonify({"error": "Model, vectorizer, or selector failed to load."}), 500

        # Vectorize and select features
        try:
            vector_input = tfidf.transform([transformed_msg]).toarray()
        except Exception as e:
            logging.error(f"Error during vectorization/feature selection: {e}")
            return jsonify({"error": "Error during vectorization/feature selection."}), 500

        # Predict using the model
        try:
            result = model.predict(vector_input)[0]
            if result == 0:
                prediction = "neutral"
            elif result == 1:
                prediction = "positive"
            else:
                prediction = "negative"

        except Exception as e:
            logging.error(f"Error during model prediction: {e}")
            return jsonify({"error": "Error during model prediction."}), 500

        # Return the prediction as a JSON response
        return jsonify({"message": prediction})
    
    except Exception as e:
        # Log any unexpected errors
        logging.error(f"Unexpected error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

import os
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))




