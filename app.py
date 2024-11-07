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

# Set custom path for NLTK data
nltk.data.path.append('./nltk_data')

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
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get message data from JSON request
        data = request.get_json()
        if not data or "message" not in data:
            logging.warning("Invalid input: 'message' key is missing.")
            return jsonify({"error": "Invalid input, 'message' key is missing."}), 400

        msg = data.get("message", "")

        # Preprocess the text
        transformed_msg = transform_text(msg)
        if not transformed_msg:
            logging.error("Error during text transformation.")
            return jsonify({"error": "Error during text transformation."}), 500

        # Check if model and tfidf are loaded
        if tfidf is None or model is None:
            logging.error("Model or vectorizer is not loaded.")
            return jsonify({"error": "Model or vectorizer failed to load."}), 500

        # Vectorize the transformed message
        try:
            vector_input = tfidf.transform([transformed_msg]).toarray()
        except Exception as e:
            logging.error(f"Error during vectorization: {e}")
            return jsonify({"error": "Error during vectorization."}), 500

        # Predict using the model
        try:
            result = model.predict(vector_input)[0]
            prediction = "neutral" if result == 0 else "positive" if result == 1 else "negative"
        except Exception as e:
            logging.error(f"Error during model prediction: {e}")
            return jsonify({"error": "Error during model prediction."}), 500

        # Return the prediction as a JSON response
        return jsonify({"message": prediction})
    
    except Exception as e:
        # Log any unexpected errors
        logging.error(f"Unexpected error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
