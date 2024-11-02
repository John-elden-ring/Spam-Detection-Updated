from flask import Flask, render_template, request, jsonify
import torch
import joblib
from text_preprocessing import preprocess_text
from spam_classifier import SpamClassifier

app = Flask(__name__)

# Load the model and vectorizer
input_dim = 42637
model = SpamClassifier(input_dim)
model.load_state_dict(torch.load('spam_classifier.pth', map_location=torch.device('cpu')))
model.eval()

vectorizer = joblib.load('vectorizer.pkl')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/detect_spam', methods=['POST'])
def detect_spam():
    data = request.json
    email_text = data.get("text", "").strip()  # Strip any leading/trailing whitespace

    # Check for empty input
    if not email_text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess the text
    processed_text = preprocess_text(email_text)
    features = vectorizer.transform([processed_text]).toarray()
    features_tensor = torch.FloatTensor(features)

    # Move input tensor to the same device as the model
    features_tensor = features_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(features_tensor)
        is_spam = output.item() > 0.5

    return jsonify({"isSpam": is_spam})


if __name__ == '__main__':
    app.run(debug=True)
