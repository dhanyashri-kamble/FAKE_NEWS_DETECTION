import pickle
import joblib

# Try with pickle
print("=== Trying pickle ===")
try:
    with open('fake_news_model_v3.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully with pickle")
except Exception as e:
    print(f"Error loading model with pickle: {e}")

try:
    with open('vectorizer_v3.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully with pickle")
except Exception as e:
    print(f"Error loading vectorizer with pickle: {e}")

# Try with joblib
print("\n=== Trying joblib ===")
try:
    model = joblib.load('fake_news_model_v3.pkl')
    print("Model loaded successfully with joblib")
except Exception as e:
    print(f"Error loading model with joblib: {e}")

try:
    vectorizer = joblib.load('vectorizer_v3.pkl')
    print("Vectorizer loaded successfully with joblib")
except Exception as e:
    print(f"Error loading vectorizer with joblib: {e}")