import pickle
import os

print("Checking model files...")

# Check if files exist
print(f"fake_news_model_v3.pkl exists: {os.path.exists('fake_news_model_v3.pkl')}")
print(f"vectorizer_v3.pkl exists: {os.path.exists('vectorizer_v3.pkl')}")

# Check file sizes
if os.path.exists('fake_news_model_v3.pkl'):
    size = os.path.getsize('fake_news_model_v3.pkl')
    print(f"Model file size: {size} bytes")
    
if os.path.exists('vectorizer_v3.pkl'):
    size = os.path.getsize('vectorizer_v3.pkl')
    print(f"Vectorizer file size: {size} bytes")

# Try to read the first few bytes to see file type
try:
    with open('fake_news_model_v3.pkl', 'rb') as f:
        first_bytes = f.read(10)
        print(f"First 10 bytes of model: {first_bytes}")
except Exception as e:
    print(f"Error reading model file: {e}")

try:
    with open('vectorizer_v3.pkl', 'rb') as f:
        first_bytes = f.read(10)
        print(f"First 10 bytes of vectorizer: {first_bytes}")
except Exception as e:
    print(f"Error reading vectorizer file: {e}")