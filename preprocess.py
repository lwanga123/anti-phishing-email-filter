import pandas as pd
import numpy as np
import os
import re
import nltk

# Ensure the required linguistic tools are downloaded
nltk.download('stopwords', quiet=True)

def scrub_pii(text):
    """The Privacy Veil: Strips PII to protect user identities."""
    # Redact Emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # Redact common credential patterns (password, pin, etc)
    text = re.sub(r'(?i)(password|pwd|secret|pin|token)[:\s]+(\S+)', r'\1: [REDACTED]', text)
    # Remove HTML tags and extra whitespace
    text = re.sub(r'<.*?>', '', text)
    return text.strip()

def load_from_folder(folder_path, label, limit=300):
    """Loads emails from a specific SpamAssassin folder and applies scrubbing."""
    data = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder not found: {folder_path}")
        return []

    print(f"📂 Loading from {folder_path}...")
    filenames = os.listdir(folder_path)
    
    for filename in filenames:
        if len(data) >= limit:
            break
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="latin1", errors="ignore") as f:
                text = f.read()
                # Split header from body
                if "\n\n" in text:
                    body = text.split("\n\n", 1)[1].strip()
                else:
                    body = text.strip()
                
                if len(body) > 100:
                    # APPLY PRIVACY VEIL HERE
                    clean_body = scrub_pii(body)
                    data.append({'text': clean_body, 'label': label})
        except Exception as e:
            continue
    return data

def process_and_save():
    print("🔍 Starting Privacy-First Preprocessing...")
    
    # Path settings based on your project structure
    spam_path = os.path.join("spamassassin", "spam")
    ham_path = os.path.join("spamassassin", "easy_ham")

    # Load and scrub data
    phish_data = load_from_folder(spam_path, 1) # Label 1 for Phishing
    safe_data = load_from_folder(ham_path, 0)   # Label 0 for Safe

    # Combine into a balanced dataset
    all_data = phish_data + safe_data
    df = pd.DataFrame(all_data)

    if not df.empty:
        df.to_csv('cleaned_data.csv', index=False)
        print(f"✅ Success! Balanced and Scrubbed dataset saved to cleaned_data.csv")
        print(f"📊 Total examples: {len(df)} ({len(phish_data)} Phish / {len(safe_data)} Safe)")
    else:
        print("❌ Error: No data was loaded. Check your 'spamassassin' folder paths.")

if __name__ == "__main__":
    process_and_save()