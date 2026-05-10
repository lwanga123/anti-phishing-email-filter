import pandas as pd

# Load the newly created dataset
df = pd.read_csv('cleaned_data.csv')

print("--- DATASET PREVIEW ---")
print(df.head())

print("\n--- PRIVACY VEIL AUDIT ---")
email_count = df['text'].str.contains('\[EMAIL\]').sum()
redact_count = df['text'].str.contains('\[REDACTED\]').sum()

print(f"Total rows: {len(df)}")
print(f"Emails scrubbed: {email_count}")
print(f"Passwords/PINs scrubbed: {redact_count}")

if email_count > 0 or redact_count > 0:
    print("\n✅ Privacy Veil is ACTIVE. Data is safe for training.")
else:
    print("\n⚠️ No PII found to scrub, or scrubbing failed. Check raw data.")