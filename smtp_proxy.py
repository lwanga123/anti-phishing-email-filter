import asyncio
from aiosmtpd.controller import Controller
import requests
import joblib
import datetime
import re

# --- CONFIG ---
DASHBOARD_URL = "http://127.0.0.1:5000/api/report-scan"

# --- LOAD THE BRAIN ---
try:
    model = joblib.load('phishing_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("🧠 Brain loaded: Ready to scan emails.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class PhishingHandler:
    async def handle_DATA(self, server, session, envelope):
        # 1. Extract the 'Body' of the email
        email_body = envelope.content.decode('utf-8', errors='ignore')
        mailfrom = envelope.mail_from
        
        # 2. Use ML Model to Predict
        vectorized_text = vectorizer.transform([email_body])
        prediction = model.predict(vectorized_text)[0]
        # Calculate confidence score
        probability = model.predict_proba(vectorized_text)[0][1] * 100
        
        result_label = "Phishing" if prediction == 1 else "Safe"
        
        # 3. Get Metadata for the Dashboard
        subject_match = re.search(r'Subject: (.*)', email_body)
        subject = subject_match.group(1) if subject_match else "No Subject"
        
        # Inside the handle_DATA function of PhishingHandler
        report = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "sender": mailfrom,
            "subject": subject,
            "result": result_label,
            "score": round(probability, 1),
            "body": email_body  # <--- ADD THIS LINE TO SAVE THE CONTENT
}
        # 4. SEND TO DASHBOARD
        try:
            # We use the API endpoint created in app.py
            requests.post(DASHBOARD_URL, json=report)
            print(f"🚀 Sent Alert: {result_label} detected from {mailfrom}")
        except Exception as e:
            print(f"⚠️ Dashboard connection failed: {e}")

        return '250 Message accepted for delivery'

if __name__ == '__main__':
    handler = PhishingHandler()
    # Listen on localhost port 1025
    controller = Controller(handler, hostname='127.0.0.1', port=1025)
    controller.start()
    
    print("🛡️ PhishGuard SMTP Proxy is ACTIVE on port 1025...")
    print("Press Ctrl+C to stop the server.")
    
    try:
        # Keep the proxy running
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        controller.stop()
