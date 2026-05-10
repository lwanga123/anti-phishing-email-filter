import smtplib

# Send an email through our Proxy (Port 1025)
try:
    with smtplib.SMTP('127.0.0.1', 1025) as server:
        msg = "Subject: URGENT: Your account has been suspended!\n\nPlease click here to reset your password immediately or your funds will be lost forever."
        server.sendmail("hacker@malicious.com", "target@company.com", msg)
        print("✅ Test email sent! Check your Dashboard.")
except Exception as e:
    print(f"❌ Failed to send: {e}. Is smtp_proxy.py running?")