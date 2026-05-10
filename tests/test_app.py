import importlib
import os
import sys
import unittest


class PhishingTankAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(os.getcwd(), "instance", "test_artifacts")
        os.makedirs(cls.temp_dir, exist_ok=True)
        cls.db_path = os.path.join(cls.temp_dir, "test.db")
        cls.dataset_path = os.path.join(cls.temp_dir, "test_cleaned_data.csv")
        cls.feed_path = os.path.join(cls.temp_dir, "test_openphish_feed.txt")
        cls.glossary_path = os.path.join(cls.temp_dir, "test_local_translation_glossary.csv")
        cls.local_dataset_path = os.path.join(cls.temp_dir, "test_local_language_data.csv")
        cls.trusted_dataset_path = os.path.join(cls.temp_dir, "test_trusted_training_data.csv")

        normalized_db_path = cls.db_path.replace("\\", "/")
        os.environ["DATABASE_URL"] = f"sqlite:///{normalized_db_path}"
        os.environ["SECRET_KEY"] = "test-secret"
        os.environ["LOCAL_PHISHING_FEED_PATH"] = cls.feed_path
        os.environ["LOCAL_TRANSLATION_GLOSSARY_PATH"] = cls.glossary_path
        os.environ["LOCAL_LANGUAGE_DATASET_PATH"] = cls.local_dataset_path
        os.environ["LOCAL_TRUSTED_DATASET_PATH"] = cls.trusted_dataset_path
        os.environ["PHISHING_FEED_URLS"] = ""
        os.environ["AUTO_PHISHING_FEED_SYNC"] = "0"
        os.environ["AUTO_TRUSTED_DATASET_SYNC"] = "0"

        if "app" in sys.modules:
            del sys.modules["app"]

        cls.app_module = importlib.import_module("app")
        cls.app_module.app.config["TESTING"] = True

    @classmethod
    def tearDownClass(cls):
        with cls.app_module.app.app_context():
            cls.app_module.db.session.remove()
            cls.app_module.db.engine.dispose()
        for path in [cls.db_path, cls.dataset_path, cls.feed_path, cls.glossary_path, cls.local_dataset_path, cls.trusted_dataset_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass

    def setUp(self):
        with open(self.dataset_path, "w", encoding="utf-8") as handle:
            handle.write("text,label\n")
            handle.write("\"Urgent verify your account now\",1\n")
            handle.write("\"Monthly staff newsletter update\",0\n")
            handle.write("\"Reset your banking password immediately\",1\n")
            handle.write("\"Team lunch schedule for Friday\",0\n")
        with open(self.feed_path, "w", encoding="utf-8") as handle:
            handle.write("http://malicious.example/login\n")
        with open(self.glossary_path, "w", encoding="utf-8") as handle:
            handle.write("source,english,language\n")
            handle.write("\"dinani\",\"click\",\"nyanja\"\n")
            handle.write("\"mutsimikizire\",\"verify\",\"nyanja\"\n")
            handle.write("\"chinsinsi\",\"password\",\"nyanja\"\n")
            handle.write("\"akaunti\",\"account\",\"nyanja\"\n")
        with open(self.local_dataset_path, "w", encoding="utf-8") as handle:
            handle.write("text,label,language\n")
            handle.write("\"Dinani apa mutsimikizire chinsinsi\",1,nyanja\n")
            handle.write("\"Moni team msonkhano uli mawa\",0,nyanja\n")
        with open(self.trusted_dataset_path, "w", encoding="utf-8") as handle:
            handle.write("text,label,language,source,verified_at\n")
            handle.write("\"Trusted phishing verify password\",1,english,verified-test,2026-05-07\n")
            handle.write("\"Trusted staff lunch schedule\",0,english,verified-test,2026-05-07\n")

        self.app_module.DATASET_PATH = self.dataset_path
        self.app_module.LOCAL_LANGUAGE_DATASET_PATH = self.local_dataset_path
        self.app_module.LOCAL_TRANSLATION_GLOSSARY_PATH = self.glossary_path
        self.app_module.LOCAL_TRUSTED_DATASET_PATH = self.trusted_dataset_path
        with self.app_module.app.app_context():
            self.app_module.db.drop_all()
            self.app_module.db.create_all()
            self.app_module.sync_phishing_feeds(force=True)
        self.client = self.app_module.app.test_client()

    def create_user(self, username, password, is_admin=False, is_approved=True, is_rejected=False):
        with self.app_module.app.app_context():
            user = self.app_module.User(
                full_name=f"{username.title()} User",
                email=f"{username}@company.test",
                organization="Company Test",
                department="Security",
                job_title="Analyst",
                username=username,
                password=self.app_module.generate_password_hash(password),
                is_admin=is_admin,
                is_approved=is_approved,
                is_rejected=is_rejected,
            )
            self.app_module.db.session.add(user)
            self.app_module.db.session.commit()
            return user.id

    def login(self, username, password):
        return self.client.post(
            "/login",
            data={"username": username, "password": password},
            follow_redirects=False,
        )

    class FakeVectorizer:
        def transform(self, values):
            return values

    class FakeModel:
        def predict(self, values):
            text = values[0].lower()
            return [1 if "verify" in text and "password" in text else 0]

        def predict_proba(self, values):
            text = values[0].lower()
            if "verify" in text and "password" in text:
                return [[0.1, 0.9]]
            return [[0.85, 0.15]]

    def test_first_registration_becomes_approved_admin(self):
        response = self.client.post(
            "/register",
            data={
                "full_name": "Founder User",
                "email": "founder@company.test",
                "organization": "Company Test",
                "department": "Security",
                "job_title": "Founder",
                "username": "founder",
                "password": "Passw0rd!12",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        with self.app_module.app.app_context():
            user = self.app_module.User.query.filter_by(username="founder").first()
            self.assertIsNotNone(user)
            self.assertTrue(user.is_admin)
            self.assertTrue(user.is_approved)

    def test_anonymous_users_are_redirected_from_protected_routes(self):
        response = self.client.get("/check", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_unapproved_users_cannot_log_in(self):
        self.create_user("pending", "pass123", is_approved=False)
        response = self.login("pending", "pass123")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"pending administrator approval", response.data)

    def test_register_notifies_admin_and_approval_notifies_applicant(self):
        admin_id = self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        response = self.client.post(
            "/register",
            data={
                "full_name": "Pending User",
                "email": "pending@company.test",
                "organization": "Company Test",
                "department": "Finance",
                "job_title": "Clerk",
                "username": "pendinguser",
                "password": "Passw0rd!12",
            },
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        with self.app_module.app.app_context():
            admin_notifications = self.app_module.Notification.query.filter_by(recipient_user_id=admin_id).all()
            self.assertTrue(admin_notifications)
            pending_user = self.app_module.User.query.filter_by(username="pendinguser").first()
            self.assertIsNotNone(pending_user)
            pending_id = pending_user.id

        self.login("admin", "pass123")
        approve_response = self.client.get(f"/approve-user/{pending_id}", follow_redirects=True)
        self.assertEqual(approve_response.status_code, 200)

        with self.app_module.app.app_context():
            user_notifications = self.app_module.Notification.query.filter_by(recipient_user_id=pending_id).all()
            self.assertTrue(any("approved" in note.title.lower() for note in user_notifications))

    def test_rejected_request_can_register_again_with_same_details(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        rejected_id = self.create_user("repeatuser", "pass123", is_approved=False, is_rejected=True)

        with self.app_module.app.app_context():
            rejected_user = self.app_module.db.session.get(self.app_module.User, rejected_id)
            rejected_user.decision_note = "Rejected once for review."
            self.app_module.db.session.commit()

        response = self.client.post(
            "/register",
            data={
                "full_name": "Repeat User",
                "email": "repeatuser@company.test",
                "organization": "Company Test",
                "department": "Security",
                "job_title": "Analyst",
                "username": "repeatuser",
                "password": "Passw0rd!12",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"registration submitted again", response.data.lower())

        with self.app_module.app.app_context():
            refreshed_user = self.app_module.User.query.filter_by(username="repeatuser").first()
            self.assertIsNotNone(refreshed_user)
            self.assertFalse(refreshed_user.is_rejected)
            self.assertFalse(refreshed_user.is_approved)

    def test_manual_scan_is_persisted_and_feedback_can_queue_review(self):
        self.create_user("analyst", "pass123", is_approved=True)
        login_response = self.login("analyst", "pass123")
        self.assertEqual(login_response.status_code, 302)

        scan_response = self.client.post(
            "/check",
            data={"email_text": "Subject: Test\n\nPlease verify your password immediately."},
            follow_redirects=False,
        )
        self.assertEqual(scan_response.status_code, 302)
        self.assertIn("/sandbox-results/", scan_response.headers["Location"])

        with self.app_module.app.app_context():
            scan = self.app_module.ScanRecord.query.one()
            self.assertEqual(scan.sender, "analyst")
            scan_id = scan.id

        feedback_response = self.client.post(
            "/feedback",
            data={"scan_id": scan_id, "feedback": "wrong"},
            follow_redirects=True,
        )
        self.assertEqual(feedback_response.status_code, 200)

        with self.app_module.app.app_context():
            scan = self.app_module.db.session.get(self.app_module.ScanRecord, scan_id)
            self.assertEqual(scan.review_status, "queued")
            self.assertEqual(scan.feedback_value, "wrong")

    def test_feedback_queue_shows_when_user_marked_ai_wrong(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        self.create_user("analyst", "pass123", is_approved=True)

        self.login("analyst", "pass123")
        self.client.post(
            "/check",
            data={"email_text": "Subject: Verify\n\nPlease verify your password immediately."},
            follow_redirects=False,
        )

        with self.app_module.app.app_context():
            scan = self.app_module.ScanRecord.query.one()
            scan_id = scan.id

        self.client.post(
            "/feedback",
            data={"scan_id": scan_id, "feedback": "wrong"},
            follow_redirects=True,
        )
        self.client.get("/logout")

        self.login("admin", "pass123")
        response = self.client.get("/review-feedback")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"User marked this as wrong", response.data)

    def test_user_can_change_password_and_log_in_with_new_one(self):
        self.create_user("analyst", "pass123", is_approved=True)
        self.login("analyst", "pass123")

        response = self.client.post(
            "/change-password",
            data={
                "current_password": "pass123",
                "new_password": "N3wStrong!Pass",
                "confirm_password": "N3wStrong!Pass",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"password has been updated", response.data.lower())

        self.client.get("/logout")
        login_response = self.login("analyst", "N3wStrong!Pass")
        self.assertEqual(login_response.status_code, 302)

    def test_change_password_rejects_wrong_current_password(self):
        self.create_user("analyst", "pass123", is_approved=True)
        self.login("analyst", "pass123")

        response = self.client.post(
            "/change-password",
            data={
                "current_password": "wrongpass",
                "new_password": "N3wStrong!Pass",
                "confirm_password": "N3wStrong!Pass",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"current password is not correct", response.data.lower())

    def test_account_locks_after_failed_attempts_and_admin_can_reset_password(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        analyst_id = self.create_user("analyst", "pass123", is_approved=True)

        response = None
        for _ in range(self.app_module.MAX_LOGIN_ATTEMPTS):
            response = self.login("analyst", "wrong-password")

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"locked after", response.data.lower())

        with self.app_module.app.app_context():
            analyst = self.app_module.db.session.get(self.app_module.User, analyst_id)
            self.assertTrue(analyst.is_locked)
            self.assertEqual(analyst.failed_login_attempts, self.app_module.MAX_LOGIN_ATTEMPTS)

        self.login("admin", "pass123")
        reset_response = self.client.post(
            f"/admin-reset-password/{analyst_id}",
            data={
                "new_password": "TempReset!55",
                "confirm_password": "TempReset!55",
            },
            follow_redirects=True,
        )
        self.assertEqual(reset_response.status_code, 200)
        self.assertIn(b"account is now unlocked", reset_response.data.lower())

        self.client.get("/logout")
        login_response = self.login("analyst", "TempReset!55")
        self.assertEqual(login_response.status_code, 302)

    def test_archived_user_can_register_again_with_same_identity(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        recycle_id = self.create_user("recycle", "pass123", is_approved=False)

        self.login("admin", "pass123")
        archive_response = self.client.get(f"/delete-user/{recycle_id}", follow_redirects=True)
        self.assertEqual(archive_response.status_code, 200)
        self.assertIn(b"details are now free", archive_response.data.lower())

        self.client.get("/logout")
        register_response = self.client.post(
            "/register",
            data={
                "full_name": "Recycle User",
                "email": "recycle@company.test",
                "organization": "Company Test",
                "department": "Security",
                "job_title": "Analyst",
                "username": "recycle",
                "password": "Passw0rd!12",
            },
            follow_redirects=True,
        )
        self.assertEqual(register_response.status_code, 200)
        self.assertIn(b"registration submitted", register_response.data.lower())

        with self.app_module.app.app_context():
            active_user = self.app_module.User.query.filter_by(username="recycle", is_deleted=False).first()
            archived_user = self.app_module.User.query.filter_by(id=recycle_id).first()
            self.assertIsNotNone(active_user)
            self.assertIsNotNone(archived_user)
            self.assertTrue(archived_user.is_deleted)

    def test_admin_created_user_does_not_switch_current_session(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        self.login("admin", "pass123")

        response = self.client.post(
            "/manage-users",
            data={
                "full_name": "Managed User",
                "email": "managed@company.test",
                "organization": "Company Test",
                "department": "Finance",
                "job_title": "Clerk",
                "username": "managed",
                "password": "Passw0rd!12",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"user created successfully", response.data.lower())
        self.assertIn(b"access control console", response.data.lower())

        with self.client.session_transaction() as session_data:
            self.assertEqual(str(session_data.get("_user_id")), "1")

        with self.app_module.app.app_context():
            created_user = self.app_module.User.query.filter_by(username="managed").first()
            self.assertIsNotNone(created_user)
            self.assertFalse(created_user.is_admin)

    def test_non_admin_dashboard_only_shows_their_own_scan_activity(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        self.create_user("alice", "pass123", is_approved=True)
        self.create_user("bob", "pass123", is_approved=True)

        with self.app_module.app.app_context():
            self.app_module.db.session.add_all(
                [
                    self.app_module.ScanRecord(
                        sender="alice",
                        subject="Alice Scan",
                        body="Alice body",
                        result="Safe",
                        score=80.0,
                        source="manual",
                    ),
                    self.app_module.ScanRecord(
                        sender="bob",
                        subject="Bob Scan",
                        body="Bob body",
                        result="Phishing",
                        score=95.0,
                        source="manual",
                        is_quarantined=True,
                    ),
                ]
            )
            self.app_module.db.session.commit()

        self.login("alice", "pass123")
        response = self.client.get("/", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"My Recent Scan Activity", response.data)
        self.assertIn(b"alice", response.data)
        self.assertNotIn(b"bob", response.data)

        self.client.get("/logout")
        self.login("admin", "pass123")
        admin_response = self.client.get("/", follow_redirects=True)
        self.assertEqual(admin_response.status_code, 200)
        self.assertIn(b"Recent Scan Activity", admin_response.data)
        self.assertIn(b"alice", admin_response.data)
        self.assertIn(b"bob", admin_response.data)

    def test_feed_url_match_forces_phishing_verdict(self):
        self.create_user("analyst", "pass123", is_approved=True)
        self.login("analyst", "pass123")

        old_ready = self.app_module.ML_READY
        self.app_module.ML_READY = False
        try:
            response = self.client.post(
                "/check",
                data={
                    "email_text": "Subject: Verify\n\nClick now: http://malicious.example/login"
                },
                follow_redirects=True,
            )
        finally:
            self.app_module.ML_READY = old_ready

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Phishing", response.data)
        self.assertIn(b"Threat Feed", response.data)
        with self.app_module.app.app_context():
            scan = self.app_module.ScanRecord.query.one()
            self.assertEqual(scan.result, "Phishing")
            self.assertEqual(scan.decision_source, "threat_feed")

    def test_safe_preview_sandbox_neutralizes_active_content_and_shows_nlp(self):
        self.create_user("analyst", "pass123", is_approved=True)
        self.login("analyst", "pass123")

        response = self.client.post(
            "/check",
            data={
                "email_text": (
                    "Subject: Script Test\n\n"
                    "<script>alert('bad')</script>"
                    "<a href=\"http://evil.example/login\">verify password</a>"
                    "<form><input type=\"password\"></form>"
                )
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)

        with self.app_module.app.app_context():
            scan = self.app_module.ScanRecord.query.one()
            scan_id = scan.id

        result_response = self.client.get(f"/sandbox-results/{scan_id}", follow_redirects=True)
        self.assertEqual(result_response.status_code, 200)
        self.assertIn(b"NLP Analysis", result_response.data)
        self.assertIn(b"password", result_response.data.lower())

        preview_response = self.client.get(f"/safe-preview/{scan_id}", follow_redirects=True)
        self.assertEqual(preview_response.status_code, 200)
        self.assertIn(b"Scripts Blocked", preview_response.data)
        self.assertIn(b"Items Neutralized", preview_response.data)
        self.assertIn(b"[SCRIPT BLOCKED]", preview_response.data)
        self.assertIn(b"[LINK DISABLED:", preview_response.data)
        self.assertNotIn(b"<script>alert", preview_response.data)

    def test_local_language_translation_copy_can_drive_phishing_verdict(self):
        self.create_user("analyst", "pass123", is_approved=True)
        self.login("analyst", "pass123")

        old_ready = self.app_module.ML_READY
        old_model = self.app_module.model
        old_vectorizer = self.app_module.vectorizer
        self.app_module.ML_READY = True
        self.app_module.model = self.FakeModel()
        self.app_module.vectorizer = self.FakeVectorizer()
        try:
            response = self.client.post(
                "/check",
                data={
                    "email_text": "Subject: Local\n\nDinani apa kuti mutsimikizire chinsinsi cha akaunti."
                },
                follow_redirects=True,
            )
        finally:
            self.app_module.ML_READY = old_ready
            self.app_module.model = old_model
            self.app_module.vectorizer = old_vectorizer

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Private Translation Copy", response.data)
        self.assertIn(b"Translated Ml", response.data)

        with self.app_module.app.app_context():
            scan = self.app_module.ScanRecord.query.one()
            self.assertEqual(scan.result, "Phishing")
            self.assertEqual(scan.decision_source, "translated_ml")
            self.assertTrue(scan.translation_applied)
            self.assertIn("verify", scan.translated_body.lower())

    def test_admin_can_ingest_api_scan_review_it_and_view_reports(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        self.login("admin", "pass123")

        old_ready = self.app_module.ML_READY
        self.app_module.ML_READY = False
        try:
            api_response = self.client.post(
                "/api/report-scan",
                json={
                    "sender": "attacker@example.com",
                    "subject": "Urgent password reset",
                    "body": "Click here to verify your account immediately.",
                    "result": "Phishing",
                    "score": 99.0,
                },
            )
        finally:
            self.app_module.ML_READY = old_ready

        self.assertEqual(api_response.status_code, 201)
        scan_id = api_response.get_json()["scan_id"]

        quarantine_response = self.client.get("/quarantine")
        self.assertEqual(quarantine_response.status_code, 200)
        self.assertIn(b"attacker@example.com", quarantine_response.data)

        review_response = self.client.post(
            "/review-feedback",
            data={"approve": scan_id},
            follow_redirects=True,
        )
        self.assertEqual(review_response.status_code, 200)

        reports_response = self.client.get("/reports")
        self.assertEqual(reports_response.status_code, 200)
        self.assertIn(b"Reviewed samples: 1", reports_response.data)

    def test_trusted_dataset_import_approval_and_retraining(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        self.login("admin", "pass123")

        sync_response = self.client.post(
            "/trusted-datasets",
            data={"action": "sync"},
            follow_redirects=True,
        )
        self.assertEqual(sync_response.status_code, 200)
        self.assertIn(b"Trusted datasets synced", sync_response.data)

        with self.app_module.app.app_context():
            entries = self.app_module.TrustedDatasetEntry.query.order_by(
                self.app_module.TrustedDatasetEntry.id.asc()
            ).all()
            self.assertEqual(len(entries), 2)
            self.assertTrue(all(entry.status == "pending" for entry in entries))
            entry_ids = [str(entry.id) for entry in entries]

        approve_response = self.client.post(
            "/trusted-datasets",
            data={"approve": entry_ids},
            follow_redirects=True,
        )
        self.assertEqual(approve_response.status_code, 200)
        self.assertIn(b"Approved 2", approve_response.data)

        with self.app_module.app.app_context():
            approved_entries = self.app_module.TrustedDatasetEntry.query.filter_by(status="approved").all()
            self.assertEqual(len(approved_entries), 2)
            self.assertTrue(all(entry.used_in_training for entry in approved_entries))
            model_version = self.app_module.ModelVersion.query.order_by(
                self.app_module.ModelVersion.created_at.desc()
            ).first()
            self.assertIsNotNone(model_version)
            self.assertEqual(model_version.trusted_rows, 2)

    def test_retrain_route_uses_reviewed_feedback_and_returns_success(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        self.login("admin", "pass123")

        with self.app_module.app.app_context():
            safe_scan = self.app_module.ScanRecord(
                sender="admin",
                subject="Internal memo",
                body="The finance meeting is at 10 AM.",
                result="Safe",
                score=88.0,
                source="manual",
                review_status="reviewed",
                review_label=0,
                is_quarantined=False,
            )
            phish_scan = self.app_module.ScanRecord(
                sender="attacker@example.com",
                subject="Verify now",
                body="Reset your banking password immediately.",
                result="Phishing",
                score=97.0,
                source="smtp",
                review_status="reviewed",
                review_label=1,
                is_quarantined=True,
            )
            self.app_module.db.session.add_all([safe_scan, phish_scan])
            self.app_module.db.session.commit()

        response = self.client.get("/retrain-now", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model retraining complete", response.data)

    def test_review_processing_triggers_automatic_retraining(self):
        self.create_user("admin", "pass123", is_admin=True, is_approved=True)
        self.login("admin", "pass123")

        with self.app_module.app.app_context():
            scan = self.app_module.ScanRecord(
                sender="analyst",
                subject="Reset now",
                body="Reset your banking password immediately.",
                result="Phishing",
                score=97.0,
                source="manual",
                review_status="queued",
                feedback_value="wrong",
                is_quarantined=True,
            )
            self.app_module.db.session.add(scan)
            self.app_module.db.session.commit()
            scan_id = scan.id

        response = self.client.post(
            "/review-feedback",
            data={"approve": scan_id},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"automatic retraining completed", response.data.lower())

        with self.app_module.app.app_context():
            scan = self.app_module.db.session.get(self.app_module.ScanRecord, scan_id)
            self.assertTrue(scan.used_in_training)
            self.assertEqual(
                self.app_module.get_app_state_value("auto_retrain_last_status"),
                "success",
            )


if __name__ == "__main__":
    unittest.main()
