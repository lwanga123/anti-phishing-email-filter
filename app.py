import csv
import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlsplit, urlunsplit

import joblib
import requests
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from markupsafe import escape
from sqlalchemy import func, inspect, or_, text
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "instance", "phishingtank.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{DEFAULT_DB_PATH}",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
LOCAL_LANGUAGE_DATASET_PATH = os.environ.get(
    "LOCAL_LANGUAGE_DATASET_PATH",
    os.path.join(BASE_DIR, "local_language_data.csv"),
)
LOCAL_TRUSTED_DATASET_PATH = os.environ.get(
    "LOCAL_TRUSTED_DATASET_PATH",
    os.path.join(BASE_DIR, "trusted_training_data.csv"),
)
TRUSTED_DATASET_URLS = [
    url.strip()
    for url in os.environ.get("TRUSTED_DATASET_URLS", "").split(",")
    if url.strip()
]
TRUSTED_DATASET_REFRESH_HOURS = max(
    1,
    int(os.environ.get("TRUSTED_DATASET_REFRESH_HOURS", "24")),
)
TRUSTED_DATASET_TIMEOUT_SECONDS = max(
    1,
    int(os.environ.get("TRUSTED_DATASET_TIMEOUT_SECONDS", "10")),
)
AUTO_TRUSTED_DATASET_SYNC = os.environ.get("AUTO_TRUSTED_DATASET_SYNC", "1") != "0"
AUTO_APPROVE_TRUSTED_DATASETS = os.environ.get("AUTO_APPROVE_TRUSTED_DATASETS", "0") == "1"
MIN_MODEL_ACCEPTANCE_ACCURACY = float(os.environ.get("MIN_MODEL_ACCEPTANCE_ACCURACY", "0.70"))
LOCAL_TRANSLATION_GLOSSARY_PATH = os.environ.get(
    "LOCAL_TRANSLATION_GLOSSARY_PATH",
    os.path.join(BASE_DIR, "local_translation_glossary.csv"),
)
LOCAL_PHISHING_FEED_PATH = os.environ.get(
    "LOCAL_PHISHING_FEED_PATH",
    os.path.join(BASE_DIR, "openphish_feed.txt"),
)
REMOTE_PHISHING_FEED_URLS = [
    url.strip()
    for url in os.environ.get("PHISHING_FEED_URLS", "https://openphish.com/feed.txt").split(",")
    if url.strip()
]
PHISHING_FEED_REFRESH_MINUTES = max(
    1,
    int(os.environ.get("PHISHING_FEED_REFRESH_MINUTES", "60")),
)
PHISHING_FEED_TIMEOUT_SECONDS = max(
    1,
    int(os.environ.get("PHISHING_FEED_TIMEOUT_SECONDS", "5")),
)
AUTO_PHISHING_FEED_SYNC = os.environ.get("AUTO_PHISHING_FEED_SYNC", "1") != "0"
MAX_LOGIN_ATTEMPTS = max(1, int(os.environ.get("MAX_LOGIN_ATTEMPTS", "5")))
AUTO_RETRAIN_ENABLED = os.environ.get("AUTO_RETRAIN_ENABLED", "1") != "0"
AUTO_RETRAIN_REVIEW_THRESHOLD = max(
    1,
    int(os.environ.get("AUTO_RETRAIN_REVIEW_THRESHOLD", "1")),
)

model = None
vectorizer = None
ML_READY = False
ML_ERROR = ""
FEED_SYNC_IN_PROGRESS = False
TRUSTED_DATASET_SYNC_IN_PROGRESS = False


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    email = db.Column(db.String(255), unique=True, nullable=True)
    organization = db.Column(db.String(150), nullable=True)
    department = db.Column(db.String(150), nullable=True)
    job_title = db.Column(db.String(150), nullable=True)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    is_approved = db.Column(db.Boolean, nullable=False, default=False)
    is_rejected = db.Column(db.Boolean, nullable=False, default=False)
    decision_note = db.Column(db.Text, nullable=True)
    failed_login_attempts = db.Column(db.Integer, nullable=False, default=0)
    is_locked = db.Column(db.Boolean, nullable=False, default=False)
    locked_at = db.Column(db.DateTime, nullable=True)
    is_deleted = db.Column(db.Boolean, nullable=False, default=False)

    @property
    def is_active(self):
        return not bool(self.is_deleted)


class ScanRecord(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    sender = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(255), nullable=False, default="No Subject")
    body = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(32), nullable=False, default="Unavailable")
    score = db.Column(db.Float, nullable=False, default=0.0)
    source = db.Column(db.String(32), nullable=False, default="manual")
    review_status = db.Column(db.String(32), nullable=False, default="unreviewed")
    review_label = db.Column(db.Integer, nullable=True)
    is_quarantined = db.Column(db.Boolean, nullable=False, default=False)
    extracted_url = db.Column(db.String(2048), nullable=True)
    decision_source = db.Column(db.String(64), nullable=False, default="ml")
    matched_feed_source = db.Column(db.String(255), nullable=True)
    feedback_value = db.Column(db.String(32), nullable=True)
    feedback_submitted_at = db.Column(db.DateTime, nullable=True)
    used_in_training = db.Column(db.Boolean, nullable=False, default=False)
    detected_language = db.Column(db.String(64), nullable=True)
    translated_body = db.Column(db.Text, nullable=True)
    translation_applied = db.Column(db.Boolean, nullable=False, default=False)
    translation_terms = db.Column(db.Integer, nullable=False, default=0)
    original_text_score = db.Column(db.Float, nullable=True)
    translated_text_score = db.Column(db.Float, nullable=True)

    @property
    def predicted_label(self):
        if self.result == "Phishing":
            return 1
        if self.result == "Safe":
            return 0
        return None


class ThreatFeedEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    normalized_url = db.Column(db.String(2048), unique=True, nullable=False, index=True)
    source = db.Column(db.String(255), nullable=False, default="snapshot")
    first_seen = db.Column(db.DateTime, nullable=False, default=datetime.now)
    last_seen = db.Column(db.DateTime, nullable=False, default=datetime.now)


class TrustedDatasetEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    label = db.Column(db.Integer, nullable=False)
    language = db.Column(db.String(64), nullable=True)
    source = db.Column(db.String(255), nullable=False, default="trusted_source")
    source_url = db.Column(db.String(2048), nullable=True)
    external_id = db.Column(db.String(255), nullable=True)
    content_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="pending")
    imported_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    verified_at = db.Column(db.String(64), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    reviewed_by_user_id = db.Column(db.Integer, nullable=True)
    used_in_training = db.Column(db.Boolean, nullable=False, default=False)
    decision_note = db.Column(db.Text, nullable=True)


class ModelVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    rows = db.Column(db.Integer, nullable=False, default=0)
    feedback_rows = db.Column(db.Integer, nullable=False, default=0)
    trusted_rows = db.Column(db.Integer, nullable=False, default=0)
    accuracy = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(32), nullable=False, default="active")
    message = db.Column(db.Text, nullable=True)


class AppState(db.Model):
    key = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipient_user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(180), nullable=False)
    message = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(32), nullable=False, default="info")
    link = db.Column(db.String(255), nullable=True)
    is_read = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)


@login_manager.user_loader
def load_user(user_id):
    user = db.session.get(User, int(user_id))
    if user and bool(user.is_deleted):
        return None
    return user


def get_user_display_name(user):
    return (user.full_name or user.username or "").strip()


def validate_email_address(email):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", (email or "").strip()))


def validate_password_strength(password):
    issues = []
    if len(password or "") < 10:
        issues.append("Password must be at least 10 characters long.")
    if not re.search(r"[A-Z]", password or ""):
        issues.append("Password must include at least one uppercase letter.")
    if not re.search(r"[a-z]", password or ""):
        issues.append("Password must include at least one lowercase letter.")
    if not re.search(r"\d", password or ""):
        issues.append("Password must include at least one number.")
    if not re.search(r"[^A-Za-z0-9]", password or ""):
        issues.append("Password must include at least one special character.")
    return issues


def create_notification(recipient_user_id, title, message, category="info", link=None):
    db.session.add(
        Notification(
            recipient_user_id=recipient_user_id,
            title=title,
            message=message,
            category=category,
            link=link,
        )
    )


def notify_admins(title, message, category="info", link=None, exclude_user_id=None):
    admins = User.query.filter_by(is_admin=True, is_approved=True, is_deleted=False).all()
    for admin in admins:
        if exclude_user_id and admin.id == exclude_user_id:
            continue
        create_notification(
            admin.id,
            title,
            message,
            category=category,
            link=link,
        )


def notify_admins_of_signup(user):
    notify_admins(
        "New access request",
        f"{get_user_display_name(user)} requested access from {user.organization or 'an organization'} "
        f"as {user.job_title or 'a team member'}.",
        category="warning",
        link=url_for("manage_users"),
    )


def get_latest_user_notice(user):
    notification = (
        Notification.query.filter_by(recipient_user_id=user.id)
        .order_by(Notification.created_at.desc())
        .first()
    )
    return notification.message if notification else user.decision_note


def build_user_identity_conflict_query(username, email):
    username_value = (username or "").strip().lower()
    email_value = (email or "").strip().lower()
    return User.query.filter(
        User.is_deleted.is_(False),
        or_(
            func.lower(User.username) == username_value,
            func.lower(User.email) == email_value,
        ),
    )


def find_user_identity_conflicts(username, email):
    return build_user_identity_conflict_query(username, email).all()


def find_user_identity_conflict(username, email):
    return build_user_identity_conflict_query(username, email).first()


def can_refresh_access_request(user):
    return (
        bool(user)
        and not bool(user.is_deleted)
        and not bool(user.is_approved)
        and not bool(user.is_admin)
    )


def archive_user_account(user):
    timestamp_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    old_username = (user.username or f"user-{user.id}").strip().lower()
    archived_username = f"archived-{user.id}-{timestamp_suffix}"
    archived_email = f"archived-{user.id}-{timestamp_suffix}@phishguard.local"

    ScanRecord.query.filter_by(sender=old_username).update(
        {"sender": archived_username},
        synchronize_session=False,
    )

    user.username = archived_username[:150]
    user.email = archived_email[:255]
    user.full_name = f"{get_user_display_name(user)} [Archived]"
    user.password = generate_password_hash(uuid.uuid4().hex)
    user.is_admin = False
    user.is_approved = False
    user.is_rejected = True
    user.failed_login_attempts = MAX_LOGIN_ATTEMPTS
    user.is_locked = True
    user.locked_at = datetime.now()
    user.is_deleted = True
    user.decision_note = "This account was archived by an administrator."


def get_pending_training_scans():
    return (
        ScanRecord.query.filter_by(review_status="reviewed", used_in_training=False)
        .filter(ScanRecord.review_label.in_([0, 1]))
        .order_by(ScanRecord.timestamp.asc())
        .all()
    )


def get_pending_trusted_training_entries():
    return (
        TrustedDatasetEntry.query.filter_by(status="approved", used_in_training=False)
        .order_by(TrustedDatasetEntry.imported_at.asc())
        .all()
    )


def get_auto_retrain_status():
    pending_training_scans = get_pending_training_scans()
    pending_trusted_entries = get_pending_trusted_training_entries()
    pending_total = len(pending_training_scans) + len(pending_trusted_entries)
    return {
        "enabled": AUTO_RETRAIN_ENABLED,
        "threshold": AUTO_RETRAIN_REVIEW_THRESHOLD,
        "pending_count": pending_total,
        "pending_reviewed_scans": len(pending_training_scans),
        "pending_trusted_samples": len(pending_trusted_entries),
        "last_status": get_app_state_value("auto_retrain_last_status", "idle"),
        "last_message": get_app_state_value("auto_retrain_last_message", "Automatic retraining is waiting for reviewed samples."),
        "last_success": get_app_state_value("auto_retrain_last_success", "Never"),
    }


def maybe_auto_retrain(trigger_reason):
    pending_training_scans = get_pending_training_scans()
    pending_trusted_entries = get_pending_trusted_training_entries()
    pending_count = len(pending_training_scans) + len(pending_trusted_entries)

    if not AUTO_RETRAIN_ENABLED:
        set_app_state_value("auto_retrain_last_status", "disabled")
        set_app_state_value(
            "auto_retrain_last_message",
            "Automatic retraining is disabled for this deployment.",
        )
        db.session.commit()
        return None

    if pending_count < AUTO_RETRAIN_REVIEW_THRESHOLD:
        set_app_state_value("auto_retrain_last_status", "waiting")
        set_app_state_value(
            "auto_retrain_last_message",
            f"Waiting for {AUTO_RETRAIN_REVIEW_THRESHOLD} reviewed samples before retraining. "
            f"Current queue: {pending_count}.",
        )
        db.session.commit()
        return None

    try:
        outcome = retrain_model()
    except Exception as exc:
        set_app_state_value("auto_retrain_last_status", "failed")
        set_app_state_value(
            "auto_retrain_last_message",
            f"Automatic retraining failed after {trigger_reason}: {exc}",
        )
        db.session.commit()
        notify_admins(
            "Automatic retraining failed",
            f"PhishGuard could not retrain after {trigger_reason}. Reason: {exc}",
            category="danger",
            link=url_for("reports"),
        )
        db.session.commit()
        return {"status": "failed", "error": str(exc)}

    for scan in pending_training_scans:
        scan.used_in_training = True
    for entry in pending_trusted_entries:
        entry.used_in_training = True

    success_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_app_state_value("auto_retrain_last_status", "success")
    set_app_state_value(
        "auto_retrain_last_message",
        f"Automatic retraining completed after {trigger_reason} using {outcome['rows']} examples.",
    )
    set_app_state_value("auto_retrain_last_success", success_time)
    notify_admins(
        "Automatic retraining complete",
        f"PhishGuard retrained itself after {trigger_reason} using {outcome['rows']} examples "
        f"and {outcome['feedback_rows']} reviewed scans.",
        category="success",
        link=url_for("reports"),
    )
    db.session.commit()
    return {"status": "success", "outcome": outcome}


def admin_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Access denied.", "danger")
            return redirect(url_for("home"))
        return func(*args, **kwargs)

    return decorated_function


def load_ml_assets():
    global model, vectorizer, ML_READY, ML_ERROR

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        ML_READY = True
        ML_ERROR = ""
    except Exception as exc:
        model = None
        vectorizer = None
        ML_READY = False
        ML_ERROR = str(exc)


def normalize_result(value):
    raw_value = (value or "").strip().lower()
    if raw_value in {"phishing", "phish", "malicious", "spam"}:
        return "Phishing"
    if raw_value in {"safe", "ham", "legit", "legitimate", "clean"}:
        return "Safe"
    return "Unavailable"


def normalize_url(raw_url):
    cleaned = (raw_url or "").strip().strip("<>\"'()[]{}")
    if not cleaned:
        return None

    try:
        parsed = urlsplit(cleaned)
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        return None

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    normalized = urlunsplit((scheme, parsed.netloc.lower(), path, parsed.query, ""))
    return normalized


def get_lookup_candidates(raw_url):
    normalized = normalize_url(raw_url)
    if not normalized:
        return []

    candidates = [normalized]
    parsed = urlsplit(normalized)
    if parsed.query:
        no_query = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        if no_query not in candidates:
            candidates.append(no_query)
    return candidates


def extract_feed_name(feed_url):
    parsed = urlsplit(feed_url)
    return parsed.netloc or "remote-feed"


def parse_feed_urls(feed_text):
    parsed_urls = set()
    for line in (feed_text or "").splitlines():
        normalized = normalize_url(line)
        if normalized:
            parsed_urls.add(normalized)
    return parsed_urls


def get_app_state_value(key, default=None):
    state = db.session.get(AppState, key)
    return state.value if state else default


def set_app_state_value(key, value):
    state = db.session.get(AppState, key)
    if state:
        state.value = value
    else:
        db.session.add(AppState(key=key, value=value))


def normalize_training_label(value):
    raw_value = str(value or "").strip().lower()
    if raw_value in {"1", "phishing", "spam", "malicious", "unsafe", "bad"}:
        return 1
    if raw_value in {"0", "ham", "safe", "legit", "legitimate", "clean"}:
        return 0
    return None


def build_training_content_hash(text_value, label_value, source_value):
    normalized_text = re.sub(r"\s+", " ", (text_value or "").strip().lower())
    raw_hash = f"{source_value}|{label_value}|{normalized_text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw_hash).hexdigest()


def extract_source_name(source_url):
    if not source_url:
        return "local_trusted_dataset"
    parsed = urlsplit(source_url)
    return parsed.netloc or os.path.basename(source_url) or "trusted_dataset"


def parse_trusted_dataset_csv(csv_text, source_name, source_url=None):
    rows = []
    errors = []
    reader = csv.DictReader((csv_text or "").splitlines())
    if not reader.fieldnames:
        return rows, ["Dataset has no CSV header."]

    for row_number, row in enumerate(reader, start=2):
        text_value = (
            row.get("text")
            or row.get("body")
            or row.get("email_text")
            or row.get("message")
            or ""
        ).strip()
        label_value = normalize_training_label(row.get("label") or row.get("class") or row.get("verdict"))
        if not text_value or label_value not in (0, 1):
            errors.append(f"Row {row_number}: missing text or valid label.")
            continue

        row_source = (row.get("source") or source_name or "trusted_dataset").strip()
        content_hash = build_training_content_hash(text_value, label_value, row_source)
        rows.append(
            {
                "text": text_value,
                "label": label_value,
                "language": (row.get("language") or row.get("lang") or "unknown").strip()[:64],
                "source": row_source[:255],
                "source_url": (source_url or row.get("source_url") or "")[:2048],
                "external_id": (row.get("id") or row.get("external_id") or "")[:255],
                "verified_at": (row.get("verified_at") or row.get("date") or "")[:64],
                "content_hash": content_hash,
            }
        )
    return rows, errors


def store_trusted_dataset_rows(rows):
    imported = 0
    existing_hashes = {
        entry.content_hash
        for entry in TrustedDatasetEntry.query.filter(
            TrustedDatasetEntry.content_hash.in_([row["content_hash"] for row in rows])
        ).all()
    } if rows else set()

    for row in rows:
        if row["content_hash"] in existing_hashes:
            continue
        status = "approved" if AUTO_APPROVE_TRUSTED_DATASETS else "pending"
        db.session.add(
            TrustedDatasetEntry(
                text=row["text"],
                label=row["label"],
                language=row["language"],
                source=row["source"],
                source_url=row["source_url"],
                external_id=row["external_id"],
                verified_at=row["verified_at"],
                content_hash=row["content_hash"],
                status=status,
                reviewed_at=datetime.now() if status == "approved" else None,
                decision_note="Auto-approved trusted source." if status == "approved" else None,
            )
        )
        existing_hashes.add(row["content_hash"])
        imported += 1
    return imported


def import_trusted_datasets(force=False):
    global TRUSTED_DATASET_SYNC_IN_PROGRESS

    if TRUSTED_DATASET_SYNC_IN_PROGRESS:
        return {"status": "busy", "imported": 0, "pending": 0, "approved": 0, "errors": []}

    now = datetime.now()
    last_attempt_raw = get_app_state_value("trusted_dataset_last_attempt")
    if not force and last_attempt_raw:
        try:
            last_attempt = datetime.fromisoformat(last_attempt_raw)
            if now - last_attempt < timedelta(hours=TRUSTED_DATASET_REFRESH_HOURS):
                return {
                    "status": "fresh",
                    "imported": 0,
                    "pending": TrustedDatasetEntry.query.filter_by(status="pending").count(),
                    "approved": TrustedDatasetEntry.query.filter_by(status="approved").count(),
                    "errors": [],
                }
        except ValueError:
            pass

    TRUSTED_DATASET_SYNC_IN_PROGRESS = True
    try:
        imported = 0
        errors = []
        set_app_state_value("trusted_dataset_last_attempt", now.isoformat())

        if os.path.exists(LOCAL_TRUSTED_DATASET_PATH):
            try:
                with open(LOCAL_TRUSTED_DATASET_PATH, "r", encoding="utf-8", errors="ignore") as handle:
                    rows, parse_errors = parse_trusted_dataset_csv(
                        handle.read(),
                        "local_trusted_dataset",
                        LOCAL_TRUSTED_DATASET_PATH,
                    )
                errors.extend(parse_errors[:10])
                imported += store_trusted_dataset_rows(rows)
            except Exception as exc:
                errors.append(f"local_trusted_dataset: {exc}")

        for dataset_url in TRUSTED_DATASET_URLS:
            source_name = extract_source_name(dataset_url)
            try:
                response = requests.get(dataset_url, timeout=TRUSTED_DATASET_TIMEOUT_SECONDS)
                response.raise_for_status()
                rows, parse_errors = parse_trusted_dataset_csv(response.text, source_name, dataset_url)
                errors.extend([f"{source_name}: {error}" for error in parse_errors[:10]])
                imported += store_trusted_dataset_rows(rows)
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")

        if imported:
            set_app_state_value("trusted_dataset_last_success", now.isoformat())
        set_app_state_value("trusted_dataset_last_error", "\n".join(errors))
        db.session.commit()

        return {
            "status": "updated" if imported else "empty",
            "imported": imported,
            "pending": TrustedDatasetEntry.query.filter_by(status="pending").count(),
            "approved": TrustedDatasetEntry.query.filter_by(status="approved").count(),
            "errors": errors,
        }
    finally:
        TRUSTED_DATASET_SYNC_IN_PROGRESS = False


def get_trusted_dataset_status():
    return {
        "pending": TrustedDatasetEntry.query.filter_by(status="pending").count(),
        "approved": TrustedDatasetEntry.query.filter_by(status="approved").count(),
        "rejected": TrustedDatasetEntry.query.filter_by(status="rejected").count(),
        "used": TrustedDatasetEntry.query.filter_by(used_in_training=True).count(),
        "last_success": get_app_state_value("trusted_dataset_last_success", "Never"),
        "last_attempt": get_app_state_value("trusted_dataset_last_attempt", "Never"),
        "last_error": get_app_state_value("trusted_dataset_last_error", ""),
        "refresh_hours": TRUSTED_DATASET_REFRESH_HOURS,
        "remote_sources": TRUSTED_DATASET_URLS,
        "local_path": LOCAL_TRUSTED_DATASET_PATH,
        "auto_approve": AUTO_APPROVE_TRUSTED_DATASETS,
    }


def lookup_feed_match(raw_url):
    candidates = get_lookup_candidates(raw_url)
    if not candidates:
        return None
    return ThreatFeedEntry.query.filter(ThreatFeedEntry.normalized_url.in_(candidates)).first()


def extract_subject(email_text):
    match = re.search(r"^Subject:\s*(.+)$", email_text or "", flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else "No Subject"


def extract_url(email_text):
    match = re.search(r"https?://[^\s<>\"]+", email_text or "", flags=re.IGNORECASE)
    if not match:
        return None
    return normalize_url(match.group(0).rstrip(".,);]"))


def extract_all_urls(email_text):
    urls = []
    for match in re.finditer(r"https?://[^\s<>\"]+", email_text or "", flags=re.IGNORECASE):
        normalized = normalize_url(match.group(0).rstrip(".,);]"))
        if normalized and normalized not in urls:
            urls.append(normalized)
    return urls


def build_sandbox_report(email_text):
    body = email_text or ""
    urls = extract_all_urls(body)
    script_blocks = len(re.findall(r"(?is)<script.*?>.*?</script>", body))
    html_links = len(re.findall(r'(?is)<a\s+[^>]*href=["\']([^"\']+)["\']', body))
    forms = len(re.findall(r"(?is)<form\b", body))
    password_fields = len(re.findall(r'(?is)<input\s+[^>]*type=["\']?password', body))
    blocked_items = len(urls) + script_blocks + html_links + forms + password_fields

    return {
        "urls": urls,
        "url_count": len(urls),
        "script_blocks": script_blocks,
        "html_links": html_links,
        "forms": forms,
        "password_fields": password_fields,
        "blocked_items": blocked_items,
        "safe_preview_mode": True,
    }


def build_nlp_analysis(scan):
    combined_text = f"{scan.body or ''}\n{scan.translated_body or ''}".lower()
    tokens = re.findall(r"\b[\w'-]{3,}\b", combined_text, flags=re.UNICODE)
    unique_tokens = sorted(set(tokens))
    suspicious_terms = [
        "verify",
        "password",
        "account",
        "login",
        "bank",
        "wallet",
        "urgent",
        "suspended",
        "blocked",
        "restore",
        "click",
        "confirm",
        "security",
        "akaunti",
        "dinani",
        "mutsimikizire",
        "chinsinsi",
    ]
    matched_terms = [
        term
        for term in suspicious_terms
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", combined_text, flags=re.IGNORECASE)
    ]
    urgency_terms = [term for term in ["urgent", "immediately", "now", "mwachangu", "suspended", "blocked"] if term in combined_text]
    credential_terms = [term for term in ["password", "login", "account", "verify", "confirm", "chinsinsi", "akaunti"] if term in combined_text]

    return {
        "token_count": len(tokens),
        "unique_token_count": len(unique_tokens),
        "matched_terms": matched_terms,
        "urgency_terms": urgency_terms,
        "credential_terms": credential_terms,
        "translation_used": bool(scan.translation_applied),
        "detected_language": scan.detected_language or "english",
        "feature_method": "Character n-gram TF-IDF",
        "decision_source": scan.decision_source.replace("_", " ").title(),
    }


def ensure_default_translation_glossary():
    if os.path.exists(LOCAL_TRANSLATION_GLOSSARY_PATH):
        return

    starter_terms = [
        ("akaunti", "account", "nyanja"),
        ("yatsekedwa", "suspended", "nyanja"),
        ("dinani", "click", "nyanja"),
        ("apa", "here", "nyanja"),
        ("mutsimikizire", "verify", "nyanja"),
        ("chinsinsi", "password", "nyanja"),
        ("mwachangu", "urgent", "nyanja"),
        ("banki", "bank", "nyanja"),
        ("moni", "hello", "nyanja"),
        ("msonkhano", "meeting", "nyanja"),
        ("mawa", "tomorrow", "nyanja"),
        ("lipoti", "report", "nyanja"),
        ("ndalama", "money", "nyanja"),
        ("tsegulani", "open", "nyanja"),
        ("tumizani", "send", "nyanja"),
        ("akaunti yanu", "your account", "nyanja"),
        ("password yanu", "your password", "mixed"),
    ]

    with open(LOCAL_TRANSLATION_GLOSSARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("source,english,language\n")
        for source, english, language in starter_terms:
            handle.write(f'"{source}","{english}","{language}"\n')


def load_translation_glossary():
    ensure_default_translation_glossary()
    terms = []
    try:
        with open(LOCAL_TRANSLATION_GLOSSARY_PATH, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.read().splitlines()[1:]
    except OSError:
        return terms

    for line in lines:
        parts = [part.strip().strip('"') for part in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            terms.append(
                {
                    "source": parts[0].lower(),
                    "english": parts[1],
                    "language": parts[2].lower() if len(parts) >= 3 and parts[2] else "local",
                }
            )
    return sorted(terms, key=lambda term: len(term["source"]), reverse=True)


def build_translation_candidate(body):
    terms = load_translation_glossary()
    if not terms or not body:
        return {
            "language": "english",
            "translated_body": None,
            "translation_applied": False,
            "translation_terms": 0,
        }

    translated = body
    matched_languages = {}
    replacement_count = 0
    for term in terms:
        pattern = re.compile(rf"(?<!\w){re.escape(term['source'])}(?!\w)", flags=re.IGNORECASE)
        translated, count = pattern.subn(term["english"], translated)
        if count:
            replacement_count += count
            matched_languages[term["language"]] = matched_languages.get(term["language"], 0) + count

    if not replacement_count or translated == body:
        return {
            "language": "english",
            "translated_body": None,
            "translation_applied": False,
            "translation_terms": 0,
        }

    detected_language = max(matched_languages, key=matched_languages.get) if matched_languages else "local"
    return {
        "language": detected_language,
        "translated_body": translated,
        "translation_applied": True,
        "translation_terms": replacement_count,
    }


def predict_with_ml(text):
    if not ML_READY or not text.strip():
        return None

    features = vectorizer.transform([text])
    predicted_label = int(model.predict(features)[0])
    result = "Phishing" if predicted_label == 1 else "Safe"
    score = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        score = round(float(probabilities[predicted_label]) * 100, 1)
    return {"result": result, "score": score}


def choose_text_prediction(original_prediction, translated_prediction):
    predictions = []
    if original_prediction:
        predictions.append(("ml", original_prediction))
    if translated_prediction:
        predictions.append(("translated_ml", translated_prediction))

    phishing_predictions = [
        (source, prediction)
        for source, prediction in predictions
        if prediction["result"] == "Phishing"
    ]
    if phishing_predictions:
        return max(phishing_predictions, key=lambda item: item[1]["score"])

    safe_predictions = [
        (source, prediction)
        for source, prediction in predictions
        if prediction["result"] == "Safe"
    ]
    if safe_predictions:
        return max(safe_predictions, key=lambda item: item[1]["score"])

    return None, None


def classify_content(body, extracted_url=None, fallback_result=None, fallback_score=None):
    feed_match = lookup_feed_match(extracted_url)
    if feed_match:
        translation = build_translation_candidate(body)
        return {
            "result": "Phishing",
            "score": 100.0,
            "decision_source": "threat_feed",
            "matched_feed_source": feed_match.source,
            "translation": translation,
            "original_text_score": None,
            "translated_text_score": None,
        }

    translation = build_translation_candidate(body)
    original_prediction = predict_with_ml(body)
    translated_prediction = None
    if translation["translation_applied"]:
        translated_prediction = predict_with_ml(translation["translated_body"])

    selected_source, selected_prediction = choose_text_prediction(original_prediction, translated_prediction)
    if selected_prediction:
        return {
            "result": selected_prediction["result"],
            "score": selected_prediction["score"],
            "decision_source": selected_source,
            "matched_feed_source": None,
            "translation": translation,
            "original_text_score": original_prediction["score"] if original_prediction else None,
            "translated_text_score": translated_prediction["score"] if translated_prediction else None,
        }

    if fallback_result:
        return {
            "result": normalize_result(fallback_result),
            "score": round(float(fallback_score or 0.0), 1),
            "decision_source": "upstream",
            "matched_feed_source": None,
            "translation": translation,
            "original_text_score": None,
            "translated_text_score": None,
        }

    return {
        "result": "Unavailable",
        "score": round(float(fallback_score or 0.0), 1),
        "decision_source": "manual_review",
        "matched_feed_source": None,
        "translation": translation,
        "original_text_score": None,
        "translated_text_score": None,
    }


def create_scan_record(body, sender, source, subject=None, fallback_result=None, fallback_score=None):
    normalized_subject = (subject or extract_subject(body) or "No Subject")[:255]
    extracted_url = extract_url(body)
    classification = classify_content(
        body,
        extracted_url=extracted_url,
        fallback_result=fallback_result,
        fallback_score=fallback_score,
    )
    translation = classification["translation"]
    scan = ScanRecord(
        sender=(sender or "unknown")[:255],
        subject=normalized_subject,
        body=body,
        result=classification["result"],
        score=classification["score"],
        source=source,
        review_status="unreviewed",
        review_label=None,
        is_quarantined=classification["result"] == "Phishing",
        extracted_url=extracted_url,
        decision_source=classification["decision_source"],
        matched_feed_source=classification["matched_feed_source"],
        detected_language=translation["language"],
        translated_body=translation["translated_body"],
        translation_applied=translation["translation_applied"],
        translation_terms=translation["translation_terms"],
        original_text_score=classification["original_text_score"],
        translated_text_score=classification["translated_text_score"],
    )
    db.session.add(scan)
    db.session.commit()
    return scan


def sanitize_email_body(raw_body):
    cleaned_body = re.sub(r"(?is)<script.*?>.*?</script>", "[SCRIPT BLOCKED]", raw_body or "")
    cleaned_body = re.sub(
        r'(?is)<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        lambda match: f"[LINK DISABLED: {match.group(1)}] {match.group(2)}",
        cleaned_body,
    )
    cleaned_body = re.sub(
        r"https?://[^\s<>\"]+",
        lambda match: f"[LINK DISABLED: {match.group(0)}]",
        cleaned_body,
        flags=re.IGNORECASE,
    )
    return escape(cleaned_body)


def sync_phishing_feeds(force=False):
    global FEED_SYNC_IN_PROGRESS

    if FEED_SYNC_IN_PROGRESS:
        return {"status": "busy", "added": 0, "total_urls": ThreatFeedEntry.query.count(), "errors": []}

    now = datetime.now()
    last_attempt_raw = get_app_state_value("phishing_feed_last_attempt")
    if not force and last_attempt_raw:
        try:
            last_attempt = datetime.fromisoformat(last_attempt_raw)
            if now - last_attempt < timedelta(minutes=PHISHING_FEED_REFRESH_MINUTES):
                return {
                    "status": "fresh",
                    "added": 0,
                    "total_urls": ThreatFeedEntry.query.count(),
                    "errors": [],
                }
        except ValueError:
            pass

    FEED_SYNC_IN_PROGRESS = True
    try:
        set_app_state_value("phishing_feed_last_attempt", now.isoformat())
        collected_urls = {}
        errors = []
        remote_success = False

        if os.path.exists(LOCAL_PHISHING_FEED_PATH):
            with open(LOCAL_PHISHING_FEED_PATH, "r", encoding="utf-8", errors="ignore") as handle:
                for url in parse_feed_urls(handle.read()):
                    collected_urls.setdefault(url, "local_snapshot")

        for feed_url in REMOTE_PHISHING_FEED_URLS:
            try:
                response = requests.get(feed_url, timeout=PHISHING_FEED_TIMEOUT_SECONDS)
                response.raise_for_status()
                remote_success = True
                source_name = extract_feed_name(feed_url)
                for url in parse_feed_urls(response.text):
                    collected_urls[url] = source_name
            except Exception as exc:
                errors.append(f"{extract_feed_name(feed_url)}: {exc}")

        added = 0
        if collected_urls:
            existing = {
                entry.normalized_url: entry
                for entry in ThreatFeedEntry.query.all()
            }

            for normalized_url, source_name in collected_urls.items():
                entry = existing.get(normalized_url)
                if entry:
                    entry.last_seen = now
                    entry.source = source_name
                else:
                    db.session.add(
                        ThreatFeedEntry(
                            normalized_url=normalized_url,
                            source=source_name,
                            first_seen=now,
                            last_seen=now,
                        )
                    )
                    added += 1

            if remote_success:
                with open(LOCAL_PHISHING_FEED_PATH, "w", encoding="utf-8") as handle:
                    for url in sorted(collected_urls):
                        handle.write(f"{url}\n")

            set_app_state_value("phishing_feed_last_success", now.isoformat())

        set_app_state_value("phishing_feed_last_error", "\n".join(errors))
        set_app_state_value("phishing_feed_total_urls", str(len(collected_urls)))
        db.session.commit()
        return {
            "status": "updated" if collected_urls else "empty",
            "added": added,
            "total_urls": ThreatFeedEntry.query.count(),
            "errors": errors,
        }
    finally:
        FEED_SYNC_IN_PROGRESS = False


def get_feed_status():
    return {
        "total_urls": ThreatFeedEntry.query.count(),
        "last_success": get_app_state_value("phishing_feed_last_success", "Never"),
        "last_attempt": get_app_state_value("phishing_feed_last_attempt", "Never"),
        "last_error": get_app_state_value("phishing_feed_last_error", ""),
        "refresh_minutes": PHISHING_FEED_REFRESH_MINUTES,
        "remote_sources": REMOTE_PHISHING_FEED_URLS,
        "snapshot_path": LOCAL_PHISHING_FEED_PATH,
    }


def user_can_access_scan(scan):
    return current_user.is_authenticated and (
        current_user.is_admin or scan.sender == current_user.username
    )


def find_accessible_scan(scan_id):
    scan = db.session.get(ScanRecord, scan_id)
    if not scan:
        flash("Scan record not found.", "warning")
        return None
    if not user_can_access_scan(scan):
        flash("You do not have permission to view that scan.", "danger")
        return None
    return scan


@app.before_request
def auto_refresh_threat_feed():
    if request.endpoint == "static" or not AUTO_PHISHING_FEED_SYNC:
        return
    try:
        sync_phishing_feeds(force=False)
    except Exception as exc:
        app.logger.warning("Automatic phishing-feed refresh failed: %s", exc)


@app.before_request
def auto_refresh_trusted_datasets():
    if request.endpoint == "static" or not AUTO_TRUSTED_DATASET_SYNC:
        return
    try:
        import_trusted_datasets(force=False)
    except Exception as exc:
        app.logger.warning("Automatic trusted-dataset import failed: %s", exc)


@app.context_processor
def inject_shell_state():
    if not current_user.is_authenticated:
        return {
            "unread_notification_count": 0,
            "latest_notifications": [],
            "max_login_attempts": MAX_LOGIN_ATTEMPTS,
        }

    latest_notifications = (
        Notification.query.filter_by(recipient_user_id=current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(5)
        .all()
    )
    unread_count = Notification.query.filter_by(
        recipient_user_id=current_user.id,
        is_read=False,
    ).count()
    return {
        "unread_notification_count": unread_count,
        "latest_notifications": latest_notifications,
        "max_login_attempts": MAX_LOGIN_ATTEMPTS,
    }


def get_dashboard_stats(viewer=None):
    if viewer and viewer.is_authenticated and not viewer.is_admin:
        scans = (
            ScanRecord.query.filter_by(sender=viewer.username)
            .order_by(ScanRecord.timestamp.desc())
            .all()
        )
    else:
        scans = ScanRecord.query.order_by(ScanRecord.timestamp.desc()).all()
    users = User.query.filter_by(is_deleted=False).all()
    total = len(scans)
    phishing_count = sum(scan.result == "Phishing" for scan in scans)
    reviewed = [scan for scan in scans if scan.review_status == "reviewed" and scan.review_label in (0, 1)]
    if viewer and viewer.is_authenticated and not viewer.is_admin:
        pending_reviews = sum(scan.review_status == "queued" for scan in scans)
    else:
        pending_reviews = sum(scan.review_status in {"queued", "unreviewed"} for scan in scans)
    last_scan = scans[0].timestamp.strftime("%Y-%m-%d %H:%M:%S") if scans else "Never"
    pending_access_requests = sum((not user.is_approved) and (not user.is_rejected) for user in users)

    reviewed_total = len(reviewed)
    reviewed_correct = sum(
        (scan.predicted_label is not None) and (scan.predicted_label == scan.review_label)
        for scan in reviewed
    )
    accuracy = round((reviewed_correct / reviewed_total) * 100, 1) if reviewed_total else 0.0
    health = 100 if ML_READY else 45

    return {
        "emails_scanned": total,
        "phishing": phishing_count,
        "threats_blocked": phishing_count,
        "safe": total - phishing_count,
        "last_scan": last_scan,
        "pending_reviews": pending_reviews,
        "pending_access_requests": pending_access_requests,
        "approved_users": sum(user.is_approved for user in users),
        "reviewed_total": reviewed_total,
        "review_accuracy": accuracy,
        "health": health,
        "recent_scans": scans[:5],
    }


def calculate_metrics():
    reviewed_scans = ScanRecord.query.filter_by(review_status="reviewed").all()
    metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for scan in reviewed_scans:
        predicted = scan.predicted_label
        actual = scan.review_label
        if predicted is None or actual not in (0, 1):
            continue
        if predicted == 1 and actual == 1:
            metrics["tp"] += 1
        elif predicted == 0 and actual == 0:
            metrics["tn"] += 1
        elif predicted == 1 and actual == 0:
            metrics["fp"] += 1
        elif predicted == 0 and actual == 1:
            metrics["fn"] += 1

    total = sum(metrics.values())
    precision_base = metrics["tp"] + metrics["fp"]
    recall_base = metrics["tp"] + metrics["fn"]
    fpr_base = metrics["fp"] + metrics["tn"]
    accuracy = round(((metrics["tp"] + metrics["tn"]) / total) * 100, 1) if total else 0.0
    precision = round((metrics["tp"] / precision_base) * 100, 1) if precision_base else 0.0
    recall = round((metrics["tp"] / recall_base) * 100, 1) if recall_base else 0.0
    f1 = round(
        (2 * precision * recall) / (precision + recall),
        1,
    ) if (precision + recall) else 0.0
    fpr = round((metrics["fp"] / fpr_base) * 100, 1) if fpr_base else 0.0

    metrics.update(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "total_eval": total,
        }
    )
    return metrics


def retrain_model():
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:
        raise RuntimeError(f"Training dependencies are missing: {exc}") from exc

    datasets = []
    if os.path.exists(DATASET_PATH):
        datasets.append(pd.read_csv(DATASET_PATH))
    if os.path.exists(LOCAL_LANGUAGE_DATASET_PATH):
        datasets.append(pd.read_csv(LOCAL_LANGUAGE_DATASET_PATH))

    reviewed_scans = ScanRecord.query.filter_by(review_status="reviewed").all()
    feedback_rows = []
    for scan in reviewed_scans:
        if scan.review_label in (0, 1) and scan.body.strip():
            feedback_rows.append({"text": scan.body, "label": int(scan.review_label)})
        if scan.review_label in (0, 1) and scan.translated_body and scan.translated_body.strip():
            feedback_rows.append({"text": scan.translated_body, "label": int(scan.review_label)})
    if feedback_rows:
        datasets.append(pd.DataFrame(feedback_rows))

    trusted_entries = TrustedDatasetEntry.query.filter_by(status="approved").all()
    trusted_rows = [
        {"text": entry.text, "label": int(entry.label)}
        for entry in trusted_entries
        if entry.label in (0, 1) and entry.text.strip()
    ]
    if trusted_rows:
        datasets.append(pd.DataFrame(trusted_rows))

    if not datasets:
        raise RuntimeError("No training data is available yet.")

    combined = pd.concat(datasets, ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].astype(str)
    combined["label"] = combined["label"].astype(int)

    if combined.empty or combined["label"].nunique() < 2:
        raise RuntimeError("Retraining needs both safe and phishing examples.")

    new_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=8000,
    )
    features = new_vectorizer.fit_transform(combined["text"])
    new_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    new_model.fit(features, combined["label"])

    joblib.dump(new_model, MODEL_PATH)
    joblib.dump(new_vectorizer, VECTORIZER_PATH)
    load_ml_assets()
    db.session.add(
        ModelVersion(
            rows=len(combined),
            feedback_rows=len(feedback_rows),
            trusted_rows=len(trusted_rows),
            status="active",
            message="Model retrained from base, local, reviewed, and approved trusted samples.",
        )
    )
    return {
        "rows": len(combined),
        "feedback_rows": len(feedback_rows),
        "trusted_rows": len(trusted_rows),
        "feature_mode": "char_wb_3_5",
    }


def bootstrap_database():
    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    db.create_all()

    with db.engine.begin() as connection:
        inspector = inspect(connection)
        table_names = inspector.get_table_names()

        if "user" in table_names:
            user_columns = {column["name"] for column in inspector.get_columns("user")}
            user_expected_columns = {
                "full_name": "ALTER TABLE user ADD COLUMN full_name VARCHAR(150)",
                "email": "ALTER TABLE user ADD COLUMN email VARCHAR(255)",
                "organization": "ALTER TABLE user ADD COLUMN organization VARCHAR(150)",
                "department": "ALTER TABLE user ADD COLUMN department VARCHAR(150)",
                "job_title": "ALTER TABLE user ADD COLUMN job_title VARCHAR(150)",
                "is_admin": "ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0",
                "is_approved": "ALTER TABLE user ADD COLUMN is_approved BOOLEAN DEFAULT 1",
                "is_rejected": "ALTER TABLE user ADD COLUMN is_rejected BOOLEAN DEFAULT 0",
                "decision_note": "ALTER TABLE user ADD COLUMN decision_note TEXT",
                "failed_login_attempts": "ALTER TABLE user ADD COLUMN failed_login_attempts INTEGER DEFAULT 0",
                "is_locked": "ALTER TABLE user ADD COLUMN is_locked BOOLEAN DEFAULT 0",
                "locked_at": "ALTER TABLE user ADD COLUMN locked_at DATETIME",
                "is_deleted": "ALTER TABLE user ADD COLUMN is_deleted BOOLEAN DEFAULT 0",
            }
            for column_name, statement in user_expected_columns.items():
                if column_name not in user_columns:
                    connection.execute(text(statement))

            # Backward compatible defaults for existing rows (SQLite may leave NULLs).
            connection.execute(text("UPDATE user SET is_approved = 1 WHERE is_approved IS NULL"))
            connection.execute(text("UPDATE user SET is_rejected = 0 WHERE is_rejected IS NULL"))
            connection.execute(text("UPDATE user SET is_admin = 0 WHERE is_admin IS NULL"))
            connection.execute(text("UPDATE user SET failed_login_attempts = 0 WHERE failed_login_attempts IS NULL"))
            connection.execute(text("UPDATE user SET is_locked = 0 WHERE is_locked IS NULL"))
            connection.execute(text("UPDATE user SET is_deleted = 0 WHERE is_deleted IS NULL"))

        if "scan_record" in table_names:
            scan_columns = {column["name"] for column in inspector.get_columns("scan_record")}
            expected_columns = {
                "subject": "ALTER TABLE scan_record ADD COLUMN subject VARCHAR(255) DEFAULT 'No Subject'",
                "source": "ALTER TABLE scan_record ADD COLUMN source VARCHAR(32) DEFAULT 'manual'",
                "review_status": "ALTER TABLE scan_record ADD COLUMN review_status VARCHAR(32) DEFAULT 'unreviewed'",
                "review_label": "ALTER TABLE scan_record ADD COLUMN review_label INTEGER",
                "is_quarantined": "ALTER TABLE scan_record ADD COLUMN is_quarantined BOOLEAN DEFAULT 0",
                "extracted_url": "ALTER TABLE scan_record ADD COLUMN extracted_url VARCHAR(2048)",
                "decision_source": "ALTER TABLE scan_record ADD COLUMN decision_source VARCHAR(64) DEFAULT 'ml'",
                "matched_feed_source": "ALTER TABLE scan_record ADD COLUMN matched_feed_source VARCHAR(255)",
                "feedback_value": "ALTER TABLE scan_record ADD COLUMN feedback_value VARCHAR(32)",
                "feedback_submitted_at": "ALTER TABLE scan_record ADD COLUMN feedback_submitted_at DATETIME",
                "used_in_training": "ALTER TABLE scan_record ADD COLUMN used_in_training BOOLEAN DEFAULT 0",
                "detected_language": "ALTER TABLE scan_record ADD COLUMN detected_language VARCHAR(64)",
                "translated_body": "ALTER TABLE scan_record ADD COLUMN translated_body TEXT",
                "translation_applied": "ALTER TABLE scan_record ADD COLUMN translation_applied BOOLEAN DEFAULT 0",
                "translation_terms": "ALTER TABLE scan_record ADD COLUMN translation_terms INTEGER DEFAULT 0",
                "original_text_score": "ALTER TABLE scan_record ADD COLUMN original_text_score FLOAT",
                "translated_text_score": "ALTER TABLE scan_record ADD COLUMN translated_text_score FLOAT",
            }
            for column_name, statement in expected_columns.items():
                if column_name not in scan_columns:
                    connection.execute(text(statement))

            connection.execute(text("UPDATE scan_record SET used_in_training = 0 WHERE used_in_training IS NULL"))
            connection.execute(text("UPDATE scan_record SET translation_applied = 0 WHERE translation_applied IS NULL"))
            connection.execute(text("UPDATE scan_record SET translation_terms = 0 WHERE translation_terms IS NULL"))


@app.route("/")
def home():
    stats = get_dashboard_stats(current_user)
    return render_template(
        "index.html",
        stats=stats,
        health=stats["health"],
        ml_ready=ML_READY,
        ml_error=ML_ERROR,
        recent_scans=stats["recent_scans"],
        feed_status=get_feed_status(),
    )


@app.route("/check", methods=["GET", "POST"])
@login_required
def check_email():
    if request.method == "POST":
        content = request.form.get("email_text", "").strip()
        if not content:
            flash("Paste the email body before scanning.", "warning")
            return redirect(url_for("check_email"))

        scan = create_scan_record(
            body=content,
            sender=current_user.username,
            source="manual",
        )
        if scan.decision_source == "threat_feed":
            flash(
                "Known malicious URL matched the phishing threat feed and was blocked immediately.",
                "danger",
            )
        elif not ML_READY and scan.decision_source == "manual_review":
            flash(
                "Model files are unavailable, so the scan was stored for review instead of classified by AI.",
                "warning",
            )
        return redirect(url_for("sandbox_results", scan_id=scan.id))

    return render_template("check.html", ml_ready=ML_READY, ml_error=ML_ERROR)


@app.route("/sandbox-results/<scan_id>")
@login_required
def sandbox_results(scan_id):
    scan = find_accessible_scan(scan_id)
    if not scan:
        return redirect(url_for("check_email"))
    return render_template(
        "sandbox_results.html",
        scan=scan,
        nlp_analysis=build_nlp_analysis(scan),
        sandbox_report=build_sandbox_report(scan.body),
    )


@app.route("/safe-preview/<scan_id>")
@login_required
def safe_preview(scan_id):
    scan = find_accessible_scan(scan_id)
    if not scan:
        return redirect(url_for("sandbox_history"))
    return render_template(
        "safe_preview.html",
        scan=scan,
        sanitized_body=sanitize_email_body(scan.body),
        sandbox_report=build_sandbox_report(scan.body),
        nlp_analysis=build_nlp_analysis(scan),
    )


@app.route("/sandbox-history")
@login_required
def sandbox_history():
    scans = (
        ScanRecord.query.filter_by(sender=current_user.username)
        .order_by(ScanRecord.timestamp.desc())
        .all()
    )
    return render_template("sandbox_history.html", scans=scans)


@app.route("/notifications")
@login_required
def notifications():
    items = (
        Notification.query.filter_by(recipient_user_id=current_user.id)
        .order_by(Notification.created_at.desc())
        .all()
    )
    return render_template("notifications.html", notifications=items)


@app.route("/notifications/mark-all-read", methods=["POST"])
@login_required
def mark_all_notifications_read():
    Notification.query.filter_by(
        recipient_user_id=current_user.id,
        is_read=False,
    ).update({"is_read": True})
    db.session.commit()
    flash("All notifications have been marked as read.", "success")
    return redirect(url_for("notifications"))


@app.route("/notifications/<int:notification_id>/open")
@login_required
def open_notification(notification_id):
    notification = db.session.get(Notification, notification_id)
    if not notification or notification.recipient_user_id != current_user.id:
        flash("Notification not found.", "warning")
        return redirect(url_for("notifications"))

    notification.is_read = True
    db.session.commit()
    return redirect(notification.link or url_for("notifications"))


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not check_password_hash(current_user.password, current_password):
            flash("Your current password is not correct.", "danger")
            return redirect(url_for("change_password"))

        if new_password != confirm_password:
            flash("New password and confirmation do not match.", "warning")
            return redirect(url_for("change_password"))

        if current_password == new_password:
            flash("Choose a new password that is different from the current one.", "warning")
            return redirect(url_for("change_password"))

        password_issues = validate_password_strength(new_password)
        if password_issues:
            flash(password_issues[0], "warning")
            return redirect(url_for("change_password"))

        current_user.password = generate_password_hash(new_password)
        current_user.failed_login_attempts = 0
        current_user.is_locked = False
        current_user.locked_at = None
        create_notification(
            current_user.id,
            "Password changed",
            "Your PhishGuard password was changed successfully. If this was not you, contact an administrator immediately.",
            category="success",
            link=url_for("notifications"),
        )
        db.session.commit()
        flash("Your password has been updated successfully.", "success")
        return redirect(url_for("notifications"))

    return render_template("change_password.html")


@app.route("/feedback", methods=["POST"])
@login_required
def feedback():
    scan_id = request.form.get("scan_id", "").strip()
    feedback_value = request.form.get("feedback", "").strip().lower()
    scan = find_accessible_scan(scan_id)
    if not scan:
        return redirect(url_for("sandbox_history"))

    if feedback_value not in {"correct", "wrong"}:
        flash("Invalid feedback selection.", "danger")
        return redirect(url_for("sandbox_results", scan_id=scan.id))

    scan.review_status = "queued"
    scan.feedback_value = feedback_value
    scan.feedback_submitted_at = datetime.now()
    notify_admins(
        "Scan feedback submitted",
        f"{current_user.username} marked scan '{scan.subject}' as {feedback_value}.",
        category="warning" if feedback_value == "wrong" else "info",
        link=url_for("review_feedback"),
        exclude_user_id=current_user.id if current_user.is_admin else None,
    )
    db.session.commit()
    flash("Feedback recorded. An administrator can now review this scan.", "success")
    return redirect(url_for("sandbox_results", scan_id=scan.id))


@app.route("/api/report-scan", methods=["POST"])
def report_scan():
    payload = request.get_json(silent=True) or {}
    body = (payload.get("body") or payload.get("email_text") or "").strip()
    if not body:
        return jsonify({"error": "body is required"}), 400

    scan = create_scan_record(
        body=body,
        sender=(payload.get("sender") or payload.get("mailfrom") or "smtp-proxy"),
        subject=payload.get("subject"),
        source="smtp",
        fallback_result=payload.get("result"),
        fallback_score=payload.get("score"),
    )
    return jsonify({"status": "stored", "scan_id": scan.id, "result": scan.result}), 201


@app.route("/quarantine")
@login_required
@admin_required
def quarantine():
    items = (
        ScanRecord.query.filter_by(is_quarantined=True)
        .order_by(ScanRecord.timestamp.desc())
        .all()
    )
    return render_template("quarantine.html", items=items)


@app.route("/live-monitor")
@login_required
@admin_required
def live_monitor():
    scans = ScanRecord.query.order_by(ScanRecord.timestamp.desc()).limit(50).all()
    return render_template("live_monitor.html", scans=scans)


@app.route("/review-feedback", methods=["GET", "POST"])
@login_required
@admin_required
def review_feedback():
    if request.method == "POST":
        approved_ids = set(request.form.getlist("approve"))
        rejected_ids = set(request.form.getlist("reject"))
        all_selected = approved_ids | rejected_ids

        if not all_selected:
            flash("Select at least one review action.", "warning")
            return redirect(url_for("review_feedback"))

        scans = ScanRecord.query.filter(ScanRecord.id.in_(all_selected)).all()
        processed = 0
        for scan in scans:
            predicted_label = scan.predicted_label
            if predicted_label is None:
                continue
            if scan.id in approved_ids:
                scan.review_label = predicted_label
                scan.review_status = "reviewed"
                scan.used_in_training = False
                processed += 1
            elif scan.id in rejected_ids:
                scan.review_label = 0 if predicted_label == 1 else 1
                scan.review_status = "reviewed"
                scan.used_in_training = False
                processed += 1

        db.session.commit()
        auto_retrain_result = maybe_auto_retrain("reviewed feedback")
        if auto_retrain_result and auto_retrain_result["status"] == "success":
            flash("Automatic retraining completed after this review batch.", "info")
        elif auto_retrain_result and auto_retrain_result["status"] == "failed":
            flash("Review was saved, but automatic retraining failed. Check Reports for details.", "warning")
        flash(f"Processed {processed} review item(s).", "success")
        return redirect(url_for("review_feedback"))

    scans = (
        ScanRecord.query.filter(ScanRecord.review_status.in_(["queued", "unreviewed"]))
        .order_by(
            text(
                "CASE review_status WHEN 'queued' THEN 0 WHEN 'unreviewed' THEN 1 ELSE 2 END, timestamp DESC"
            )
        )
        .all()
    )
    queued_scans = [scan for scan in scans if scan.review_status == "queued"]
    unreviewed_scans = [scan for scan in scans if scan.review_status == "unreviewed"]
    return render_template(
        "review_feedback.html",
        scans=scans,
        queued_scans=queued_scans,
        unreviewed_scans=unreviewed_scans,
        auto_retrain_status=get_auto_retrain_status(),
    )


@app.route("/reports")
@login_required
@admin_required
def reports():
    return render_template(
        "reports.html",
        stats=get_dashboard_stats(),
        metrics=calculate_metrics(),
        feed_status=get_feed_status(),
        auto_retrain_status=get_auto_retrain_status(),
        trusted_dataset_status=get_trusted_dataset_status(),
    )


@app.route("/manage-keywords", methods=["GET", "POST"])
@login_required
@admin_required
def manage_keywords():
    if request.method == "POST":
        summary = sync_phishing_feeds(force=True)
        if summary["errors"]:
            flash(
                f"Threat feeds synced with {summary['total_urls']} URLs, but some sources failed.",
                "warning",
            )
        else:
            flash(
                f"Threat feeds updated successfully. {summary['total_urls']} phishing URLs are now tracked.",
                "success",
            )
        return redirect(url_for("manage_keywords"))

    recent_urls = ThreatFeedEntry.query.order_by(ThreatFeedEntry.last_seen.desc()).limit(25).all()
    return render_template(
        "manage_keywords.html",
        feed_status=get_feed_status(),
        recent_urls=recent_urls,
    )


@app.route("/trusted-datasets", methods=["GET", "POST"])
@login_required
@admin_required
def trusted_datasets():
    if request.method == "POST":
        action = request.form.get("action", "").strip().lower()
        if action == "sync":
            summary = import_trusted_datasets(force=True)
            if summary["errors"]:
                flash(
                    f"Trusted datasets synced with {summary['imported']} new sample(s), but some rows or sources failed.",
                    "warning",
                )
            else:
                flash(f"Trusted datasets synced. {summary['imported']} new sample(s) imported.", "success")
            return redirect(url_for("trusted_datasets"))

        approved_ids = {int(item_id) for item_id in request.form.getlist("approve") if item_id.isdigit()}
        rejected_ids = {int(item_id) for item_id in request.form.getlist("reject") if item_id.isdigit()}
        selected_ids = approved_ids | rejected_ids
        if not selected_ids:
            flash("Select at least one trusted dataset action.", "warning")
            return redirect(url_for("trusted_datasets"))

        entries = TrustedDatasetEntry.query.filter(TrustedDatasetEntry.id.in_(selected_ids)).all()
        approved_count = 0
        rejected_count = 0
        for entry in entries:
            if entry.id in approved_ids:
                entry.status = "approved"
                entry.reviewed_at = datetime.now()
                entry.reviewed_by_user_id = current_user.id
                entry.used_in_training = False
                entry.decision_note = "Approved for model retraining."
                approved_count += 1
            elif entry.id in rejected_ids:
                entry.status = "rejected"
                entry.reviewed_at = datetime.now()
                entry.reviewed_by_user_id = current_user.id
                entry.decision_note = "Rejected by administrator."
                rejected_count += 1

        db.session.commit()
        auto_retrain_result = maybe_auto_retrain("approved trusted dataset samples")
        if auto_retrain_result and auto_retrain_result["status"] == "success":
            flash("Automatic retraining completed after approving trusted samples.", "info")
        elif auto_retrain_result and auto_retrain_result["status"] == "failed":
            flash("Trusted samples were saved, but automatic retraining failed. Check Reports for details.", "warning")
        flash(f"Approved {approved_count} and rejected {rejected_count} trusted sample(s).", "success")
        return redirect(url_for("trusted_datasets"))

    pending_entries = (
        TrustedDatasetEntry.query.filter_by(status="pending")
        .order_by(TrustedDatasetEntry.imported_at.desc())
        .limit(100)
        .all()
    )
    recent_entries = (
        TrustedDatasetEntry.query.order_by(TrustedDatasetEntry.imported_at.desc())
        .limit(25)
        .all()
    )
    return render_template(
        "trusted_datasets.html",
        pending_entries=pending_entries,
        recent_entries=recent_entries,
        trusted_dataset_status=get_trusted_dataset_status(),
    )


@app.route("/retrain-now")
@login_required
@admin_required
def retrain_now():
    try:
        outcome = retrain_model()
        for scan in get_pending_training_scans():
            scan.used_in_training = True
        for entry in get_pending_trusted_training_entries():
            entry.used_in_training = True
        set_app_state_value("auto_retrain_last_status", "manual")
        set_app_state_value(
            "auto_retrain_last_message",
            f"Manual retraining completed using {outcome['rows']} examples.",
        )
        set_app_state_value(
            "auto_retrain_last_success",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        db.session.commit()
        flash(
            f"Model retraining complete using {outcome['rows']} examples "
            f"({outcome['feedback_rows']} reviewed scans, {outcome['trusted_rows']} trusted samples).",
            "success",
        )
    except Exception as exc:
        flash(f"Retraining failed: {exc}", "danger")
    return redirect(url_for("reports"))


@app.route("/manage-users", methods=["GET", "POST"])
@login_required
@admin_required
def manage_users():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        organization = request.form.get("organization", "").strip()
        department = request.form.get("department", "").strip()
        job_title = request.form.get("job_title", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        is_admin = request.form.get("is_admin") == "on"

        if not all([full_name, email, organization, department, job_title, username, password]):
            flash("All user profile fields are required.", "warning")
            return redirect(url_for("manage_users"))

        if not validate_email_address(email):
            flash("Enter a valid work email address.", "warning")
            return redirect(url_for("manage_users"))

        password_issues = validate_password_strength(password)
        if password_issues:
            flash(password_issues[0], "warning")
            return redirect(url_for("manage_users"))

        conflicts = find_user_identity_conflicts(username, email)
        if conflicts:
            flash("That username or email already exists.", "danger")
            return redirect(url_for("manage_users"))

        user = User(
            full_name=full_name,
            email=email,
            organization=organization,
            department=department,
            job_title=job_title,
            username=username,
            password=generate_password_hash(password),
            is_admin=is_admin,
            is_approved=True,
            is_rejected=False,
            decision_note="Provisioned directly by an administrator.",
        )
        db.session.add(user)
        db.session.flush()
        create_notification(
            user.id,
            "Account created",
            "Your PhishGuard access was created by an administrator. You can sign in immediately.",
            category="success",
            link=url_for("login"),
        )
        db.session.commit()
        flash("User created successfully.", "success")
        return redirect(url_for("manage_users"))

    users = (
        User.query.filter_by(is_deleted=False)
        .order_by(User.is_approved.asc(), User.username.asc())
        .all()
    )
    return render_template("manage_users.html", users=users)


@app.route("/approve-user/<int:user_id>")
@login_required
@admin_required
def approve_user(user_id):
    user = db.session.get(User, user_id)
    if not user or bool(user.is_deleted):
        flash("User not found.", "warning")
        return redirect(url_for("manage_users"))
    user.is_approved = True
    user.is_rejected = False
    user.failed_login_attempts = 0
    user.is_locked = False
    user.locked_at = None
    user.decision_note = "Your access request was approved. You can now sign in to PhishGuard."
    create_notification(
        user.id,
        "Access approved",
        user.decision_note,
        category="success",
        link=url_for("login"),
    )
    db.session.commit()
    flash(f"{user.username} is now approved.", "success")
    return redirect(url_for("manage_users"))


@app.route("/reject-user/<int:user_id>")
@login_required
@admin_required
def reject_user(user_id):
    user = db.session.get(User, user_id)
    if not user or bool(user.is_deleted):
        flash("User not found.", "warning")
        return redirect(url_for("manage_users"))
    user.is_approved = False
    user.is_rejected = True
    user.decision_note = "Your access request was rejected. Please contact the security administrator."
    create_notification(
        user.id,
        "Access request rejected",
        user.decision_note,
        category="danger",
        link=url_for("login"),
    )
    db.session.commit()
    flash(f"{user.username} has been rejected.", "warning")
    return redirect(url_for("manage_users"))


@app.route("/admin-reset-password/<int:user_id>", methods=["GET", "POST"])
@login_required
@admin_required
def admin_reset_password(user_id):
    user = db.session.get(User, user_id)
    if not user or bool(user.is_deleted):
        flash("User not found.", "warning")
        return redirect(url_for("manage_users"))

    if request.method == "POST":
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if new_password != confirm_password:
            flash("New password and confirmation do not match.", "warning")
            return redirect(url_for("admin_reset_password", user_id=user.id))

        password_issues = validate_password_strength(new_password)
        if password_issues:
            flash(password_issues[0], "warning")
            return redirect(url_for("admin_reset_password", user_id=user.id))

        user.password = generate_password_hash(new_password)
        user.failed_login_attempts = 0
        user.is_locked = False
        user.locked_at = None
        create_notification(
            user.id,
            "Password reset by administrator",
            "Your PhishGuard password was reset by an administrator. Use the new password provided by your security team and change it after login.",
            category="warning",
            link=url_for("login"),
        )
        db.session.commit()
        flash(f"Password reset for {user.username}. The account is now unlocked.", "success")
        return redirect(url_for("manage_users"))

    return render_template("admin_reset_password.html", managed_user=user)


@app.route("/delete-user/<int:user_id>")
@login_required
@admin_required
def delete_user(user_id):
    user = db.session.get(User, user_id)
    if not user or bool(user.is_deleted):
        flash("User not found.", "warning")
        return redirect(url_for("manage_users"))
    if user.id == current_user.id:
        flash("You cannot delete your own account while logged in.", "warning")
        return redirect(url_for("manage_users"))

    archive_user_account(user)
    db.session.commit()
    flash("User archived. Their old details are now free for a new access request.", "info")
    return redirect(url_for("manage_users"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        organization = request.form.get("organization", "").strip()
        department = request.form.get("department", "").strip()
        job_title = request.form.get("job_title", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not all([full_name, email, organization, department, job_title, username, password]):
            flash("All registration fields are required.", "warning")
            return redirect(url_for("register"))

        if not validate_email_address(email):
            flash("Enter a valid work email address.", "warning")
            return redirect(url_for("register"))

        password_issues = validate_password_strength(password)
        if password_issues:
            flash(password_issues[0], "warning")
            return redirect(url_for("register"))

        conflicts = find_user_identity_conflicts(username, email)
        active_user_count = User.query.filter_by(is_deleted=False).count()
        first_user = active_user_count == 0

        reusable_user = next((candidate for candidate in conflicts if can_refresh_access_request(candidate)), None)
        blocking_conflict = next(
            (
                candidate
                for candidate in conflicts
                if not reusable_user or candidate.id != reusable_user.id
            ),
            None,
        )

        if blocking_conflict and not can_refresh_access_request(blocking_conflict):
            flash("That username or email is already taken.", "danger")
            return redirect(url_for("register"))

        if reusable_user and blocking_conflict:
            flash("That username or email is already taken by another active account.", "danger")
            return redirect(url_for("register"))

        if reusable_user:
            user = reusable_user
            user.full_name = full_name
            user.email = email
            user.organization = organization
            user.department = department
            user.job_title = job_title
            user.username = username
            user.password = generate_password_hash(password)
            user.is_admin = False
            user.is_approved = False
            user.is_rejected = False
            user.is_locked = False
            user.locked_at = None
            user.failed_login_attempts = 0
            user.decision_note = "Registration resubmitted. Waiting for administrator approval."
            create_notification(
                user.id,
                "Registration resubmitted",
                "Your access request was resubmitted and is waiting for administrator approval.",
                category="info",
                link=url_for("login"),
            )
            notify_admins_of_signup(user)
            db.session.commit()
            flash("Registration submitted again. Wait for an administrator to approve your access.", "info")
            return redirect(url_for("login"))

        user = User(
            full_name=full_name,
            email=email,
            organization=organization,
            department=department,
            job_title=job_title,
            username=username,
            password=generate_password_hash(password),
            is_admin=first_user,
            is_approved=first_user,
            is_rejected=False,
            is_deleted=False,
            decision_note=(
                "First account bootstrap completed. You are the administrator."
                if first_user
                else "Registration submitted. Waiting for administrator approval."
            ),
        )
        db.session.add(user)
        db.session.flush()

        if first_user:
            create_notification(
                user.id,
                "Administrator access granted",
                "Your account was created as the first administrator for PhishGuard.",
                category="success",
                link=url_for("home"),
            )
        else:
            create_notification(
                user.id,
                "Registration received",
                "Your access request is pending review by an administrator.",
                category="info",
                link=url_for("login"),
            )
            notify_admins_of_signup(user)
        db.session.commit()

        if first_user:
            flash("First account created. You are now the approved administrator.", "success")
        else:
            flash("Registration submitted. Wait for an administrator to approve your access.", "info")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/release/<scan_id>")
@login_required
@admin_required
def release(scan_id):
    scan = db.session.get(ScanRecord, scan_id)
    if not scan:
        flash("Threat record not found.", "warning")
    else:
        scan.is_quarantined = False
        db.session.commit()
        flash("Message released from quarantine.", "success")
    return redirect(url_for("quarantine"))


@app.route("/delete-threat/<scan_id>")
@login_required
@admin_required
def delete_threat(scan_id):
    scan = db.session.get(ScanRecord, scan_id)
    if not scan:
        flash("Threat record not found.", "warning")
    else:
        db.session.delete(scan)
        db.session.commit()
        flash("Threat record deleted.", "info")
    return redirect(url_for("quarantine"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter(
            User.is_deleted.is_(False),
            or_(
                func.lower(User.username) == username,
                func.lower(User.email) == username,
            )
        ).first()

        if not user:
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        if bool(user.is_locked):
            flash(
                f"Your account is locked after too many failed password attempts. "
                f"An administrator must reset it.",
                "danger",
            )
            return render_template("login.html")

        if not check_password_hash(user.password, password):
            user.failed_login_attempts = int(user.failed_login_attempts or 0) + 1
            if user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
                user.is_locked = True
                user.locked_at = datetime.now()
                notify_admins(
                    "Account locked",
                    f"{get_user_display_name(user)} was locked after reaching the maximum failed login limit.",
                    category="danger",
                    link=url_for("manage_users"),
                )
            db.session.commit()
            remaining_attempts = max(0, MAX_LOGIN_ATTEMPTS - int(user.failed_login_attempts or 0))
            if bool(user.is_locked):
                flash(
                    f"Your account is locked after {MAX_LOGIN_ATTEMPTS} failed password attempts. "
                    f"Contact an administrator for a password reset.",
                    "danger",
                )
            else:
                flash(
                    f"Invalid username or password. {remaining_attempts} login attempt(s) remaining before lockout.",
                    "danger",
                )
            return render_template("login.html")

        if bool(user.is_rejected):
            flash(get_latest_user_notice(user) or "Your access request was rejected.", "danger")
            return render_template("login.html")

        if not bool(user.is_approved):
            flash(get_latest_user_notice(user) or "Your account is still pending administrator approval.", "warning")
            return render_template("login.html")

        user.failed_login_attempts = 0
        user.is_locked = False
        user.locked_at = None
        db.session.commit()
        login_user(user)
        flash("Welcome back.", "success")
        next_url = request.args.get("next")
        return redirect(next_url or url_for("home"))

    return render_template("login.html")


def initialize_app():
    with app.app_context():
        bootstrap_database()
        ensure_default_translation_glossary()
        load_ml_assets()
        try:
            sync_phishing_feeds(force=True)
        except Exception as exc:
            app.logger.warning("Initial phishing-feed sync failed: %s", exc)
        try:
            import_trusted_datasets(force=True)
        except Exception as exc:
            app.logger.warning("Initial trusted-dataset import failed: %s", exc)


initialize_app()


if __name__ == "__main__":
    app.run(debug=True)
