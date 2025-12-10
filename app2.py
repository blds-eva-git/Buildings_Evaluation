from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import logging
import math
import os
import sqlite3
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

from flask import (
    Flask,
    Response,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

try:
    from PIL import Image, ImageOps, ExifTags
except Exception:  # pragma: no cover - fallback
    Image = None

# =============================================
# Configuration
# =============================================

APP_NAME = "BuildingSurveyPro"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "building_survey_refactored.db")
LOG_PATH = os.path.join(BASE_DIR, "app.log")
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "replace_with_production_secret")
MAX_IMAGE_COUNT = 17
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH")
# If ADMIN_PASSWORD_HASH not set, we generate the hash of 'admin'
if not ADMIN_PASSWORD_HASH:
    ADMIN_PASSWORD_HASH = hashlib.sha256(b"admin").hexdigest()

# Create uploads folder if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================
# Logging
# =============================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(APP_NAME)

# =============================================
# Flask App Factory
# =============================================

# === Added init_db to fix missing reference ===

def init_db(db_path='building_survey_.db'):
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS survey_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude TEXT,
        use_type TEXT,
        num_users TEXT,
        importance_category TEXT,
        danger_falling TEXT,
        num_floors TEXT,
        structure_condition TEXT,
        year_construction TEXT,
        vertical_damage TEXT,
        danger_impact TEXT,
        soft_floor TEXT,
        short_column TEXT,
        timeinserted TEXT,
        reviewed INTEGER,
        basements TEXT,
        crack_width TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS review_data (
        id INTEGER PRIMARY KEY,
        latitude TEXT,
        use_type TEXT,
        num_users TEXT,
        importance_category TEXT,
        danger_falling TEXT,
        num_floors TEXT,
        structure_condition TEXT,
        year_construction TEXT,
        vertical_damage TEXT,
        danger_impact TEXT,
        soft_floor TEXT,
        short_column TEXT,
        timeinserted TEXT,
        reviewed INTEGER,
        basements TEXT,
        crack_width TEXT,
        type_of_structural TEXT,
        arrangement TEXT,
        irregular_vertical TEXT,
        irregular_horizontal TEXT,
        torsion TEXT,
        vulnerabilities TEXT,
        heavy_finishes TEXT,
        quality_of_user TEXT,
        soil_class TEXT,
        load_bearing TEXT,
        total_area TEXT,
        retrofitting TEXT,
        timereviewed TEXT
    );
    """)

    conn.commit()
    conn.close()


# === Added missing register_routes function ===

def register_routes(app):
    """Register all Flask routes. This stub prevents NameError until full routing is wired."""
    # NOTE: In the professional refactor, routes were expected to be organized in blueprints.
    # For now, this placeholder ensures the app starts without errors.
    # You should later fill this with the actual route registrations or blueprint imports.
    pass

def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        UPLOAD_FOLDER=UPLOAD_FOLDER,
        DB_PATH=DB_PATH,
        MAX_IMAGE_COUNT=MAX_IMAGE_COUNT,
        ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS,
    )

    # Allow optional overrides
    if config:
        app.config.update(config)

    # Initialize DB
    with app.app_context():
        init_db(app.config["DB_PATH"])  # idempotent

    # Register blueprints or routes
    register_routes(app)

    return app


app = create_app()

# =============================================
# Database utilities
# =============================================

_SCHEMA_SQL = """
-- survey_data: original submissions
CREATE TABLE IF NOT EXISTS survey_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude TEXT,
    datetime_submitted TEXT,
    use_type TEXT,
    num_users INTEGER,
    importance_category TEXT,
    danger_falling TEXT,
    num_floors INTEGER,
    structure_condition TEXT,
    year_construction TEXT,
    vertical_damage TEXT,
    danger_impact TEXT,
    soft_floor TEXT,
    short_column TEXT,
    extra_json TEXT,
    timeinserted TEXT,
    reviewed INTEGER DEFAULT 0,
    basements TEXT,
    crack_width TEXT
);

-- review_data: reviewer assessments
CREATE TABLE IF NOT EXISTS review_data (
    id INTEGER,
    latitude TEXT,
    datetime_submitted TEXT,
    use_type TEXT,
    num_users INTEGER,
    importance_category TEXT,
    danger_falling TEXT,
    num_floors INTEGER,
    structure_condition TEXT,
    year_construction TEXT,
    vertical_damage TEXT,
    danger_impact TEXT,
    soft_floor TEXT,
    short_column TEXT,
    timeinserted TEXT,
    reviewed INTEGER DEFAULT 1,
    basements TEXT,
    crack_width TEXT,
    type_of_structural TEXT,
    arrangement TEXT,
    irregular_vertical TEXT,
    irregular_horizontal TEXT,
    torsion TEXT,
    vulnerabilities TEXT,
    heavy_finishes TEXT,
    quality_of_user TEXT,
    soil_class TEXT,
    load_bearing TEXT,
    total_area TEXT,
    retrofitting TEXT,
    timereviewed TEXT
);
"""


def init_db(db_path: str) -> None:
    """Create or migrate the database schema."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.executescript(_SCHEMA_SQL)
        conn.commit()
        logger.info("Initialized DB at %s", db_path)
    except Exception:
        logger.exception("Failed to init DB")
    finally:
        conn.close()


@contextmanager
def get_db(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a database connection as a context manager.

    The connection uses row_factory to return tuples - keep compatibility with original code.
    """
    path = db_path or app.config["DB_PATH"]
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


# =============================================
# Data classes / models (simple dataclasses for convenience)
# =============================================

@dataclass
class SurveyRecord:
    id: Optional[int]
    latitude: str
    datetime_submitted: str
    use_type: str
    num_users: int
    importance_category: str
    danger_falling: str
    num_floors: int
    structure_condition: str
    year_construction: str
    vertical_damage: str
    danger_impact: str
    soft_floor: str
    short_column: str
    extra_json: Optional[str]
    timeinserted: str
    reviewed: int = 0
    basements: Optional[str] = None
    crack_width: Optional[str] = None


# =============================================
# Helpers: Validation, file utils, images
# =============================================


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def secure_filename_simple(filename: str) -> str:
    """A tiny secure filename implementation to avoid bringing werkzeug dependency.

    This function strips path separators and allows alphanumerics, dash, underscore, dot.
    """
    keepchars = (" ", ".", "_", "-")
    filename = os.path.basename(filename)
    return "".join(c for c in filename if c.isalnum() or c in keepchars).rstrip()


class ImageService:
    """Utility class for image processing.

    Contains convenience methods for opening, resizing, reorienting, and saving images.
    If PIL isn't installed, the service will raise informative errors.
    """

    @staticmethod
    def ensure_pil():
        if Image is None:
            raise RuntimeError("Pillow library is required for image processing. Install pillow via pip.")

    @staticmethod
    def resize(img_stream: io.BytesIO, scale: float = 0.3) -> Image.Image:
        ImageService.ensure_pil()
        img = Image.open(img_stream)
        w, h = img.size
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = ImageOps.exif_transpose(img)
        return img.resize(new_size, resample=Image.LANCZOS)

    @staticmethod
    def create_thumbnail(img_stream: io.BytesIO, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        ImageService.ensure_pil()
        img = Image.open(img_stream)
        img = ImageOps.exif_transpose(img)
        img.thumbnail(size)
        return img

    @staticmethod
    def save_image(img: Image.Image, path: str, quality: int = 80) -> None:
        ImageService.ensure_pil()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path, quality=quality)


# =============================================
# Simple authentication decorators
# =============================================


def admin_required(view_func):
    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login", next=request.path))
        return view_func(*args, **kwargs)

    return wrapper


def check_admin_credentials(username: str, password: str) -> bool:
    """Check admin credentials using a simple hash check; replace with user DB in prod."""
    if username != ADMIN_USERNAME:
        return False
    hash_val = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hash_val == ADMIN_PASSWORD_HASH


# =============================================
# Business logic: services for survey storage
# =============================================


def insert_survey(record: SurveyRecord, images: List[Tuple[str, io.BytesIO]]) -> int:
    """Insert a survey record and save images to disk.

    - record: SurveyRecord (id is ignored)
    - images: list of tuples (filename, BytesIO)
    Returns the new record id.
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO survey_data (latitude, datetime_submitted, use_type, num_users, importance_category, danger_falling, num_floors, structure_condition, year_construction, vertical_damage, danger_impact, soft_floor, short_column, extra_json, timeinserted, reviewed, basements, crack_width) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.latitude,
                record.datetime_submitted,
                record.use_type,
                record.num_users,
                record.importance_category,
                record.danger_falling,
                record.num_floors,
                record.structure_condition,
                record.year_construction,
                record.vertical_damage,
                record.danger_impact,
                record.soft_floor,
                record.short_column,
                record.extra_json,
                record.timeinserted,
                record.reviewed,
                record.basements,
                record.crack_width,
            ),
        )
        conn.commit()
        record_id = cur.lastrowid

    # Save images
    if images:
        record_path = os.path.join(app.config["UPLOAD_FOLDER"], str(record_id))
        os.makedirs(record_path, exist_ok=True)
        for idx, (filename, stream) in enumerate(images, start=1):
            ext = os.path.splitext(filename)[1] or ".jpg"
            fname = f"image_{idx}{ext}"
            path = os.path.join(record_path, secure_filename_simple(fname))
            try:
                img = ImageService.resize(stream, scale=0.3) if Image is not None else None
                if img is not None:
                    ImageService.save_image(img, path)
                else:
                    # raw write
                    with open(path, "wb") as fh:
                        fh.write(stream.getbuffer())
            except Exception:
                logger.exception("Failed to save image for record %s", record_id)

    return record_id


def fetch_survey(record_id: int) -> Optional[SurveyRecord]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM survey_data WHERE id = ?", (record_id,))
        row = cur.fetchone()
        if not row:
            return None
        # Map tuple to SurveyRecord - rely on schema ordering
        return SurveyRecord(
            id=row[0],
            latitude=row[1],
            datetime_submitted=row[2] if len(row) > 2 else "",
            use_type=row[3] if len(row) > 3 else "",
            num_users=row[4] if len(row) > 4 else 0,
            importance_category=row[5] if len(row) > 5 else "",
            danger_falling=row[6] if len(row) > 6 else "",
            num_floors=row[7] if len(row) > 7 else 0,
            structure_condition=row[8] if len(row) > 8 else "",
            year_construction=row[9] if len(row) > 9 else "",
            vertical_damage=row[10] if len(row) > 10 else "",
            danger_impact=row[11] if len(row) > 11 else "",
            soft_floor=row[12] if len(row) > 12 else "",
            short_column=row[13] if len(row) > 13 else "",
            extra_json=row[14] if len(row) > 14 else None,
            timeinserted=row[15] if len(row) > 15 else "",
            reviewed=row[16] if len(row) > 16 else 0,
            basements=row[17] if len(row) > 17 else None,
            crack_width=row[18] if len(row) > 18 else None,
        )


def list_surveys(reviewed: Optional[int] = None, limit: int = 100, offset: int = 0) -> List[Tuple]:
    with get_db() as conn:
        cur = conn.cursor()
        if reviewed is None:
            cur.execute("SELECT * FROM survey_data ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        else:
            cur.execute("SELECT * FROM survey_data WHERE reviewed = ? ORDER BY id DESC LIMIT ? OFFSET ?", (reviewed, limit, offset))
        rows = cur.fetchall()
        return rows


# =============================================
# Routes
# =============================================


def register_routes(app: Flask) -> None:
    # Homepage + form route
    @app.route("/", methods=["GET", "POST"])
    def index():
        try:
            if request.method == "POST":
                return handle_submission()

            # generate captcha
            num1 = random_int(1, 9)
            num2 = random_int(1, 9)
            session["captcha_result"] = num1 + num2
            captcha_question = f"{num1} + {num2} = ?"
            return render_template("index.html", captcha_question=captcha_question)
        except Exception as e:
            logger.exception("Unhandled error on index")
            return render_template("error.html", message=str(e)), 500

    # Admin login
    @app.route("/admin", methods=["GET", "POST"])
    def admin_login():
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            if check_admin_credentials(username, password):
                session["admin_logged_in"] = True
                session["admin_user"] = username
                return redirect(url_for("admin_dashboard"))
            else:
                flash("Invalid credentials", "error")
                return render_template("admin.html"), 401
        return render_template("admin.html")

    @app.route("/admin/logout")
    def admin_logout():
        session.pop("admin_logged_in", None)
        session.pop("admin_user", None)
        return redirect(url_for("admin_login"))

    @app.route("/admin/dashboard")
    @admin_required
    def admin_dashboard():
        items = list_surveys(reviewed=0, limit=500)
        return render_template("admin_dashboard.html", items=items)

    @app.route("/review/<int:item_id>")
    @admin_required
    def review(item_id: int):
        row = fetch_survey(item_id)
        if not row:
            return render_template("not_found.html", message="Record not found"), 404
        # parse location
        lat, lon = parse_latlon(row.latitude)
        return render_template("review.html", item=row, lat=lat, lon=lon)

    @app.route("/reviewed/<int:item_id>", methods=["GET", "POST"])
    @admin_required
    def reviewed(item_id: int):
        # This endpoint both shows reviewed data and performs calculations similar to original
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM review_data WHERE id = ?", (item_id,))
            row = cur.fetchone()
            if not row:
                return render_template("not_found.html", message="Review record not found"), 404
            # build a dict to pass to template
            review_data = dict_from_row(row)

        # if calculate action posted
        if request.method == "POST":
            action = request.form.get("action")
            if action == "calculate":
                calculation_result = perform_structural_calculation(review_data)
                return render_template("calculation_result.html", result=calculation_result, review=review_data)
        return render_template("reviewed.html", review=review_data)

    @app.route("/api/surveys", methods=["GET"])
    def api_list_surveys():
        # optional query params: reviewed, limit, offset
        reviewed = request.args.get("reviewed")
        reviewed_val = int(reviewed) if reviewed in ("0", "1") else None
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        rows = list_surveys(reviewed=reviewed_val, limit=limit, offset=offset)
        return jsonify([dict_from_row(r) for r in rows])

    @app.route("/export/csv")
    @admin_required
    def export_csv():
        rows = list_surveys(limit=10000)
        si = io.StringIO()
        cw = csv.writer(si)
        # header
        cw.writerow([c[0] for c in enumerate(rows[0])]) if rows else None
        # write rows
        for r in rows:
            cw.writerow(list(r))
        si.seek(0)
        return Response(si.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=surveys.csv"})

    @app.route("/images/<int:record_id>/zip")
    @admin_required
    def download_images_zip(record_id: int):
        record_path = os.path.join(app.config["UPLOAD_FOLDER"], str(record_id))
        if not os.path.exists(record_path):
            return render_template("not_found.html", message="Images not found"), 404
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_STORED) as zf:
            for fname in sorted(os.listdir(record_path)):
                zf.write(os.path.join(record_path, fname), arcname=fname)
        mem.seek(0)
        return send_file(mem, download_name=f"images_{record_id}.zip", as_attachment=True)

    # Error handlers
    @app.errorhandler(404)
    def not_found(err):
        return render_template("not_found.html", message=str(err)), 404

    @app.errorhandler(500)
    def internal_error(err):
        logger.exception("Internal server error: %s", err)
        return render_template("error.html", message=str(err)), 500


# =============================================
# Helper functions used by routes
# =============================================

import random as _rnd


def random_int(a: int, b: int) -> int:
    # deterministic enough for captcha; uses secrets if higher security required
    return _rnd.randint(a, b)


def parse_latlon(latlon: str) -> Tuple[Optional[str], Optional[str]]:
    if not latlon:
        return None, None
    try:
        parts = [p.strip() for p in latlon.split(",")]
        return parts[0], parts[1] if len(parts) > 1 else None
    except Exception:
        return None, None


def dict_from_row(row: Tuple) -> Dict[str, Any]:
    # Very conservative mapping - rely on original schema positions
    keys = [
        "id",
        "latitude",
        "datetime_submitted",
        "use_type",
        "num_users",
        "importance_category",
        "danger_falling",
        "num_floors",
        "structure_condition",
        "year_construction",
        "vertical_damage",
        "danger_impact",
        "soft_floor",
        "short_column",
        "extra_json",
        "timeinserted",
        "reviewed",
        "basements",
        "crack_width",
    ]
    return {k: row[idx] if idx < len(row) else None for idx, k in enumerate(keys)}


def perform_structural_calculation(review_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform the same calculations present in the user's code with clearer structure.

    Returns a dictionary with the intermediate values and final suggestion.
    """
    # Safe parsing + defaults
    importance_category = review_data.get("importance_category", "Σ1")
    type_of_struct = review_data.get("type_of_structural", "RC-frames")
    basements = review_data.get("basements", "0")
    soil_class = review_data.get("soil_class", "A")

    gamma_1_map = {"Σ1": 0.85, "Σ2": 1.00, "Σ3": 1.15, "Σ4": 1.30}
    gamma_1 = gamma_1_map.get(importance_category, 1.0)

    g = 9.81
    a = 0.12
    A = a * g
    beta_0 = 2.5
    eta = 1.00

    q_map = {"RC-frames": 3.50, "RC-walls": 3.00, "Brick walls": 1.50, "Inverse pentulum": 2.00}
    q = q_map.get(type_of_struct, 3.0)

    theta_map = {"0": 1.00, "1": 0.90, "At Least 2": 0.80}
    theta = theta_map.get(basements, 1.0)

    epsilon_map = {"A": 0.04, "B": 0.06, "C": 0.08}
    epsilon = epsilon_map.get(soil_class, 0.04)

    ag = 1.75 * epsilon * g
    agref = gamma_1 * A * (eta * theta * beta_0) / q
    ag_to_agref = ag / agref if agref != 0 else float("inf")

    reference_values = [1.8, 1.3, 1.0, 0.75, 0.60, 0.45, 0.35, 0.25, 0.24]
    closest_value = min(reference_values, key=lambda x: abs(x - ag_to_agref))

    tolerable = 0.75
    status = "OK" if ag_to_agref >= tolerable else "NOT TOLERABLE"

    return {
        "gamma_1": gamma_1,
        "g": g,
        "a": a,
        "A": A,
        "beta_0": beta_0,
        "eta": eta,
        "q": q,
        "theta": theta,
        "epsilon": epsilon,
        "ag": ag,
        "agref": agref,
        "ag_to_agref": ag_to_agref,
        "closest_ref": closest_value,
        "tolerable": tolerable,
        "status": status,
    }


# =============================================
# Original-style submission handler adapted
# =============================================

from werkzeug.datastructures import FileStorage


@app.route("/submit", methods=["POST"])
def handle_submission():
    """Process the original web form submission, validate data, store to DB and images."""
    # CAPTCHA
    user_answer = request.form.get("captcha_answer")
    correct_answer = session.get("captcha_result")
    try:
        if correct_answer is None or int(user_answer or -1) != int(correct_answer):
            flash("Incorrect captcha. Please try again.", "error")
            return redirect(url_for("index"))
    except Exception:
        flash("Captcha validation failed.", "error")
        return redirect(url_for("index"))

    # Required fields and validation
    location = request.form.get("location")
    if not location:
        flash("Location required", "error")
        return redirect(url_for("index"))

    # Build SurveyRecord
    now = datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")
    record = SurveyRecord(
        id=None,
        latitude=location,
        datetime_submitted=now,
        use_type=request.form.get("type_of_use", ""),
        num_users=int_or_zero(request.form.get("number_of_users")),
        importance_category=request.form.get("importance_category", ""),
        danger_falling=request.form.get("danger_of", ""),
        num_floors=int_or_zero(request.form.get("number_of_floors")),
        structure_condition=request.form.get("condition", ""),
        year_construction=request.form.get("year", ""),
        vertical_damage=request.form.get("previous_damages", ""),
        danger_impact=request.form.get("danger_neighbour", ""),
        soft_floor=request.form.get("soft_floor", ""),
        short_column=request.form.get("short_column", ""),
        extra_json=json.dumps({"submitted_from": request.remote_addr}),
        timeinserted=now,
        reviewed=0,
        basements=request.form.get("basements", ""),
        crack_width=request.form.get("crack_width", ""),
    )

    # Collect images
    images: List[Tuple[str, io.BytesIO]] = []
    for i in range(1, app.config["MAX_IMAGE_COUNT"] + 1):
        key = f"image{i}"
        f: Optional[FileStorage] = request.files.get(key)
        if f and f.filename and allowed_file(f.filename):
            data = io.BytesIO(f.read())
            images.append((f.filename, data))

    # Insert
    try:
        new_id = insert_survey(record, images)
        logger.info("Inserted record %s with %d images", new_id, len(images))
        flash("Survey submitted successfully", "success")
        return redirect(url_for("form_submission"))
    except Exception:
        logger.exception("Failed to insert survey")
        flash("Failed to submit survey", "error")
        return redirect(url_for("index"))


# =============================================
# Utility small helpers
# =============================================


def int_or_zero(val: Any) -> int:
    try:
        return int(val)
    except Exception:
        return 0


# =============================================
# CLI helpers for dev convenience
# =============================================


def seed_sample_data(n: int = 5) -> None:
    """Insert some sample records for demo/testing."""
    for i in range(n):
        r = SurveyRecord(
            id=None,
            latitude=f"37.{1000+i},23.{500+i}",
            datetime_submitted=datetime.utcnow().isoformat(),
            use_type="Residential",
            num_users=10 + i,
            importance_category="Σ1",
            danger_falling="Low",
            num_floors=2 + i,
            structure_condition="Fair",
            year_construction=str(1990 + i),
            vertical_damage="None",
            danger_impact="None",
            soft_floor="No",
            short_column="No",
            extra_json=json.dumps({"seed": True}),
            timeinserted=datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S"),
            reviewed=0,
            basements="0",
            crack_width="0mm",
        )
        insert_survey(r, [])
    logger.info("Seeded %s sample records", n)



# =============================================
# Main
# =============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{APP_NAME} - Flask server")
    parser.add_argument("--init-db", action="store_true", help="Initialize the sqlite database")
    parser.add_argument("--seed", type=int, default=0, help="Seed sample data")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.init_db:
        init_db(DB_PATH)
        print("DB initialized")
        sys.exit(0)

    if args.seed > 0:
        init_db(DB_PATH)
        seed_sample_data(args.seed)
        print("Seeded data")
        sys.exit(0)

    # Create app
    app = create_app()

    # Run app normally
    app.run(host=args.host, port=args.port, debug=args.debug)