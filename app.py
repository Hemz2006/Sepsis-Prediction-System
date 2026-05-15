import os
import numpy as np
import pandas as pd
import logging
from functools import wraps
from datetime import datetime, timedelta, timezone
from benchmarks import BENCHMARKS
from datetime import timezone
from zoneinfo import ZoneInfo
import joblib
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import tensorflow as tf



MEDICAL_THRESHOLDS = {
    # =====================
    # Vital Signs
    # =====================
    "HR": {"high": 100, "label": "Tachycardia"},
    "O2Sat": {"low": 95, "label": "Hypoxemia"},
    "Temp": {"high": 38.0, "label": "Fever"},
    "SBP": {"low": 90, "label": "Hypotension"},
    "MAP": {"low": 65, "label": "Low MAP (Shock)"},
    "DBP": {"low": 60, "label": "Low Diastolic BP"},
    "Resp": {"high": 22, "label": "Tachypnea"},
    "EtCO2": {"low": 35, "label": "Low Cardiac Output"},
    "HCO3": {"low": 22, "label": "Metabolic Acidosis"},
    "pH": {"low": 7.35, "label": "Acidemia"},
    "PaCO2": {"low": 35, "label": "Respiratory Compensation"},
    "SaO2": {"low": 95, "label": "Low Arterial Oxygen"},

    # =====================
    # Laboratory Values
    # =====================
    "Lactate": {"high": 2.0, "label": "Elevated Lactate (Tissue Hypoxia)"},
    "Creatinine": {"high": 1.3, "label": "Acute Kidney Injury"},
    "BUN": {"high": 20, "label": "Renal Hypoperfusion"},
    "WBC": {"high": 11, "label": "Infection / Systemic Inflammation"},
    "Platelets": {"low": 150, "label": "Thrombocytopenia"},
    "Hgb": {"low": 12, "label": "Reduced Oxygen Carrying Capacity"},
    "Hct": {"low": 36, "label": "Tissue Hypoxia"},
    "PTT": {"high": 35, "label": "Coagulation Dysfunction"},
    "Bilirubin_direct": {"high": 0.3, "label": "Liver Dysfunction"},
    "TroponinI": {"high": 0.04, "label": "Cardiac Injury"},
    "Glucose": {"high": 180, "label": "Stress Hyperglycemia"},
    "Potassium": {"high": 5.0, "label": "Electrolyte Imbalance"},
    "Calcium": {"low": 8.5, "label": "Hypocalcemia"},
    "Magnesium": {"low": 1.7, "label": "Low Magnesium"},
    "Phosphate": {"low": 2.5, "label": "Low Phosphate"},

    # =====================
    # 🔥 FIBRINOGEN (Added)
    # =====================
    "Fibrinogen": {
        "high": 450,
        "label": "Elevated Fibrinogen (Acute-Phase Inflammatory Response)"
    }
}


def analyze_sepsis_drivers(input_data):
    """
    Identifies which clinical variables exceed sepsis-related thresholds
    and returns human-readable medical explanations.
    """
    exceeded = []

    for feature, rules in MEDICAL_THRESHOLDS.items():
        value = input_data.get(feature)

        if value is None:
            continue

        if "high" in rules and value > rules["high"]:
            exceeded.append(f"{rules['label']} ({feature}: {value})")

        if "low" in rules and value < rules["low"]:
            exceeded.append(f"{rules['label']} ({feature}: {value})")

    return exceeded




from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

from clinical_scoring import SepsisScoring
from config import config


# =====================================================
# APP INITIALIZATION
# =====================================================
app = Flask(__name__)

config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sepsis_scorer = SepsisScoring()

def TCNBlock(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    prev_x = x

    x = Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu"
    )(x)
    x = BatchNormalization()(x)

    if prev_x.shape[-1] != filters:
        prev_x = Conv1D(filters, 1, padding="same")(prev_x)

    return tf.keras.layers.Add()([prev_x, x])

def build_tcn_model(sequence_len=12, num_features=32):
    inputs = Input(shape=(sequence_len, num_features))

    x = Conv1D(64, 3, padding="causal", activation="relu")(inputs)
    x = BatchNormalization()(x)

    x = TCNBlock(x, filters=64, kernel_size=3, dilation_rate=1)
    x = TCNBlock(x, filters=64, kernel_size=3, dilation_rate=2)
    x = TCNBlock(x, filters=64, kernel_size=3, dilation_rate=4)
    x = TCNBlock(x, filters=64, kernel_size=3, dilation_rate=8)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model


# =====================================================
# TCN MODEL (USED ONLY FOR AI MODEL TEST)
# =====================================================

TCN_BASE_PATH = "model/TCN"



tcn_scaler = joblib.load(os.path.join(TCN_BASE_PATH, "scaler_tcn.pkl"))
tcn_features = joblib.load(os.path.join(TCN_BASE_PATH, "feature_cols_tcn.pkl"))
tcn_seq_len = joblib.load(os.path.join(TCN_BASE_PATH, "sequence_len_tcn.pkl"))
tcn_threshold = joblib.load(os.path.join(TCN_BASE_PATH, "threshold_tcn.pkl"))

tcn_model = build_tcn_model(
    sequence_len=tcn_seq_len,
    num_features=len(tcn_features)
)

tcn_model.load_weights(
    os.path.join(TCN_BASE_PATH, "tcn_sepsis_model.h5")
)


logger.info("TCN model loaded for AI Model Test only")
logger.info("Clinical scoring system initialized successfully")


# =====================================================
# DATABASE MODELS
# =====================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    patients = db.relationship('Patient', backref='doctor', lazy=True)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class SofaTest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

    test_type = db.Column(db.String(20), nullable=False)

    timestamp = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    total_score = db.Column(db.Integer)
    sepsis_criteria_met = db.Column(db.Boolean)
    high_risk = db.Column(db.Boolean)
    septic_shock_present = db.Column(db.Boolean)
    mortality_risk = db.Column(db.String(50))
    interpretation = db.Column(db.Text)

    respiration_score = db.Column(db.Integer)
    coagulation_score = db.Column(db.Integer)
    liver_score = db.Column(db.Integer)
    cardiovascular_score = db.Column(db.Integer)
    cns_score = db.Column(db.Integer)
    renal_score = db.Column(db.Integer)
    delta_sofa = db.Column(db.Integer)
    baseline_sofa = db.Column(db.Integer)

    respiratory_rate_22_or_higher = db.Column(db.Boolean)
    altered_mentation = db.Column(db.Boolean)
    systolic_bp_100_or_lower = db.Column(db.Boolean)

    sepsis_present = db.Column(db.Boolean)
    persistent_hypotension_on_vasopressors = db.Column(db.Boolean)
    lactate_greater_than_2 = db.Column(db.Boolean)
    adequate_volume_resuscitation = db.Column(db.Boolean)

    pao2_fio2 = db.Column(db.Float)
    respiratory_support = db.Column(db.Boolean)
    respiratory_rate = db.Column(db.Float)

    map_mmhg = db.Column(db.Float)
    systolic_bp = db.Column(db.Float)
    dopamine_dose = db.Column(db.Float)
    dobutamine_dose = db.Column(db.Float)
    epinephrine_dose = db.Column(db.Float)
    norepinephrine_dose = db.Column(db.Float)

    platelets = db.Column(db.Float)
    bilirubin = db.Column(db.Float)
    creatinine = db.Column(db.Float)
    lactate = db.Column(db.Float)
    urine_output = db.Column(db.Float)

    gcs = db.Column(db.Integer)
    suspected_infection = db.Column(db.Boolean)
    on_vasopressors = db.Column(db.Boolean)


# =====================================================
# LOGIN HANDLING
# =====================================================
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# =====================================================
# ROUTES
# =====================================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    """Public Privacy Policy page"""
    return render_template('privacy_policy.html')


@app.route('/ai-model', methods=['GET', 'POST'])
@doctor_required
def ai_model():
    """
    AI Model Test
    Uses TCN ONLY
    """

    FORM_TO_TCN_FEATURES = {
        "heart_rate": "HR",
        "o2_saturation": "O2Sat",
        "temperature": "Temp",
        "systolic_bp": "SBP",
        "map": "MAP",
        "diastolic_bp": "DBP",
        "respiratory_rate": "Resp",
        "etco2": "EtCO2",
        "hco3": "HCO3",
        "fio2": "FiO2",
        "ph": "pH",
        "paco2": "PaCO2",
        "sao2": "SaO2",

        "bun": "BUN",
        "calcium": "Calcium",
        "chloride": "Chloride",
        "creatinine": "Creatinine",
        "bilirubin_direct": "Bilirubin_direct",
        "glucose": "Glucose",
        "lactate": "Lactate",
        "magnesium": "Magnesium",
        "phosphate": "Phosphate",
        "potassium": "Potassium",
        "troponini": "TroponinI",
        "hct": "Hct",
        "hemoglobin": "Hgb",
        "ptt": "PTT",
        "wbc": "WBC",
        "platelets": "Platelets",
        "fibrinogen": "Fibrinogen",

        "age": "Age",
        "gender": "Gender"
    }

    result = None
    probability = None
    sepsis_drivers = []

    if request.method == "POST":
        try:
            input_data = {}

            for form_key, tcn_key in FORM_TO_TCN_FEATURES.items():
                val = request.form.get(form_key)

                if val is None or val == "":
                    input_data[tcn_key] = 0.0
                elif tcn_key == "Gender":
                    input_data[tcn_key] = 1 if val.lower() in ["male", "m"] else 0
                else:
                    input_data[tcn_key] = float(val)

            df = pd.DataFrame([input_data])
            print("TCN FEATURES:", tcn_features)
            print("DF COLUMNS:", df.columns.tolist())
            df = df[tcn_features]  # ensure correct order

            X = tcn_scaler.transform(df)
            X_seq = np.repeat(X, tcn_seq_len, axis=0)
            X_seq = np.expand_dims(X_seq, axis=0)

            probability = float(tcn_model.predict(X_seq, verbose=0)[0][0])
            result = "Sepsis Likely" if probability >= tcn_threshold else "Sepsis Unlikely"

            # Medical threshold explanation
            sepsis_drivers = analyze_sepsis_drivers(input_data)

        except Exception as e:
            flash(f"TCN prediction error: {str(e)}", "error")

    return render_template(
        'ai_model.html',
        result=result,
        probability=probability,
        sepsis_drivers=sepsis_drivers
    )



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password_hash, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')

        if not email or not password or not name:
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))

        user = User(
            email=email,
            name=name,
            password_hash=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
@doctor_required
def dashboard():
    patients = Patient.query.filter_by(doctor_id=current_user.id).all()

    total_tests = 0
    sepsis_alerts = 0
    recent_tests = 0

    patient_data = []

    now_utc = datetime.now(timezone.utc)

    for patient in patients:
        tests = SofaTest.query.filter_by(patient_id=patient.id).all()
        total_tests += len(tests)

        # Recent activity (last 24h)
        for t in tests:
            ts = t.timestamp
            if ts and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if ts and (now_utc - ts).total_seconds() < 86400:
                recent_tests += 1

            if t.high_risk or t.septic_shock_present:
                sepsis_alerts += 1

        # Latest NEWS2 risk for dashboard badge
        latest_news2 = (
            SofaTest.query
            .filter_by(patient_id=patient.id, test_type='news2')
            .order_by(SofaTest.timestamp.desc())
            .first()
        )

        news2_risk = None
        if latest_news2:
            if latest_news2.high_risk:
                news2_risk = "High"
            else:
                news2_risk = "Low"

        patient_data.append({
            'patient': patient,
            'news2_risk': news2_risk
        })

    return render_template(
        'dashboard.html',
        patient_data=patient_data,          # ✅ REQUIRED BY TEMPLATE
        total_patients=len(patients),       # ✅ FIXES "0 patients"
        total_tests=total_tests,
        sepsis_alerts=sepsis_alerts,
        recent_tests=recent_tests
    )

@app.route('/patient/<int:patient_id>/test/new', methods=['GET', 'POST'])
@doctor_required
def new_sofa_test(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    if request.method == 'POST':
        test_type = request.form.get('test_type')

        if test_type == 'qsofa':
            return redirect(url_for('qsofa_test', patient_id=patient_id))
        elif test_type == 'sofa':
            return redirect(url_for('sofa_test', patient_id=patient_id))
        elif test_type =="septic_shock":
            return redirect(url_for('septic_shock_test', patient_id=patient_id))
        elif test_type == 'news2':
            return redirect(url_for('news2_test', patient_id=patient_id))
        elif test_type == 'ai_model':
            # Redirect to the existing AI Model page (standalone overview / model interface)
            return redirect(url_for('ai_model'))

    return render_template('new_sofa_test.html', patient=patient)


@app.route('/patient/new', methods=['GET', 'POST'])
@doctor_required
def new_patient():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')

        patient = Patient(
            doctor_id=current_user.id,
            name=name,
            age=age,
            gender=gender
        )

        db.session.add(patient)
        db.session.commit()

        flash('Patient added successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('new_patient.html')

@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@doctor_required
def delete_patient(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    # Delete all associated tests first
    SofaTest.query.filter_by(patient_id=patient.id).delete()

    # Delete patient
    db.session.delete(patient)
    db.session.commit()

    flash(f'Patient {patient.name} deleted successfully.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/patient/<int:patient_id>')
@doctor_required
def patient_detail(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    sofa_tests = (
        SofaTest.query
        .filter_by(patient_id=patient.id)
        .order_by(SofaTest.timestamp.desc())
        .all()
    )

    # =========================
    # GRAPH DATA
    # =========================
    graph_points = [
        {
            "x": t.timestamp.isoformat(),
            "y": t.total_score
        }
        for t in sofa_tests
        if t.total_score is not None
    ]

    # =========================
    # TREND CALCULATION
    # =========================
    scores = [t.total_score for t in sofa_tests if t.total_score is not None]

    if len(scores) < 2:
        trend = "Insufficient data"
    else:
        if scores[-1] > scores[0]:
            trend = "Worsening"
        elif scores[-1] < scores[0]:
            trend = "Improving"
        else:
            trend = "Stable"

    return render_template(
    "patient_detail.html",
    patient=patient,
    sofa_tests=sofa_tests,
    graph_points=graph_points,
    trend=trend,
    benchmarks=BENCHMARKS   # ✅ FIX
)

@app.route('/patient/<int:patient_id>/test/<int:test_id>/delete', methods=['POST'])
@doctor_required
def delete_sofa_test(patient_id, test_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    test = SofaTest.query.filter_by(
        id=test_id,
        patient_id=patient.id
    ).first_or_404()

    db.session.delete(test)
    db.session.commit()

    flash('SOFA test deleted successfully.', 'success')
    return redirect(url_for('patient_detail', patient_id=patient.id))

@app.route('/patient/<int:patient_id>/test/qsofa', methods=['GET', 'POST'])
@doctor_required
def qsofa_test(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    if request.method == 'POST':
        try:
            respiratory_rate = float(request.form.get('respiratory_rate'))
            systolic_bp = float(request.form.get('systolic_bp'))
            gcs = int(request.form.get('gcs'))
            suspected_infection = request.form.get('suspected_infection') == 'yes'

            result = sepsis_scorer.calculate_qsofa(
                respiratory_rate=respiratory_rate,
                systolic_bp=systolic_bp,
                gcs=gcs,
                suspected_infection=suspected_infection
            )

            test = SofaTest(
                patient_id=patient.id,
                test_type='qsofa',
                total_score=result['qsofa_score'],
                high_risk=result['high_risk_for_poor_outcomes'],
                interpretation=result['interpretation'],
                respiratory_rate=respiratory_rate,
                systolic_bp=systolic_bp,
                gcs=gcs,
                suspected_infection=suspected_infection,
                respiratory_rate_22_or_higher=result['criteria_met']['respiratory_rate_22_or_higher'],
                altered_mentation=result['criteria_met']['altered_mentation'],
                systolic_bp_100_or_lower=result['criteria_met']['systolic_bp_100_or_lower']
            )

            db.session.add(test)
            db.session.commit()

            flash('qSOFA test added successfully.', 'success')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        except Exception as e:
            flash(f'Error calculating qSOFA: {str(e)}', 'error')

    return render_template('qsofa_test_form.html', patient=patient)

@app.route('/patient/<int:patient_id>/test/news2', methods=['GET', 'POST'])
@doctor_required
def news2_test(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    if request.method == 'POST':
        try:
            respiratory_rate = int(request.form['respiratory_rate'])
            spo2_scale_1 = int(request.form['SpO2_Scale_1'])
            spo2_scale_2 = int(request.form['SpO2_Scale_2'])
            supplemental_oxygen = request.form.get('supplemental_oxygen') == 'on'
            systolic_bp = int(request.form['systolic_bp'])
            heart_rate = int(request.form['heart_rate'])
            level_of_consciousness = request.form['level_of_consciousness']
            temperature = float(request.form['temperature'])
            age = float(request.form['Age'])

            result = sepsis_scorer.calculate_news2(
                respiratory_rate,
                spo2_scale_1,
                spo2_scale_2,
                supplemental_oxygen,
                systolic_bp,
                heart_rate,
                level_of_consciousness,
                temperature,
                age
            )

            test = SofaTest(
                patient_id=patient.id,
                test_type='news2',
                total_score=result['total_score'],
                high_risk=(result.get('risk_band') == 'High'),
                interpretation=result.get('interpretation'),
                respiratory_rate=respiratory_rate,
                systolic_bp=systolic_bp
            )

            db.session.add(test)
            db.session.commit()

            flash('NEWS2 test added successfully.', 'success')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        except Exception as e:
            flash(f'Error calculating NEWS2: {str(e)}', 'error')

    return render_template('news2_test_form.html', patient=patient)


@app.route('/api/patient/<int:patient_id>/sofa-tests')
@doctor_required
def patient_sofa_tests_api(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    tests = (
        SofaTest.query
        .filter_by(patient_id=patient.id)
        .order_by(SofaTest.timestamp.asc())
        .all()
    )

    IST = ZoneInfo("Asia/Kolkata")

    timestamps = []
    for t in tests:
        ts = t.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts.astimezone(IST).isoformat())

    return jsonify({
        "timestamps": timestamps,
        "scores": [t.total_score for t in tests],
        "types": [t.test_type for t in tests]
    })


# ==============================
# CALCULATOR ROUTES (REQUIRED)
# ==============================

@app.route('/calculator/qsofa_calculator', methods=['GET', 'POST'])
@doctor_required
def qsofa_calculator():
    result = None
    if request.method == 'POST':
        try:
            respiratory_rate = float(request.form.get('respiratory_rate', 20))
            systolic_bp = float(request.form.get('systolic_bp', 120))
            gcs = int(request.form.get('gcs', 15))
            suspected_infection = request.form.get('suspected_infection') == 'yes'

            result = sepsis_scorer.calculate_qsofa(
                respiratory_rate=respiratory_rate,
                systolic_bp=systolic_bp,
                gcs=gcs,
                suspected_infection=suspected_infection
            )
        except Exception as e:
            flash(f'Error calculating qSOFA: {str(e)}', 'error')

    return render_template('qsofa_calculator.html', result=result)


@app.route('/calculator/sofa_calculator', methods=['GET', 'POST'])
@doctor_required
def sofa_calculator():
    return render_template('sofa_calculator.html')


@app.route('/calculator/septic_shock_calculator', methods=['GET', 'POST'])
@doctor_required
def septic_shock_calculator():
    return render_template('septic_shock_calculator.html')


@app.route('/calculator/news2_calculator', methods=['GET', 'POST'])
@doctor_required
def news2_calculator():
    return render_template('news2_calculator.html')

@app.route('/patient/<int:patient_id>/test/sofa', methods=['GET', 'POST'])
@doctor_required
def sofa_test(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    if request.method == 'POST':
        try:
            pao2_fio2 = float(request.form.get('pao2_fio2', 300))
            platelets = float(request.form.get('platelets', 150))
            bilirubin = float(request.form.get('bilirubin', 1.0))
            map_mmhg = float(request.form.get('map_mmhg', 70))
            gcs = int(request.form.get('gcs', 15))
            creatinine = float(request.form.get('creatinine', 1.0))
            respiratory_support = request.form.get('respiratory_support') == 'yes'
            baseline_sofa = int(request.form.get('baseline_sofa', 0))

            dopamine_dose = float(request.form.get('dopamine_dose', 0))
            dobutamine_dose = float(request.form.get('dobutamine_dose', 0))
            epinephrine_dose = float(request.form.get('epinephrine_dose', 0))
            norepinephrine_dose = float(request.form.get('norepinephrine_dose', 0))
            urine_output = float(request.form.get('urine_output', 0)) if request.form.get('urine_output') else None

            result = sepsis_scorer.calculate_total_sofa(
                pao2_fio2=pao2_fio2,
                platelets=platelets,
                bilirubin=bilirubin,
                map_mmhg=map_mmhg,
                gcs=gcs,
                creatinine=creatinine,
                respiratory_support=respiratory_support,
                dopamine_dose=dopamine_dose,
                dobutamine_dose=dobutamine_dose,
                epinephrine_dose=epinephrine_dose,
                norepinephrine_dose=norepinephrine_dose,
                urine_output_ml_day=urine_output,
                baseline_sofa=baseline_sofa
            )

            test = SofaTest(
                patient_id=patient.id,
                test_type='sofa',
                total_score=result['total_sofa'],
                sepsis_criteria_met=result['sepsis_criteria_met'],
                mortality_risk=f"{result['estimated_mortality_risk_percent']}%",
                interpretation=result['interpretation'],
                respiration_score=result['individual_scores']['respiration'],
                coagulation_score=result['individual_scores']['coagulation'],
                liver_score=result['individual_scores']['liver'],
                cardiovascular_score=result['individual_scores']['cardiovascular'],
                cns_score=result['individual_scores']['cns'],
                renal_score=result['individual_scores']['renal'],
                delta_sofa=result['delta_sofa'],
                baseline_sofa=baseline_sofa,
                pao2_fio2=pao2_fio2,
                respiratory_support=respiratory_support,
                map_mmhg=map_mmhg,
                gcs=gcs,
                platelets=platelets,
                bilirubin=bilirubin,
                creatinine=creatinine,
                dopamine_dose=dopamine_dose,
                dobutamine_dose=dobutamine_dose,
                epinephrine_dose=epinephrine_dose,
                norepinephrine_dose=norepinephrine_dose,
                urine_output=urine_output
            )

            db.session.add(test)
            db.session.commit()

            flash('SOFA test added successfully.', 'success')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        except Exception as e:
            flash(f'Error calculating SOFA: {str(e)}', 'error')

    return render_template('sofa_test_form.html', patient=patient)

@app.route('/patient/<int:patient_id>/test/septic-shock', methods=['GET', 'POST'])
@doctor_required
def septic_shock_test(patient_id):
    patient = Patient.query.filter_by(
        id=patient_id,
        doctor_id=current_user.id
    ).first_or_404()

    if request.method == 'POST':
        try:
            map_mmhg = float(request.form.get('map_mmhg'))
            lactate = float(request.form.get('lactate'))
            vasopressors = request.form.get('vasopressors') == 'yes'

            result = sepsis_scorer.calculate_septic_shock(
                map_mmhg=map_mmhg,
                lactate=lactate,
                vasopressors=vasopressors
            )

            test = SofaTest(
                patient_id=patient.id,
                test_type="septic_shock",          # ✅ REQUIRED
                total_score=0,
                high_risk=result['septic_shock_present'],
                septic_shock_present=result['septic_shock_present'],
                interpretation=result['interpretation']
)

            db.session.add(test)
            db.session.commit()


            flash('Septic shock assessment saved.', 'success')
            return redirect(url_for('patient_detail', patient_id=patient.id))

        except Exception as e:
            flash(f'Error calculating septic shock: {str(e)}', 'error')

    return render_template('septic_shock_test_form.html', patient=patient)


# =====================================================
# DB INIT
# =====================================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )
