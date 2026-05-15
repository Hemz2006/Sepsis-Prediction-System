import pytest
from app import app, db, predict_sepsis
import csv
import os

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client


def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Sepsis' in response.data or b'Login' in response.data


def test_login_page(client):
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Login' in response.data


def test_register_page(client):
    response = client.get('/register')
    assert response.status_code == 200
    assert b'Register' in response.data


def test_model_prediction_from_csv():
    # Read the first data row from 75data.csv (skip header)
    csv_path = '75data.csv'
    assert os.path.exists(csv_path), '75data.csv not found.'
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Map CSV columns to model input keys (matching app.py)
            data = {
                'hr': float(row['HR']),
                'o2sat': float(row['O2Sat']),
                'temp': float(row['Temp']),
                'sbp': float(row['SBP']),
                'map': float(row['MAP']),
                'dbp': float(row['DBP']),
                'resp': float(row['Resp']),
                'etco2': float(row['EtCO2']),
                'hco3': float(row['HCO3']),
                'fio2': float(row['FiO2']),
                'ph': float(row['pH']),
                'paco2': float(row['PaCO2']),
                'sao2': float(row['SaO2']),
                'bun': float(row['BUN']),
                'calcium': float(row['Calcium']),
                'chloride': float(row['Chloride']),
                'creatinine': float(row['Creatinine']),
                'bilirubin_direct': float(row['Bilirubin_direct']),
                'glucose': float(row['Glucose']),
                'lactate': float(row['Lactate']),
                'magnesium': float(row['Magnesium']),
                'phosphate': float(row['Phosphate']),
                'potassium': float(row['Potassium']),
                'troponini': float(row['TroponinI']),
                'hct': float(row['Hct']),
                'hgb': float(row['Hgb']),
                'ptt': float(row['PTT']),
                'wbc': float(row['WBC']),
                'fibrinogen': float(row['Fibrinogen']),
                'platelets': float(row['Platelets']),
                'age': int(float(row['Age'])),
                'gender': int(float(row['Gender'])),
            }
            prediction, confidence, source = predict_sepsis(data)
            print(f"Prediction: {prediction}, Confidence: {confidence:.2f}%, Source: {source}")
            # Assert that the source is either 'model' or 'fallback'
            assert source in ('model', 'fallback')
            break  # Only test the first row