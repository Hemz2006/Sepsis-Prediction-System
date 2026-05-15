# Sepsis Prediction Simulator

This is a web-based simulator for predicting sepsis risk in patients using a deep learning model (CNN-LSTM) built with PyTorch. The app allows doctors to add patients, record vital signs and lab values, and receive real-time sepsis risk predictions.

## Features
- User authentication (doctor login/register)
- Add new patients and readings
- Sepsis risk prediction using a trained AI model
- Fallback rule-based prediction if the model is unavailable
- Dashboard and patient detail views

## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure model and scaler files are present:**
   - `model/cnn_lstm_model.pth` (PyTorch model)
   - `model/scaler.pkl` (feature scaler)
4. **Database:**
   - The app uses SQLite by default (`instance/sepsis_prediction.db`).
   - The database will be created automatically on first run.

## Running the App
```bash
python app.py
```
- The app will be available at `http://127.0.0.1:5000/`

## Notes
- If the AI model or scaler cannot be loaded, the app will use a rule-based fallback for predictions. A warning will be shown in the patient detail view.
- Only share the `model/` files if you want to include the trained model. The database file (`instance/sepsis_prediction.db`) can be reset or omitted for a fresh start.

## Directory Structure
- `app.py` - Main Flask application
- `config.py` - Configuration settings
- `model/` - Contains the trained model and scaler
- `templates/` - HTML templates for the web UI
- `instance/` - Contains the SQLite database

## License
This project is for educational and demonstration purposes.