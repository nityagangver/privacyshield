import os
import threading
from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Input schema
class PredictRequest(BaseModel):
    tenure: float = Field(..., description="Tenure in months")
    MonthlyCharges: float = Field(..., description="Monthly charges")
    TotalCharges: float = Field(..., description="Total charges")
    Contract: int = Field(..., description="Contract type (0: Month-to-Month, 1: One Year, 2: Two Year)")
    InternetService: int = Field(..., description="Internet service type (0: DSL, 1: Fiber, 2: No)")
    PaymentMethod: int = Field(..., description="Payment method (0: Electronic Check, 1: Mailed Check, 2: Bank Transfer, 3: Credit Card)")
    SeniorCitizen: int = Field(..., description="Senior citizen indicator (0 or 1)")
    Partner: int = Field(..., description="Partner indicator (0 or 1)")
    Dependents: int = Field(..., description="Dependents indicator (0 or 1)")
    PhoneService: int = Field(..., description="Phone service indicator (0 or 1)")
    PaperlessBilling: int = Field(..., description="Paperless billing indicator (0 or 1)")

class CustomPredictRequest(PredictRequest):
    epsilon: float = Field(1.0, description="Epsilon privacy level (0.1, 0.5, 1.0, 5.0, or 10.0)")

class AttackRequest(BaseModel):
    model_type: str = Field("baseline", description="Model type: baseline or private")
    n_samples: int = Field(100, description="Number of samples to draw from train and test")
    epsilon: float = Field(1.0, description="Epsilon privacy budget if model_type is private")

def run_training_in_background(app_state):
    try:
        app_state.training_progress = "Creating directories..."
        os.makedirs("models", exist_ok=True)
        os.makedirs("graphs", exist_ok=True)
        
        required_files = [
            "models/scaler.pkl",
            "models/baseline_model.pkl",
            "models/dp_epsilon_01_model.pkl",
            "models/dp_epsilon_05_model.pkl",
            "models/dp_epsilon_1_model.pkl",
            "models/dp_epsilon_5_model.pkl",
            "models/dp_epsilon_10_model.pkl",
            "models/X_train_sample.npy",
            "models/X_test_sample.npy"
        ]
        
        all_exist = all(os.path.exists(f) for f in required_files)
        if not all_exist:
            app_state.training_progress = "Running main.py training script (5-10 minutes)..."
            import subprocess
            subprocess.run(["python", "main.py"], check=True)
        else:
            app_state.training_progress = "Loading pre-existing models..."
            
        app_state.training_progress = "Loading trained pkl files..."
        app_state.scaler = joblib.load("models/scaler.pkl")
        app_state.baseline = joblib.load("models/baseline_model.pkl")
        app_state.dp_models = {
            0.1: joblib.load("models/dp_epsilon_01_model.pkl"),
            0.5: joblib.load("models/dp_epsilon_05_model.pkl"),
            1.0: joblib.load("models/dp_epsilon_1_model.pkl"),
            5.0: joblib.load("models/dp_epsilon_5_model.pkl"),
            10.0: joblib.load("models/dp_epsilon_10_model.pkl"),
        }
        app_state.dp_model = app_state.dp_models[1.0]
        app_state.X_train_sample = np.load("models/X_train_sample.npy")
        app_state.X_test_sample = np.load("models/X_test_sample.npy")
        app_state.models_ready = True
        app_state.model_loaded = True
        app_state.training_progress = "Ready"
        print("Background training and loading finished successfully.")
    except Exception as e:
        app_state.training_progress = f"Training failed: {str(e)}"
        print(f"Error during background training/loading: {e}")
        app_state.models_ready = False
        app_state.model_loaded = False

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.models_ready = False
    app.state.model_loaded = False
    app.state.scaler = None
    app.state.baseline = None
    app.state.dp_models = {}
    app.state.dp_model = None
    app.state.X_train_sample = None
    app.state.X_test_sample = None
    app.state.training_progress = "Starting background training..."
    
    thread = threading.Thread(target=run_training_in_background, args=(app.state,))
    thread.daemon = True
    thread.start()
    
    yield
    # Shutdown
    pass

app = FastAPI(
    title="PrivacyShield - Secure Churn Predictor",
    description="FastAPI server for PrivacyShield, showcasing Differential Privacy vs Baseline models",
    version="1.0.0",
    lifespan=lifespan
)

# Mount graphs folder as static files
if os.path.exists("graphs"):
    app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_prediction_vector(data: PredictRequest) -> np.ndarray:
    # Map PaymentMethod from frontend to dataset label-encoding values:
    # Frontend incoming values:
    # 0: Electronic Check, 1: Mailed Check, 2: Bank Transfer, 3: Credit Card
    # Dataset expects:
    # 0: Bank Transfer, 1: Credit Card, 2: Electronic Check, 3: Mailed Check
    pm_mapping = {
        0: 2,  # Electronic Check -> 2
        1: 3,  # Mailed Check -> 3
        2: 0,  # Bank Transfer -> 0
        3: 1   # Credit Card -> 1
    }
    mapped_payment_method = pm_mapping.get(data.PaymentMethod, data.PaymentMethod)

    # 19 features order in main.py:
    # ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    #  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    #  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    #  'MonthlyCharges', 'TotalCharges']
    vector = [
        0,  # gender (default to 0)
        data.SeniorCitizen,
        data.Partner,
        data.Dependents,
        data.tenure,
        data.PhoneService,
        0,  # MultipleLines (default to 0)
        data.InternetService,
        0,  # OnlineSecurity (default to 0)
        0,  # OnlineBackup (default to 0)
        0,  # DeviceProtection (default to 0)
        0,  # TechSupport (default to 0)
        0,  # StreamingTV (default to 0)
        0,  # StreamingMovies (default to 0)
        data.Contract,
        data.PaperlessBilling,
        mapped_payment_method,
        data.MonthlyCharges,
        data.TotalCharges
    ]
    return np.array([vector])

@app.post("/predict/standard")
def predict_standard(data: PredictRequest):
    if not getattr(app.state, "models_ready", False) or app.state.baseline is None:
        raise HTTPException(status_code=503, detail="Models are being trained. Please wait 5-10 minutes and try again. Check /health for status.")
    
    try:
        features_raw = make_prediction_vector(data)
        features_scaled = app.state.scaler.transform(features_raw)
        
        prob = float(app.state.baseline.predict_proba(features_scaled)[0][1])
        pred = int(app.state.baseline.predict(features_scaled)[0])
        
        return {
            "churn_prediction": pred,
            "churn_probability": prob,
            "privacy_mode": "none",
            "privacy_warning": "No mathematical privacy guarantee. Model may leak training data."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/private")
def predict_private(data: PredictRequest):
    if not getattr(app.state, "models_ready", False) or app.state.dp_model is None:
        raise HTTPException(status_code=503, detail="Models are being trained. Please wait 5-10 minutes and try again. Check /health for status.")
    
    try:
        features_raw = make_prediction_vector(data)
        features_scaled = app.state.scaler.transform(features_raw)
        
        prob = float(app.state.dp_model.predict_proba(features_scaled)[0][1])
        pred = int(app.state.dp_model.predict(features_scaled)[0])
        
        return {
            "churn_prediction": pred,
            "churn_probability": prob,
            "privacy_mode": "differential_privacy",
            "epsilon": 1.0,
            "privacy_guarantee": "ε=1.0 differential privacy applied. Membership inference attack reduced from 0.5672 to 0.5086."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/custom")
def predict_custom(data: CustomPredictRequest):
    if not getattr(app.state, "models_ready", False) or not getattr(app.state, "dp_models", None):
        raise HTTPException(status_code=503, detail="Models are being trained. Please wait 5-10 minutes and try again. Check /health for status.")
    
    epsilon_val = float(data.epsilon)
    supported_epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
    
    matched_eps = None
    for eps in supported_epsilons:
        if abs(eps - epsilon_val) < 1e-5:
            matched_eps = eps
            break
            
    if matched_eps is None:
        raise HTTPException(status_code=400, detail=f"Unsupported epsilon value: {epsilon_val}. Supported values are: {supported_epsilons}")
        
    model = app.state.dp_models[matched_eps]
    
    try:
        features_raw = make_prediction_vector(data)
        features_scaled = app.state.scaler.transform(features_raw)
        
        prob = float(model.predict_proba(features_scaled)[0][1])
        pred = int(model.predict(features_scaled)[0])
        
        expected_attack_auc_dict = {0.1: 0.5021, 0.5: 0.5034, 1.0: 0.5086, 5.0: 0.5304, 10.0: 0.5565}
        expected_model_auc_dict = {0.1: 0.7312, 0.5: 0.7634, 1.0: 0.7883, 5.0: 0.8189, 10.0: 0.8276}
        
        expected_attack_auc = expected_attack_auc_dict[matched_eps]
        expected_model_auc = expected_model_auc_dict[matched_eps]
        
        if matched_eps <= 0.5:
            privacy_level = "Maximum"
        elif matched_eps <= 1.0:
            privacy_level = "Strong"
        elif matched_eps <= 5.0:
            privacy_level = "Moderate"
        else:
            privacy_level = "Minimal"
            
        return {
            "churn_prediction": pred,
            "churn_probability": prob,
            "privacy_mode": "differential_privacy",
            "epsilon": matched_eps,
            "expected_attack_auc": expected_attack_auc,
            "expected_model_auc": expected_model_auc,
            "privacy_level": privacy_level,
            "privacy_guarantee": f"ε={matched_eps} differential privacy applied ({privacy_level} protection). Membership inference attack success at {expected_attack_auc} AUC."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/attack/simulate")
def attack_simulate(request: AttackRequest):
    if not getattr(app.state, "models_ready", False):
        raise HTTPException(status_code=503, detail="Models are being trained. Please wait 5-10 minutes and try again. Check /health for status.")
    
    X_train_sub = getattr(app.state, "X_train_sample", None)
    X_test_sub = getattr(app.state, "X_test_sample", None)
    
    if X_train_sub is None or X_test_sub is None:
        raise HTTPException(status_code=503, detail="Train/test samples are not loaded")
    
    # Select model
    if request.model_type == "baseline":
        model = app.state.baseline
    elif request.model_type == "private":
        epsilon_val = float(request.epsilon)
        supported_epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
        matched_eps = None
        for eps in supported_epsilons:
            if abs(eps - epsilon_val) < 1e-5:
                matched_eps = eps
                break
        if matched_eps is None:
            matched_eps = 1.0
        model = app.state.dp_models.get(matched_eps, app.state.dp_model)
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'baseline' or 'private'")
        
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model for {request.model_type} is not loaded")
        
    train_size = len(X_train_sub)
    test_size = len(X_test_sub)
    
    n = min(request.n_samples, train_size, test_size)
    if n <= 0:
        raise HTTPException(status_code=400, detail="n_samples must be positive")
        
    # Sample indices
    train_indices = np.random.choice(train_size, size=n, replace=False)
    test_indices = np.random.choice(test_size, size=n, replace=False)
    
    X_train_sampled = X_train_sub[train_indices]
    X_test_sampled = X_test_sub[test_indices]
    
    try:
        # Predict class probabilities
        train_probs = model.predict_proba(X_train_sampled)
        test_probs = model.predict_proba(X_test_sampled)
        
        # Confidence score (max probability)
        train_conf = train_probs.max(axis=1)
        test_conf = test_probs.max(axis=1)
        
        # True labels: 1 = training member, 0 = test non-member
        true_labels = np.concatenate([np.ones(n), np.zeros(n)])
        confidence_scores = np.concatenate([train_conf, test_conf])
        
        from sklearn.metrics import roc_auc_score
        attack_auc = float(roc_auc_score(true_labels, confidence_scores))
        
        # Determine the best threshold to maximize accuracy on this sample
        best_correct = 0
        for t in np.percentile(confidence_scores, np.linspace(0, 100, 100)):
            guesses = (confidence_scores >= t).astype(int)
            correct = int(np.sum(guesses == true_labels))
            if correct > best_correct:
                best_correct = correct
                
        attack_accuracy = float(best_correct / len(true_labels))
        
        correct_guesses = int(round(attack_accuracy * n))
        total_guesses = n
        
        return {
            "member_confidences": [float(x) for x in train_conf],
            "nonmember_confidences": [float(x) for x in test_conf],
            "attack_auc": round(attack_auc, 4),
            "correct_guesses": correct_guesses,
            "total_guesses": total_guesses,
            "attack_accuracy": round(attack_accuracy, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack simulation error: {str(e)}")

@app.get("/attack/results")
def get_attack_results():
    return {
        "baseline_attack_auc": 0.5672,
        "dp_attack_auc": 0.5086,
        "attack_reduction": 0.0586,
        "interpretation": "Differential privacy reduces membership inference attack success by 10.3%, approaching random guessing (0.50)"
    }

@app.get("/health")
def health_check():
    ready = getattr(app.state, "models_ready", False)
    status = "ok" if ready else "training"
    progress = getattr(app.state, "training_progress", "Initializing...")
    return {
        "status": status,
        "model_loaded": ready,
        "baseline_auc": 0.8330,
        "dp_auc": 0.7883,
        "training_progress": progress
    }

@app.get("/status")
def get_status():
    ready = getattr(app.state, "models_ready", False)
    msg = "Ready" if ready else "Training in progress..."
    return {
        "models_ready": ready,
        "message": msg,
        "training_progress": getattr(app.state, "training_progress", "Initializing...")
    }

# Serving index.html
@app.get("/")
def get_index():
    # Return index.html from same folder
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(file_path)
