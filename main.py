from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Init FastAPI app
app = FastAPI(title="Disease Prevalence Prediction API")

# Load model dan scaler
try:
    rf_model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    logging.info("✅ Model dan Scaler berhasil dimuat.")
except Exception as e:
    logging.error(f"❌ Gagal load model atau scaler: {e}")
    rf_model = None
    scaler = None

# Skema input data
class DiseaseData(BaseModel):
    PrevalenceRate: float
    IncidenceRate: float
    MortalityRate: float
    AgeGroup: str
    Gender: int
    Country: str
    AverageTreatmentCost: float
    AvailabilityOfVaccinesTreatment: int

    def __init__(self, **data):
        super().__init__(**data)
        if self.Gender not in [0, 1]:
            raise ValueError("Gender harus 0 (Female) atau 1 (Male)")
        if self.AvailabilityOfVaccinesTreatment not in [0, 1]:
            raise ValueError("AvailabilityOfVaccinesTreatment harus 0 (No) atau 1 (Yes)")

# Preprocessing function
def preprocess_input(data: DiseaseData):
    try:
        df = pd.DataFrame([{
            "Prevalence Rate (%)": data.PrevalenceRate,
            "Incidence Rate (%)": data.IncidenceRate,
            "Mortality Rate (%)": data.MortalityRate,
            "Age Group": data.AgeGroup,
            "Gender": data.Gender,
            "Country": data.Country,
            "Average Treatment Cost (USD)": data.AverageTreatmentCost,
            "Availability of Vaccines/Treatment": "Yes" if data.AvailabilityOfVaccinesTreatment == 1 else "No"
        }])

        # Map AgeGroup
        age_mapping = {'19-35': 0, '36-60': 1, '61+': 2}
        df['Age Group'] = df['Age Group'].map(age_mapping).fillna(-1).astype(int)
        if (df['Age Group'] == -1).any():
            raise ValueError("AgeGroup tidak valid. Harus salah satu dari: 19-35, 36-60, atau 61+")

        # One-hot encode Country
        df = pd.concat([df.drop('Country', axis=1), pd.get_dummies(df['Country'], prefix="Country")], axis=1)

        # One-hot encode Availability of Vaccines/Treatment
        df = pd.concat([df.drop('Availability of Vaccines/Treatment', axis=1),
                        pd.get_dummies(df['Availability of Vaccines/Treatment'],
                                       prefix="Availability of Vaccines/Treatment")], axis=1)

        # Pastikan fitur sesuai dengan scaler
        scaler_features = getattr(scaler, "feature_names_in_", None)
        if scaler_features is not None:
            df = df.reindex(columns=scaler_features, fill_value=0)

        # Scaling
        df_scaled = scaler.transform(df)
        return df_scaled

    except ValueError as ve:
        logging.error(f"ValueError preprocessing: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Unexpected error preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Preprocessing gagal.")

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "Disease Prevalence Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
async def predict_prevalence(data: DiseaseData):
    if rf_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model atau Scaler belum siap.")

    try:
        processed_data = preprocess_input(data)

        # Fix: convert prediction to float
        prediction = float(rf_model.predict(processed_data)[0])

        return {
            "PrevalenceRate": data.PrevalenceRate,
            "IncidenceRate": data.IncidenceRate,
            "MortalityRate": data.MortalityRate,
            "AgeGroup": data.AgeGroup,
            "Gender": data.Gender,
            "Country": data.Country,
            "AverageTreatmentCost": data.AverageTreatmentCost,
            "AvailabilityOfVaccinesTreatment": data.AvailabilityOfVaccinesTreatment,
            "Predicted Disease Prevalence": prediction
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Unexpected error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))
