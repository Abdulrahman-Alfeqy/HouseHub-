"""
scripts/generate_data.py
------------------------
Procedural generation of synthetic dataset for HouseHub+ ML Models.
Generates:
- Income, Rent, Family Size, Marital Status, Special Conditions (Categorical mapping)
- GPS Coordinates (Lat, Lon) based in a city (e.g., Riyadh)
- Generates Need Score algorithmically to provide training targets for the ML model.
- Trains the RandomForestRegressor and KMeans clustering models, saving them as .pkl.
"""

import os
import random
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_PATH, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_PATH, exist_ok=True)

def generate_synthetic_data(num_records=1000):
    np.random.seed(42)
    
    # City center: Riyadh (lat 24.77, lon 46.74)
    CENTER_LAT, CENTER_LON = 24.7741, 46.7380
    
    data = []
    for _ in range(num_records):
        # Generate demographic & financial data
        income = round(random.uniform(500, 20000), 2)
        rent = round(random.uniform(200, income * 0.8), 2)  # Rent shouldn't exceed 80% of income usually
        family_size = random.randint(1, 10)
        marital_status = random.choice(["Single", "Married", "Divorced", "Widowed"])
        special_conditions = random.choice([0, 1])  # 1 = Yes (Disability, Medical), 0 = No
        
        # Generate GPS
        lat = CENTER_LAT + np.random.normal(0, 0.05)
        lon = CENTER_LON + np.random.normal(0, 0.05)
        
        # Calculate algorithmic Need Score for training
        # Higher score = Higher Need
        # Formula: 
        #   Base = (Rent / Income) * 40
        #   Family factor = Family size * 5
        #   Special conditions = 20 if True
        #   Low Income penalty = if income < 3000 -> +20
        
        income_safeguard = max(income, 1) # Prevent div by 0
        rent_ratio = min(rent / income_safeguard, 1.0)
        
        score = (rent_ratio * 40) + (family_size * 5) + (special_conditions * 20)
        if income < 3000:
            score += 20
        
        # Cap at 100
        need_score = min(score, 100)
        
        data.append({
            "income": income,
            "rent": rent,
            "family_size": family_size,
            "marital_status": marital_status,
            "special_conditions": special_conditions,
            "lat": lat,
            "lon": lon,
            "need_score": round(need_score, 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_PATH, "training_data.csv"), index=False)
    print(f"Generated {num_records} synthetic records.")
    return df

def train_models(df):
    print("Training KMeans for Geo-AI District Clustering...")
    kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto')
    coords = df[['lat', 'lon']]
    df['district_cluster'] = kmeans.fit_predict(coords)
    
    # Save KMeans model
    joblib.dump(kmeans, os.path.join(MODEL_PATH, "geo_kmeans.pkl"))
    
    print("Training RandomForestRegressor for Priority Need Scoring...")
    
    # Features: income, rent, family_size, special_conditions, district_cluster
    # Marital status is excluded from ML directly to prevent bias, or encoded.
    # Let's keep it objective (financial + geo + dependents).
    features = ['income', 'rent', 'family_size', 'special_conditions', 'district_cluster']
    X = df[features]
    y = df['need_score']
    
    rf_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    rf_model.fit(X, y)
    
    # Evaluate
    preds = rf_model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"RandomForest MAE: {mae:.2f} points.")
    
    # Save RF Model
    joblib.dump(rf_model, os.path.join(MODEL_PATH, "ranking_engine.pkl"))
    print("Models saved successfully.")

if __name__ == "__main__":
    df = generate_synthetic_data(1500)
    train_models(df)
