"""
modules/models.py
-----------------
Interfaces for the ML Ranking Engine and Geo-AI Clustering.
"""

import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

class IntelligenceEngine:
    def __init__(self):
        self.rf_model_path = os.path.join(MODEL_PATH, "ranking_engine.pkl")
        self.kmeans_model_path = os.path.join(MODEL_PATH, "geo_kmeans.pkl")
        
        self.rf_model = None
        self.kmeans_model = None
        
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.rf_model_path) and os.path.exists(self.kmeans_model_path):
            self.rf_model = joblib.load(self.rf_model_path)
            self.kmeans_model = joblib.load(self.kmeans_model_path)
        else:
            raise FileNotFoundError("ML Models not found. Please run 'python scripts/generate_data.py' first.")

    def get_district_cluster(self, lat: float, lon: float) -> int:
        """Uses KMeans to assign a GPS coordinate to a district cluster (anonymizing exact location)."""
        if not self.kmeans_model:
            return 0
        coords = np.array([[lat, lon]])
        return int(self.kmeans_model.predict(coords)[0])

    def calculate_priority(self, income: float, rent: float, family_size: int, special_conditions: int, lat: float, lon: float) -> tuple[float, str, str]:
        """
        Calculates the objective Housing Need Score and assigns a Priority Tier.
        Returns:
            need_score (float), tier (str), breakdown (str)
        """
        district_cluster = self.get_district_cluster(lat, lon)
        
        if not self.rf_model:
            return 0.0, "Low", "Model not loaded."
            
        features = np.array([[income, rent, family_size, special_conditions, district_cluster]])
        score = float(self.rf_model.predict(features)[0])
        score = max(0.0, min(100.0, score))  # Ensure bounds
        
        if score >= 80:
            tier = "Urgent"
        elif score >= 50:
            tier = "Medium"
        else:
            tier = "Low"
            
        breakdown = (
            f"Income: {income:,.2f} | Rent: {rent:,.2f} | Family: {family_size} | "
            f"Special Needs: {'Yes' if special_conditions else 'No'} | "
            f"Geo-Cluster: {district_cluster}"
        )
            
        return round(score, 1), tier, breakdown
