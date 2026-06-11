"""
ml_scoring.py
-------------
Machine Learning scoring engine for the HouseHub public housing allocation
system. Implements a transparent, auditable, and bias-resistant Need Score
calculator using a Scikit-learn Random Forest Regressor trained on synthetic
but realistic housing need data.

The scoring model produces a Need Score from 0 to 100:
  - 100 = Most urgent housing need (low income, high dependents, poor housing)
  -   0 = No apparent housing need

Features used in scoring:
  - rent_to_income_ratio: Rent burden (rent / monthly income). Higher = worse.
  - dependents: Number of dependents. Higher = higher priority.
  - current_housing_quality: Enum 0–3 (0=homeless, 1=severe, 2=moderate, 3=adequate).
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


class RankingEngine:
    """
    A transparent ML-based housing need scoring engine.

    The engine trains a Random Forest Regressor on synthetic data that models
    realistic housing need scenarios. The model is trained once at initialization
    and cached for subsequent scoring calls.

    Attributes:
        model (RandomForestRegressor): The trained scikit-learn model.
        scaler (MinMaxScaler): Feature scaler fitted on training data.
        feature_columns (list): Names of features the model expects.
        is_trained (bool): Whether the model has been successfully trained.
        training_mae (float): Mean Absolute Error on holdout test set.
    """

    FEATURE_COLUMNS = ["rent_to_income_ratio", "dependents", "current_housing_quality"]

    def __init__(self) -> None:
        """Initializes the RankingEngine and trains the model immediately."""
        self.model: RandomForestRegressor = None
        self.scaler: MinMaxScaler = MinMaxScaler()
        self.feature_columns: list = self.FEATURE_COLUMNS
        self.is_trained: bool = False
        self.training_mae: float = float("inf")

        logger.info("RankingEngine initializing — training model on synthetic data.")
        self.train_model()

    def _generate_synthetic_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Generates a synthetic but realistic dataset of housing applicants
        for training the Need Score model.

        The dataset is designed to reflect real-world housing need patterns:
          - Higher rent burden → higher need.
          - More dependents → higher need.
          - Poorer housing quality → higher need.
          - A small amount of noise is added to prevent overfitting to
            a perfectly linear relationship.

        Args:
            n_samples (int): Number of synthetic applicants to generate.

        Returns:
            pd.DataFrame: DataFrame with feature columns and a 'need_score' target.
        """
        rng = np.random.default_rng(seed=42)  # Reproducible

        # --- Feature generation ---
        # Rent-to-income ratio: 0.10 (healthy) to 1.50 (in crisis)
        rent_to_income = rng.uniform(0.1, 1.5, n_samples)

        # Dependents: 0 to 8 (integer)
        dependents = rng.integers(0, 9, n_samples).astype(float)

        # Housing quality: 0=homeless, 1=severe overcrowding, 2=moderate, 3=adequate
        housing_quality = rng.integers(0, 4, n_samples).astype(float)

        # --- Target: Need Score (0–100) ---
        # Deterministic scoring function reflecting real policy weights:
        #   - Rent burden: 40% weight (most important factor)
        #   - Dependents: 35% weight
        #   - Housing quality: 25% weight (inverted: worse quality = higher need)
        rent_score = np.clip((rent_to_income - 0.1) / 1.4, 0, 1) * 40
        dependent_score = (dependents / 8.0) * 35
        quality_score = ((3 - housing_quality) / 3.0) * 25  # Inverted

        # Combine and add Gaussian noise to simulate real-world imperfection
        raw_score = rent_score + dependent_score + quality_score
        noise = rng.normal(0, 3, n_samples)  # ±3 points of noise
        need_score = np.clip(raw_score + noise, 0, 100)

        df = pd.DataFrame(
            {
                "rent_to_income_ratio": rent_to_income,
                "dependents": dependents,
                "current_housing_quality": housing_quality,
                "need_score": need_score,
            }
        )

        logger.debug(
            "Synthetic dataset generated: %d samples, score range: [%.1f, %.1f]",
            n_samples,
            df["need_score"].min(),
            df["need_score"].max(),
        )
        return df

    def train_model(self) -> None:
        """
        Generates synthetic training data, scales features, trains a Random
        Forest Regressor, and validates it on a holdout test set.

        The training process:
          1. Generate 2000 synthetic applicant records.
          2. Fit a MinMaxScaler on training features.
          3. Train a Random Forest with 150 trees (n_estimators=150).
          4. Evaluate on a 20% holdout set and log Mean Absolute Error.

        After this method completes, `self.is_trained` is True and subsequent
        calls to `calculate_score()` will use the fitted model.
        """
        try:
            # Step 1: Generate training data
            df = self._generate_synthetic_data(n_samples=2000)

            X = df[self.feature_columns].values
            y = df["need_score"].values

            # Step 2: Train/test split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )

            # Step 3: Fit scaler on training data only (prevent data leakage)
            self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Step 4: Train Random Forest Regressor
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,  # Use all available CPU cores
            )
            self.model.fit(X_train_scaled, y_train)

            # Step 5: Evaluate on holdout set
            y_pred = self.model.predict(X_test_scaled)
            self.training_mae = float(mean_absolute_error(y_test, y_pred))

            self.is_trained = True
            logger.info(
                "RankingEngine training complete. MAE on holdout: %.2f points",
                self.training_mae,
            )

        except Exception as exc:
            logger.error("Model training failed: %s", exc)
            self.is_trained = False
            raise RuntimeError(f"Failed to train scoring model: {exc}") from exc

    def _derive_features_from_income(
        self, income: int, dependents: int = 2, current_housing_quality: int = 1
    ) -> Dict[str, float]:
        """
        Derives the rent_to_income_ratio and other features from an income value
        when a full application form is not provided.

        Uses a realistic model: average rent in the city is 1200 currency units,
        adjusted by household size.

        Args:
            income (int): Monthly income in local currency.
            dependents (int): Number of dependents (0–8).
            current_housing_quality (int): Housing quality level (0–3).

        Returns:
            Dict[str, float]: Feature dictionary ready for scoring.
        """
        # Estimate rent based on dependents (larger families need larger homes)
        estimated_rent = 800 + (dependents * 200)
        rent_to_income = estimated_rent / max(income, 1)
        rent_to_income = min(rent_to_income, 1.5)  # Cap at 1.5 (150% of income)

        return {
            "rent_to_income_ratio": float(rent_to_income),
            "dependents": float(dependents),
            "current_housing_quality": float(current_housing_quality),
        }

    def calculate_score(self, user_data: Dict) -> Tuple[float, str]:
        """
        Calculates the objective housing Need Score for a citizen.

        Accepts a flexible dictionary of user data and derives all necessary
        features for the model. If the model is not trained, falls back to a
        deterministic algorithmic score.

        Args:
            user_data (Dict): A dictionary that may contain:
                - income (int): Monthly income (required).
                - dependents (int): Number of dependents. Default: 2.
                - current_housing_quality (int): 0–3. Default: 1.
                - rent_to_income_ratio (float): Direct ratio if known.

        Returns:
            Tuple[float, str]:
                - score (float): Need Score from 0 to 100 (higher = more urgent).
                - breakdown (str): Human-readable score explanation.

        Raises:
            ValueError: If income is not provided or is non-positive.
        """
        # --- Extract and validate inputs ---
        income = int(user_data.get("income", 0))
        dependents = int(user_data.get("dependents", 2))
        current_housing_quality = int(user_data.get("current_housing_quality", 1))

        if income <= 0:
            raise ValueError("Income must be a positive integer.")

        # Clamp inputs to valid ranges
        dependents = max(0, min(dependents, 8))
        current_housing_quality = max(0, min(current_housing_quality, 3))

        # --- Feature engineering ---
        if "rent_to_income_ratio" in user_data:
            rent_to_income = float(user_data["rent_to_income_ratio"])
        else:
            features_dict = self._derive_features_from_income(
                income, dependents, current_housing_quality
            )
            rent_to_income = features_dict["rent_to_income_ratio"]

        feature_vector = np.array(
            [[rent_to_income, float(dependents), float(current_housing_quality)]]
        )

        # --- Model prediction (or algorithmic fallback) ---
        if self.is_trained and self.model is not None:
            try:
                scaled_features = self.scaler.transform(feature_vector)
                raw_score = float(self.model.predict(scaled_features)[0])
                score = round(max(0.0, min(100.0, raw_score)), 2)
                method = "random_forest_v1"
            except Exception as exc:
                logger.warning("Model prediction failed, using fallback: %s", exc)
                score = self._algorithmic_fallback(
                    rent_to_income, dependents, current_housing_quality
                )
                method = "algorithmic_fallback"
        else:
            score = self._algorithmic_fallback(
                rent_to_income, dependents, current_housing_quality
            )
            method = "algorithmic_fallback"

        # --- Build human-readable breakdown ---
        quality_labels = {
            0: "Homeless/Emergency Shelter",
            1: "Severe Overcrowding",
            2: "Moderate Substandard",
            3: "Adequate but Unaffordable",
        }
        breakdown = (
            f"Need Score: {score:.1f}/100 | "
            f"Income: {income:,} | "
            f"Rent Burden: {rent_to_income:.0%} | "
            f"Dependents: {dependents} | "
            f"Housing: {quality_labels.get(current_housing_quality, 'Unknown')} | "
            f"Method: {method}"
        )

        logger.info("Score calculated: %.2f for income=%d", score, income)
        return score, breakdown

    def _algorithmic_fallback(
        self,
        rent_to_income: float,
        dependents: int,
        housing_quality: int,
    ) -> float:
        """
        A pure deterministic scoring algorithm used as fallback when the ML
        model is unavailable. Mirrors the synthetic data generation formula.

        Args:
            rent_to_income (float): Rent-to-income ratio.
            dependents (int): Number of dependents.
            housing_quality (int): Housing quality score (0–3).

        Returns:
            float: Need score from 0 to 100.
        """
        rent_score = min(((rent_to_income - 0.1) / 1.4), 1.0) * 40
        dependent_score = (min(dependents, 8) / 8.0) * 35
        quality_score = ((3 - min(housing_quality, 3)) / 3.0) * 25

        total = rent_score + dependent_score + quality_score
        return round(max(0.0, min(100.0, total)), 2)

    def get_model_info(self) -> Dict:
        """
        Returns metadata about the trained model for transparency and
        auditability in the Admin Dashboard.

        Returns:
            Dict: Model metadata including MAE, feature importance, and status.
        """
        if not self.is_trained or self.model is None:
            return {"status": "not_trained", "mae": None, "feature_importance": {}}

        importance = dict(
            zip(self.feature_columns, self.model.feature_importances_.tolist())
        )
        return {
            "status": "trained",
            "algorithm": "RandomForestRegressor",
            "n_estimators": self.model.n_estimators,
            "mae": round(self.training_mae, 3),
            "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        }
