"""
modules/database.py
-------------------
SQLite3 database manager for HouseHub+.
Enforces strict Data Layer Separation for Privacy-by-Design.

Tables:
  1. identity_data: Stores encrypted PII.
  2. assessment_data: Stores anonymous scoring, demographics, and GPS data.
"""

import sqlite3
import logging
import os
from typing import Optional, Dict, Any, List

from modules.utils import PrivacyShield

# Configuration
DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "housing_system.db")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s")
log = logging.getLogger("HouseHub+ DB")

class DatabaseManager:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initializes tables for identity and anonymous assessment data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table 1: PII Layer
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS identity_data (
                        user_token TEXT PRIMARY KEY,
                        encrypted_name TEXT NOT NULL
                    )
                """)
                
                # Table 2: Assessment Layer (Anonymous)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS assessment_data (
                        user_token TEXT PRIMARY KEY,
                        income REAL NOT NULL,
                        rent REAL NOT NULL,
                        family_size INTEGER NOT NULL,
                        marital_status TEXT NOT NULL,
                        special_conditions INTEGER NOT NULL,
                        lat REAL NOT NULL,
                        lon REAL NOT NULL,
                        district_cluster INTEGER NOT NULL,
                        need_score REAL NOT NULL,
                        tier TEXT NOT NULL,
                        is_forged INTEGER NOT NULL DEFAULT 0,
                        forgery_confidence REAL DEFAULT 0,
                        forgery_reason TEXT,
                        status TEXT NOT NULL DEFAULT 'Pending',
                        rejection_count INTEGER NOT NULL DEFAULT 0,
                        rejection_reason TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except sqlite3.Error as e:
            log.error(f"Database initialization failed: {e}")
            raise

    def get_applicant_by_token(self, user_token: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the anonymous assessment data for an applicant using their hashed token.
        Used for the auto-reload feature.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM assessment_data WHERE user_token = ?", 
                    (user_token,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            log.error(f"Fetch failed: {e}")
            return None

    def upsert_applicant(self, 
                         name: str, 
                         national_id: str, 
                         income: float, 
                         rent: float, 
                         family_size: int, 
                         marital_status: str, 
                         special_conditions: int,
                         lat: float,
                         lon: float,
                         district_cluster: int,
                         need_score: float,
                         tier: str,
                         is_forged: bool,
                         forgery_confidence: float = 0.0,
                         forgery_reason: str = "") -> str:
        """
        Inserts or updates an applicant.
        If the user_token exists, updates the assessment_data.
        Otherwise, inserts into both identity_data and assessment_data.
        """
        user_token = PrivacyShield.hash_national_id(national_id)
        encrypted_name = PrivacyShield.encrypt_name(name)
        is_forged_int = 1 if is_forged else 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if exists
                cursor.execute("SELECT 1 FROM identity_data WHERE user_token = ?", (user_token,))
                exists = cursor.fetchone() is not None

                if not exists:
                    # Insert Identity
                    cursor.execute(
                        "INSERT INTO identity_data (user_token, encrypted_name) VALUES (?, ?)",
                        (user_token, encrypted_name)
                    )
                else:
                    # Optionally update the encrypted name if it changed
                    cursor.execute(
                        "UPDATE identity_data SET encrypted_name = ? WHERE user_token = ?",
                        (encrypted_name, user_token)
                    )

                # Upsert Assessment Data
                cursor.execute("""
                    INSERT INTO assessment_data (
                        user_token, income, rent, family_size, marital_status,
                        special_conditions, lat, lon, district_cluster, need_score,
                        tier, is_forged, forgery_confidence, forgery_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_token) DO UPDATE SET
                        income=excluded.income,
                        rent=excluded.rent,
                        family_size=excluded.family_size,
                        marital_status=excluded.marital_status,
                        special_conditions=excluded.special_conditions,
                        lat=excluded.lat,
                        lon=excluded.lon,
                        district_cluster=excluded.district_cluster,
                        need_score=excluded.need_score,
                        tier=excluded.tier,
                        is_forged=excluded.is_forged,
                        forgery_confidence=excluded.forgery_confidence,
                        forgery_reason=excluded.forgery_reason,
                        status='Pending',
                        timestamp=CURRENT_TIMESTAMP
                """, (user_token, income, rent, family_size, marital_status, special_conditions,
                      lat, lon, district_cluster, need_score, tier, is_forged_int, 
                      forgery_confidence, forgery_reason))
                
                conn.commit()
                return user_token
        except sqlite3.Error as e:
            log.error(f"Upsert failed: {e}")
            raise

    def get_queue(self, forged_only: bool = False) -> List[Dict[str, Any]]:
        """
        Fetches the anonymous queue for the Admin Dashboard.
        Sorted by need_score DESC.
        """
        query = "SELECT * FROM assessment_data WHERE is_forged = ? ORDER BY need_score DESC, timestamp ASC"
        flag = 1 if forged_only else 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (flag,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            log.error(f"Queue fetch failed: {e}")
            return []
            
    def get_heatmap_data(self) -> List[Dict[str, Any]]:
        """Fetches all valid Lat/Lon points and their need scores for the Heatmap."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT lat, lon, need_score, district_cluster FROM assessment_data WHERE is_forged = 0")
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            log.error(f"Heatmap data fetch failed: {e}")
            return []

    def approve_applicant(self, user_token: str) -> Optional[str]:
        """
        Approves an application. Returns the decrypted plaintext name so the admin
        can contact the applicant. This represents the Human-in-the-Loop decision.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Update status
                cursor.execute("UPDATE assessment_data SET status = 'Approved' WHERE user_token = ?", (user_token,))
                if cursor.rowcount == 0:
                    return None
                
                # Fetch encrypted name
                cursor.execute("SELECT encrypted_name FROM identity_data WHERE user_token = ?", (user_token,))
                row = cursor.fetchone()
                if not row:
                    return None
                
                encrypted_name = row[0]
                conn.commit()
                
                return PrivacyShield.decrypt_name(encrypted_name)
        except sqlite3.Error as e:
            log.error(f"Approve failed: {e}")
            return None

    def reject_applicant(self, user_token: str, reason: str) -> Dict[str, Any]:
        """
        Rejects an application, increments rejection count.
        Returns the new state (for escalation logic).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE assessment_data 
                    SET status = 'Rejected', 
                        rejection_count = rejection_count + 1,
                        rejection_reason = ?
                    WHERE user_token = ?
                """, (reason, user_token))
                
                conn.commit()
                
                cursor.row_factory = sqlite3.Row
                cursor.execute("SELECT status, rejection_count FROM assessment_data WHERE user_token = ?", (user_token,))
                row = cursor.fetchone()
                
                # Escalate if rejected 3 or more times
                if row and row['rejection_count'] >= 3:
                    cursor.execute("UPDATE assessment_data SET status = 'Escalated to Senior Supervisor' WHERE user_token = ?", (user_token,))
                    conn.commit()
                    return {"status": "Escalated to Senior Supervisor", "rejection_count": row['rejection_count']}
                    
                return {"status": "Rejected", "rejection_count": row['rejection_count'] if row else 0}
        except sqlite3.Error as e:
            log.error(f"Reject failed: {e}")
            return {"status": "Error", "rejection_count": 0}
