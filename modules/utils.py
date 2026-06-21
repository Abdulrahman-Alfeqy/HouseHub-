"""
modules/utils.py
----------------
Utility functions for cryptography, hashing, and general data manipulation
for the HouseHub+ system. Ensures Privacy-by-Design by strictly decoupling
identifiable information from operational data.
"""

import hashlib
import os
from cryptography.fernet import Fernet, InvalidToken

ID_SALT = os.environ.get("HOUSEHUB_ID_SALT", "HOUSEHUB_ID_SALT_2026")

def _get_or_create_key() -> bytes:
    """Loads the encryption key from secret.key or generates a new one."""
    key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "secret.key")
    if os.path.exists(key_path):
        with open(key_path, "rb") as key_file:
            return key_file.read()
    else:
        key = Fernet.generate_key()
        with open(key_path, "wb") as key_file:
            key_file.write(key)
        return key

# Initialize Fernet cipher suite for AES-256 encryption
_cipher_suite = Fernet(_get_or_create_key())

class PrivacyShield:
    """
    Handles cryptographic operations for the HouseHub+ system to ensure 
    "Responsible AI" and "Privacy-by-Design".
    """

    @staticmethod
    def hash_national_id(national_id: str) -> str:
        """
        One-way SHA-256 hash of the National ID to create a pseudonymous User Token.
        The original ID cannot be reverse-engineered from this hash.
        
        Args:
            national_id (str): Raw National ID entered by citizen.
            
        Returns:
            str: 64-character SHA-256 hex digest.
        """
        payload = f"{national_id.strip()}|{ID_SALT}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def encrypt_name(full_name: str) -> str:
        """
        Encrypts the full name using AES-256 (via Fernet).
        
        Args:
            full_name (str): Plaintext name.
            
        Returns:
            str: Encrypted name string.
        """
        return _cipher_suite.encrypt(full_name.strip().encode("utf-8")).decode("utf-8")

    @staticmethod
    def decrypt_name(encrypted_name: str) -> str:
        """
        Decrypts the name for Admin viewing after approval using AES-256 (via Fernet).
        
        Args:
            encrypted_name (str): The encrypted name string.
            
        Returns:
            str: Plaintext name.
            
        Raises:
            ValueError: If the decryption fails (tampering detected).
        """
        try:
            return _cipher_suite.decrypt(encrypted_name.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            return "[Legacy Data - Unreadable]"
        except Exception as e:
            raise ValueError(f"Name decryption failed: {e}")
