import pytest
import os
from modules.utils import PrivacyShield, _get_or_create_key

def test_encryption_decryption():
    """Test that AES-256 encryption and decryption work correctly."""
    original_name = "Jane Doe"
    
    # Encrypt
    encrypted = PrivacyShield.encrypt_name(original_name)
    assert encrypted != original_name
    assert isinstance(encrypted, str)
    
    # Decrypt
    decrypted = PrivacyShield.decrypt_name(encrypted)
    assert decrypted == original_name

def test_hashing():
    """Test that National ID hashing is consistent."""
    nid = "12345678901234"
    hash1 = PrivacyShield.hash_national_id(nid)
    hash2 = PrivacyShield.hash_national_id(nid)
    
    assert hash1 == hash2
    assert len(hash1) == 64

def test_key_management():
    """Test that the key is created and can be loaded."""
    key = _get_or_create_key()
    assert len(key) == 44 # Fernet keys are 32 url-safe base64-encoded bytes (44 chars)
