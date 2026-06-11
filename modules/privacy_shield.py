"""
privacy_shield.py
-----------------
Implements Privacy-First PII anonymization for the HouseHub system.

All Personally Identifiable Information (PII) is hashed immediately upon
ingestion using SHA-256 to ensure unbiased, privacy-preserving processing.
No raw names or national IDs are stored anywhere in the pipeline.
"""

import hashlib
import secrets
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PrivacyShield:
    """
    A privacy-first utility class that anonymizes citizen PII before any
    processing occurs. Implements the principle of data minimization by
    converting identifying information into deterministic, irreversible hashes.

    Design Principles:
        - Deterministic: The same inputs always produce the same hashed ID,
          allowing consistent tracking without storing PII.
        - Irreversible: SHA-256 hashing cannot be reversed to recover PII.
        - Consistent: A citizen re-applying will receive the same hashed ID,
          enabling deduplication without re-storing their private data.
    """

    _SALT: str = "HouseHub-v1-AntiCollisionSalt-2024"

    @staticmethod
    def anonymize_data(name: str, national_id: str) -> str:
        """
        Converts a citizen's full name and national ID into a single
        deterministic SHA-256 anonymized identifier.

        The method concatenates the inputs with a known application-level salt
        to prevent rainbow table attacks, then hashes the combined string.

        Args:
            name (str): The citizen's full legal name.
            national_id (str): The citizen's national ID number.

        Returns:
            str: A 64-character hexadecimal SHA-256 hash representing the
                 anonymous citizen identity. Safe to store in databases and
                 display publicly.

        Raises:
            ValueError: If either name or national_id is empty or not a string.

        Example:
            >>> shield = PrivacyShield()
            >>> hashed = PrivacyShield.anonymize_data("Jane Doe", "ID-987654")
            >>> len(hashed)
            64
        """
        # --- Input Validation ---
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Citizen name must be a non-empty string.")
        if not isinstance(national_id, str) or not national_id.strip():
            raise ValueError("National ID must be a non-empty string.")

        # --- Normalize inputs to prevent trivial bypass attempts ---
        normalized_name = name.strip().upper()
        normalized_id = national_id.strip().upper()

        # --- Construct the canonical plaintext for hashing ---
        # Format: SALT::NAME||ID — the salt prevents cross-system rainbow tables,
        # and the separator prevents collision attacks (e.g., "JohnDoe" vs "John" + "Doe").
        canonical_input = (
            f"{PrivacyShield._SALT}::{normalized_name}||{normalized_id}"
        )

        # --- Compute SHA-256 hash ---
        hashed_bytes = hashlib.sha256(canonical_input.encode("utf-8"))
        hashed_id = hashed_bytes.hexdigest()

        logger.info(
            "PII anonymized successfully. Hash prefix: %s...", hashed_id[:8]
        )

        return hashed_id

    @staticmethod
    def generate_session_token() -> str:
        """
        Generates a cryptographically secure random token for a citizen's
        session, used to track the submission within a single session
        without tying it to any identity.

        Returns:
            str: A 32-character URL-safe random token.
        """
        return secrets.token_urlsafe(24)

    @staticmethod
    def mask_hash_for_display(hashed_id: str, visible_chars: int = 12) -> str:
        """
        Creates a partially masked version of a hash for safer UI display,
        showing only the first N characters followed by asterisks.

        Args:
            hashed_id (str): The full SHA-256 hash string.
            visible_chars (int): How many leading characters to show. Default 12.

        Returns:
            str: A masked string like "a3f8c1d29b0e****...****"
        """
        if not hashed_id or len(hashed_id) < visible_chars:
            return hashed_id
        return hashed_id[:visible_chars] + "..." + "*" * 8
