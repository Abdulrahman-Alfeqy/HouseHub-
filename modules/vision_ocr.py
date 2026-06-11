"""
vision_ocr.py
-------------
Computer Vision module for document authenticity verification and income
extraction in the HouseHub anti-fraud system.

This module implements two key functionalities:
  1. Forgery Detection via Error Level Analysis (ELA) simulation using OpenCV.
  2. Income Extraction simulation to retrieve financial data from uploaded docs.

Note on ELA:
    Error Level Analysis works by re-saving a JPEG image at a known quality
    level and computing the absolute difference between the original and
    re-saved image. Areas that have been digitally altered retain different
    error levels than authentic areas, creating visible artifacts.
    We simulate this using pixel variance analysis on the image data.
"""

import io
import logging
import random
import re
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DocumentInspector:
    """
    An AI-powered document verification class that uses Computer Vision
    techniques to detect potential forgeries and extract financial data
    from uploaded identity and income documents.

    Attributes:
        FORGERY_VARIANCE_THRESHOLD (float): The pixel variance threshold above
            which a document is flagged as potentially forged. Empirically
            derived; higher variance suggests JPEG re-compression artifacts.
        ELA_QUALITY (int): JPEG quality level for the re-save step in ELA.
            Lower values amplify differences in altered regions.
    """

    FORGERY_VARIANCE_THRESHOLD: float = 18.5
    ELA_QUALITY: int = 75

    def __init__(self) -> None:
        """Initializes the DocumentInspector with logging."""
        logger.info("DocumentInspector initialized.")

    def _load_image_as_numpy(self, image_file) -> np.ndarray:
        """
        Converts an uploaded Streamlit file object into an OpenCV-compatible
        NumPy array in BGR format.

        Args:
            image_file: A Streamlit UploadedFile object or any file-like object
                        with a .read() method.

        Returns:
            np.ndarray: The image as a BGR NumPy array.

        Raises:
            ValueError: If the file cannot be decoded as a valid image.
        """
        try:
            # Reset stream pointer to beginning in case it was already read
            image_file.seek(0)
            raw_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV could not decode the image file.")
            return img
        except Exception as exc:
            logger.error("Failed to load image: %s", exc)
            raise ValueError(f"Image loading failed: {exc}") from exc

    def _perform_ela(self, image_file) -> np.ndarray:
        """
        Performs Error Level Analysis by:
          1. Loading the original image via PIL.
          2. Re-saving it as a JPEG at a controlled lower quality.
          3. Computing the absolute pixel-level difference between the two.
          4. Scaling the difference for better visual/numeric contrast.

        Args:
            image_file: A file-like object for the document image.

        Returns:
            np.ndarray: The ELA difference image as a NumPy array (grayscale).
        """
        try:
            image_file.seek(0)
            original = Image.open(image_file).convert("RGB")

            # Re-save at controlled quality to introduce standard JPEG artifacts
            buffer = io.BytesIO()
            original.save(buffer, format="JPEG", quality=self.ELA_QUALITY)
            buffer.seek(0)
            recompressed = Image.open(buffer).convert("RGB")

            # Compute absolute difference and amplify it for analysis
            original_arr = np.array(original, dtype=np.float32)
            recompressed_arr = np.array(recompressed, dtype=np.float32)
            ela_image = np.abs(original_arr - recompressed_arr)

            # Scale to full 0–255 range for variance computation
            ela_scaled = (ela_image * 10).clip(0, 255).astype(np.uint8)
            ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)

            return ela_gray

        except Exception as exc:
            logger.warning(
                "ELA processing failed, falling back to noise analysis: %s", exc
            )
            # Fallback: load via OpenCV and return a grayscale version
            image_file.seek(0)
            raw_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.zeros((100, 100), dtype=np.uint8)

    def detect_forgery(self, image_file) -> Tuple[bool, float, str]:
        """
        Analyzes an uploaded document image for signs of digital forgery using
        a simulated Error Level Analysis (ELA) pipeline.

        The detection logic:
          - Computes the ELA image (difference after recompression).
          - Calculates the standard deviation of ELA pixel values.
          - High standard deviation indicates non-uniform compression artifacts,
            a hallmark of pasted or manipulated regions.
          - A simulated probabilistic component models the imperfection of
            real-world models, ensuring the system is not deterministically
            exploitable by file-format manipulation.

        Args:
            image_file: A Streamlit UploadedFile object for the document.

        Returns:
            Tuple[bool, float, str]:
                - is_forged (bool): True if the document is likely forged.
                - confidence_score (float): Confidence percentage (0–100).
                - reason (str): A human-readable explanation of the flag.
        """
        if image_file is None:
            return False, 0.0, "No image provided."

        try:
            ela_gray = self._perform_ela(image_file)

            # --- Primary Signal: ELA Standard Deviation ---
            # High std dev in ELA map = inconsistent recompression = forgery
            std_dev = float(np.std(ela_gray))
            mean_val = float(np.mean(ela_gray))

            logger.debug(
                "ELA stats — std_dev: %.2f, mean: %.2f, threshold: %.2f",
                std_dev,
                mean_val,
                self.FORGERY_VARIANCE_THRESHOLD,
            )

            # --- Secondary Signal: Local variance analysis ---
            # Checks for abrupt local discontinuities that suggest pasted regions
            kernel_size = min(ela_gray.shape[0] // 4, ela_gray.shape[1] // 4, 16)
            kernel_size = max(kernel_size, 4)
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
            local_mean = cv2.filter2D(ela_gray.astype(np.float32), -1, kernel)
            local_variance = np.var(ela_gray.astype(np.float32) - local_mean)

            # --- Composite score combining ELA std and local variance ---
            # Normalize to 0–100 confidence scale
            ela_score = min((std_dev / self.FORGERY_VARIANCE_THRESHOLD) * 50, 50)
            var_score = min((local_variance / 500) * 50, 50)
            raw_confidence = ela_score + var_score

            # --- Simulated model uncertainty (±5% noise) ---
            noise = random.uniform(-5, 5)
            confidence = max(0.0, min(100.0, raw_confidence + noise))

            is_forged = confidence >= 40.0

            if is_forged:
                reason = (
                    f"Document flagged: ELA analysis detected non-uniform "
                    f"recompression artifacts (σ={std_dev:.1f}, "
                    f"confidence={confidence:.1f}%). "
                    "Possible copy-paste or digital alteration detected."
                )
            else:
                reason = (
                    f"Document appears authentic: ELA variance within "
                    f"acceptable range (σ={std_dev:.1f}, "
                    f"confidence={100 - confidence:.1f}% authentic)."
                )

            logger.info(
                "Forgery detection complete. is_forged=%s, confidence=%.1f%%",
                is_forged,
                confidence,
            )
            return is_forged, confidence, reason

        except Exception as exc:
            logger.error("Forgery detection error: %s", exc)
            # Safe default: do not accuse documents we can't analyze
            return False, 0.0, f"Analysis could not be completed: {exc}"

    def extract_income(self, image_file) -> Tuple[int, str]:
        """
        Simulates Optical Character Recognition (OCR) to extract a citizen's
        declared monthly income from an income certificate document.

        In production, this would integrate EasyOCR or Tesseract to read actual
        text from the image. Here we simulate the process deterministically
        based on file properties to provide realistic variance for demo purposes.

        The simulation strategy:
          - Uses the image's pixel statistics as a deterministic seed.
          - Generates a plausible income range (800–8000 currency units).
          - Adds realistic OCR uncertainty noise.

        Args:
            image_file: A Streamlit UploadedFile object for the income
                        certificate document.

        Returns:
            Tuple[int, str]:
                - income (int): The extracted monthly income in local currency.
                - extraction_method (str): Description of the extraction method used.
        """
        if image_file is None:
            logger.warning("No income certificate provided; defaulting to median income.")
            return 2500, "default_median"

        try:
            image_file.seek(0)
            raw_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("Could not decode image for income extraction.")

            # --- Deterministic seed from image statistics ---
            # Using mean pixel value as a stable, document-specific seed
            pixel_mean = float(np.mean(img))
            pixel_sum = int(np.sum(img)) % 10000  # Bounded integer from image

            # --- Simulate OCR income extraction ---
            # The formula maps pixel characteristics to realistic income brackets:
            # Dark documents (low mean) → lower income (e.g., handwritten forms)
            # Bright documents (high mean) → higher income (e.g., printed payslips)
            base_income = int(800 + (pixel_mean / 255.0) * 6200)  # Range: 800–7000
            variance = int(pixel_sum % 1000)  # Document-specific variance: 0–999
            income = base_income + variance

            # Clamp to realistic range
            income = max(500, min(income, 15000))

            logger.info(
                "Income extracted via simulated OCR: %d (pixel_mean=%.1f)",
                income,
                pixel_mean,
            )
            return income, "simulated_ocr_v1"

        except Exception as exc:
            logger.error("Income extraction failed: %s", exc)
            return 2500, "fallback_median"
