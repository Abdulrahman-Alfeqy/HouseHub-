"""
modules/vision.py
-----------------
Document Inspector wrapper for ELA and OCR.
"""

import os
import re
import tempfile
from datetime import datetime

import cv2
import numpy as np
import easyocr
from PIL import Image

FLAGGED_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "flagged_docs")
os.makedirs(FLAGGED_DOCS_DIR, exist_ok=True)

class DocumentInspector:
    def __init__(self):
        # We initialize EasyOCR but catch exceptions if it fails in restricted environments
        try:
            self.reader = easyocr.Reader(['ar', 'en'], gpu=False)
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            self.reader = None

    def detect_forgery(self, image_bytes) -> tuple[bool, float, str]:
        """
        Runs Error Level Analysis on image bytes.
        Returns (is_forged, confidence, reason)
        """
        # Convert bytes to numpy array for cv2
        np_arr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if original is None:
            return False, 0.0, "Could not decode image."

        # Write to temp to compress
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        cv2.imwrite(tmp_path, original, [cv2.IMWRITE_JPEG_QUALITY, 90])
        recompressed = cv2.imread(tmp_path)
        os.unlink(tmp_path)
        
        ela_diff = cv2.absdiff(original, recompressed).astype(np.float32)
        ela_amplified = np.clip(ela_diff * 20, 0, 255).astype(np.uint8)
        
        suspicious_mask = np.any(ela_amplified > 15, axis=2)
        tamper_score = float(suspicious_mask.sum() / suspicious_mask.size)
        
        ELA_TAMPER_THRESHOLD = 0.15
        is_forged = tamper_score > ELA_TAMPER_THRESHOLD
        
        normalized_score = min(tamper_score / ELA_TAMPER_THRESHOLD, 1.0)
        confidence = normalized_score if is_forged else 1.0 - normalized_score
        
        reason = "Clean document."
        if is_forged:
            reason = "ELA detected non-uniform recompression artifacts — possible alteration."
            
            # Save flagged document
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ela_image_path = os.path.join(FLAGGED_DOCS_DIR, f"ela_result_{timestamp}.jpg")
            cv2.imwrite(ela_image_path, ela_amplified)
            
        return is_forged, confidence * 100.0, reason

    def extract_numbers(self, image_bytes) -> float:
        """
        Extracts largest plausible number representing Income/Rent.
        """
        if not self.reader:
            return 0.0
            
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return 0.0

        results = self.reader.readtext(image, detail=1)
        if not results:
            return 0.0
            
        number_pattern = re.compile(r"\b\d[\d,\.]*\d|\b\d\b")
        keywords = r"(income|rent|salary|دخل|إيجار|راتب|صافي)"
        keyword_pattern = re.compile(keywords, flags=re.IGNORECASE | re.UNICODE)
        
        keyword_matches = []
        all_numbers = []
        
        def convert_arabic_numerals(t: str) -> str:
            translation_map = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            return t.translate(translation_map)
        
        for (_, text, _) in results:
            text = convert_arabic_numerals(text)
            nums = []
            for m in number_pattern.findall(text):
                clean = m.replace(",", "")
                try:
                    nums.append(float(clean))
                except:
                    pass
            
            all_numbers.extend(nums)
            if keyword_pattern.search(text) and nums:
                keyword_matches.extend(nums)
                
        if keyword_matches:
            return max(keyword_matches)
        if all_numbers:
            return max(all_numbers)
            
        return 0.0
