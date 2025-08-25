import cv2
import numpy as np
import os
import time
import logging
from dataclasses import dataclass
from typing import List
from enum import Enum

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceDetectPro")

# =========================
# Structures
# =========================
@dataclass
class RelativeBBox:
    xmin: float
    ymin: float
    width: float
    height: float

@dataclass
class Detection:
    bbox: RelativeBBox
    score: float

class DetectionMode(Enum):
    LONG_RANGE = 0
    STANDARD = 1
    HIGH_PRECISION = 2

# =========================
# Détecteur
# =========================
class ProfessionalFaceDetector:
    def __init__(self):
        # --- Haarcascade
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(haar_path)

        # --- DNN (fallback si dispo localement)
        proto = cv2.data.haarcascades + "../dnn/deploy.prototxt"
        model = cv2.data.haarcascades + "../dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        if os.path.exists(proto) and os.path.exists(model):
            self.dnn = cv2.dnn.readNetFromCaffe(proto, model)
            logger.info("✅ DNN Face Detector chargé")
        else:
            self.dnn = None
            logger.warning("⚠️ DNN Face Detector introuvable, seul Haarcascade sera utilisé")

        # --- Paramètres physiques (approximations par défaut)
        self.focal_length_mm = 3.6        # focale typique
        self.sensor_width_mm = 3.68       # largeur capteur typique
        self.image_width_px = 1280        # largeur image (px) référence
        self.face_width_mm = 140          # largeur visage (~14 cm)

        # Focale en pixels pour la largeur d'image de référence
        self.focal_length_px = (self.focal_length_mm * self.image_width_px) / self.sensor_width_mm

        # --- Facteur d’échelle sur la distance (1.0 = théorique)
        self.distance_scale = 0.5         # Diviser par 2 par défaut
        self.current_mode = DetectionMode.STANDARD

    # --------- Détection ----------
    def detect_faces(self, frame) -> List[Detection]:
        detections: List[Detection] = []
        h, w = frame.shape[:2]

        # Haarcascade (plus permissif)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(28, 28)
        )

        for (x, y, fw, fh) in faces:
            detections.append(
                Detection(
                    bbox=RelativeBBox(x / w, y / h, fw / w, fh / h),
                    score=0.80
                )
            )

        # Fallback DNN si rien trouvé
        if self.dnn is not None and not detections:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            self.dnn.setInput(blob)
            outs = self.dnn.forward()

            for i in range(outs.shape[2]):
                conf = float(outs[0, 0, i, 2])
                if conf > 0.5:
                    box = outs[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    detections.append(
                        Detection(
                            bbox=RelativeBBox(x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h),
                            score=conf
                        )
                    )
        return detections

    # --------- Distance ----------
    def estimate_distance(self, bbox: RelativeBBox, frame_shape) -> float:
        """Distance = (LargeurRÉELLE_visage * Focale_px) / LargeurPixels_visage * distance_scale + 0.10m"""
        ih, iw, _ = frame_shape
        face_width_px = max(bbox.width * iw, 1.0)
        distance_mm = (self.face_width_mm * self.focal_length_px) / face_width_px
        distance_m = (distance_mm / 1000.0) * self.distance_scale
        distance_m += 0.10  # ✅ Correction demandée : +10 cm
        return float(distance_m)

    def calibrate_with_measure(self, measured_distance_m: float, bbox: RelativeBBox, frame_shape):
        """Ajuste self.distance_scale = (distance_mesurée / distance_estimee_actuelle)."""
        if measured_distance_m <= 0:
            return
        est = self.estimate_distance(bbox, frame_shape)
        if est > 0:
            self.distance_scale *= (measured_distance_m / est)
            logger.info(f"🔧 Calibration: distance_scale = {self.distance_scale:.3f}")

    # --------- Dessin ----------
    def draw(self, frame, det: Detection):
        ih, iw, _ = frame.shape
        x = int(det.bbox.xmin * iw)
        y = int(det.bbox.ymin * ih)
        w = int(det.bbox.width * iw)
        h = int(det.bbox.height * ih)
        dist = self.estimate_distance(det.bbox, frame.shape)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Conf: {det.score:.2f} | Dist: {dist:.2f} m"
        cv2.putText(frame, label, (x, max(10, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

# =========================
# Main
# =========================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error("❌ Impossible d'ouvrir la webcam")
        return

    # Résolution de travail
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    det = ProfessionalFaceDetector()
    logger.info("🚀 Système prêt (distance divisée par 2 + ajout de 10 cm).")
    print("Commandes:  Q=Quitter   C=Calibration (entrer distance réelle en m)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = det.detect_faces(frame)
        for d in detections:
            det.draw(frame, d)

        cv2.imshow("Détection visage Pro (OpenCV) — Dist corrigée x0.5 + 10cm", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and detections:
            # calibration rapide sur le 1er visage
            try:
                val = input("Distance réelle en mètres (ex: 1.80) : ").strip()
                real_m = float(val)
                det.calibrate_with_measure(real_m, detections[0].bbox, frame.shape)
                print(f"OK. Nouveau scale = {det.distance_scale:.3f}")
            except Exception:
                print("Entrée invalide.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
