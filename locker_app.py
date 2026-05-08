import tkinter as tk
from tkinter import messagebox
import cv2
import dlib
import numpy as np
import os
import time
import threading
import tensorflow as tf

# Configuration
STORAGE_DIR = 'locker_storage'
os.makedirs(STORAGE_DIR, exist_ok=True)
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
FRAMES_TO_RECORD = 90   # capture more frames so we always get enough 60-frame windows
WINDOW_SIZE = 60        # must match train_model.py
FEATURES = 40           # must match train_model.py
UNLOCK_THRESHOLD = 0.60 # LSTM confidence >= this → UNLOCKED
MODEL_PATH = 'lip_model.keras'

print("Initializing Biometrics Engine with LSTM Model...")
detector = dlib.get_frontal_face_detector()
if not os.path.exists(PREDICTOR_PATH):
    print(f"Error: {PREDICTOR_PATH} missing.")
    exit(1)
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ── Load LSTM model ──────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"Error: Trained model '{MODEL_PATH}' not found.")
    print("Please run train_model.py first to generate the model!")
    exit(1)
print(f"Loading LSTM model from '{MODEL_PATH}'...")
lstm_model = tf.keras.models.load_model(MODEL_PATH)
print("✔ LSTM model loaded successfully!")

def get_dlib_lip_features(landmarks):
    """
    ULTIMATE INVARIANT EXTRACTION (FIXED):
    Extracts 20 lip landmark coordinates.
    Crucially, we now anchor scale and rotation to the EYES, not the mouth!
    Anchoring to the mouth dynamically destroyed the real stretching variance.
    """
    if landmarks.num_parts < 68: return np.zeros(40).tolist()
    
    # 20 Points of lips (48 through 67)
    pts = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 68)], dtype=np.float32)
    
    # 1. Translation Invariance (Center to Origins)
    center = np.mean(pts, axis=0)
    pts -= center
    
    # 2. Rotation Invariance (Align based on EYES to preserve asymmetric mouth shapes)
    eye_left = np.array([landmarks.part(36).x, landmarks.part(36).y])
    eye_right = np.array([landmarks.part(45).x, landmarks.part(45).y])
    delta = eye_right - eye_left
    angle = np.arctan2(delta[1], delta[0]) 
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    pts = np.dot(pts, R.T)
    
    # 3. Scale Invariance (Divide by static face width (Eye distance), NOT dynamic mouth width!)
    face_width = np.linalg.norm(eye_right - eye_left)
    if face_width < 1.0: face_width = 1.0
    pts /= face_width
    
    return pts.flatten().tolist()

def draw_landmarks(image, landmarks, color=(0, 255, 0), thickness=2):
    outer = [i for i in range(48, 60)]
    inner = [i for i in range(60, 68)]
    for lips in [outer, inner]:
        for i in range(len(lips)):
            p1 = landmarks.part(lips[i])
            p2 = landmarks.part(lips[(i + 1) % len(lips)])
            cv2.line(image, (p1.x, p1.y), (p2.x, p2.y), color, thickness)

def lstm_confidence(sequence):
    """
    Runs the captured lip sequence through the LSTM model using the same
    sliding-window strategy as training (WINDOW_SIZE=60, step=30).
    Returns the MAX confidence score across all windows.
    A score >= UNLOCK_THRESHOLD means the lip movement matches the password.

    sequence: list of frames, each frame is a list of 40 floats.
    """
    data = np.array(sequence, dtype=np.float32)  # shape: (N, 40)
    if len(data) < WINDOW_SIZE:
        # Pad if shorter than one full window
        padded = np.zeros((WINDOW_SIZE, FEATURES), dtype=np.float32)
        padded[:len(data), :] = data
        windows = [padded]
    else:
        step = WINDOW_SIZE // 2
        windows = [
            data[i : i + WINDOW_SIZE]
            for i in range(0, len(data) - WINDOW_SIZE + 1, step)
        ]

    batch = np.array(windows)                        # shape: (num_windows, 60, 40)
    predictions = lstm_model.predict(batch, verbose=0).flatten()  # shape: (num_windows,)
    max_confidence = float(np.max(predictions))
    avg_confidence = float(np.mean(predictions))
    return max_confidence, avg_confidence

class LockerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepLipUnlocker — LSTM Engine")
        self.root.geometry("400x500")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(False, False)
        
        self.header = tk.Label(root, text="DEEP LIP LOCKER", font=("Helvetica", 20, "bold"), bg="#1e1e1e", fg="#00e676")
        self.header.pack(pady=30)
        
        self.status_var = tk.StringVar(value="Status: LOCKED")
        self.status_label = tk.Label(root, textvariable=self.status_var, font=("Helvetica", 24, "bold"), bg="#1e1e1e", fg="#ff1744")
        self.status_label.pack(pady=40)
        
        self.btn_set = tk.Button(root, text="🛡️ Set Master Password", font=("Helvetica", 14), bg="#37474f", fg="white", 
                                 command=self.set_password, width=25, height=2, borderwidth=0, cursor="hand2")
        self.btn_set.pack(pady=10)
        
        self.btn_unlock = tk.Button(root, text="🔓 Attempt Unlock", font=("Helvetica", 14), bg="#0277bd", fg="white", 
                                    command=self.unlock_locker, width=25, height=2, borderwidth=0, cursor="hand2")
        self.btn_unlock.pack(pady=10)
        
        self.btn_lock = tk.Button(root, text="🔒 Lock System", font=("Helvetica", 14), bg="#d32f2f", fg="white", 
                                  command=self.lock_system, width=25, height=2, borderwidth=0, cursor="hand2")
        self.btn_lock.pack(pady=10)
        
        self.info = tk.Label(root, text=f"Engine: LSTM Model  |  Threshold: {UNLOCK_THRESHOLD*100:.0f}%", font=("Helvetica", 9), bg="#1e1e1e", fg="#9e9e9e")
        self.info.pack(side="bottom", pady=20)
        
    def lock_system(self):
        self.status_var.set("Status: LOCKED")
        self.status_label.configure(fg="#ff1744")
        
    def capture_lip_sequence(self, window_name="Biometric Capture"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam.")
            return None
            
        cv2.namedWindow(window_name)
        for i in range(20):
            ret, img = cap.read()
            cv2.putText(img, "GET READY...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(window_name, img)
            cv2.waitKey(100)
            
        frames_data = []
        for i in range(FRAMES_TO_RECORD):
            ret, img = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            features = np.zeros(40).tolist()
            if len(faces) > 0:
                face = faces[0]
                landmarks = predictor(gray, face)
                features = get_dlib_lip_features(landmarks)
                draw_landmarks(img, landmarks, color=(255, 105, 180)) # Pink/Purple
                
            frames_data.append(features)
            
            cv2.putText(img, f"RECORDING: {i+1}/{FRAMES_TO_RECORD}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, img)
            cv2.waitKey(30)
            
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.2) 
        return frames_data
        
    def set_password(self):
        self.root.after(10, lambda: messagebox.showinfo(
            "Setup",
            "The LSTM model authenticates based on its TRAINING data.\n"
            "\n"
            "'Set Master Password' is no longer needed — the model already\n"
            "learned your lip pattern during training.\n"
            "\n"
            "Just press 'Attempt Unlock' to authenticate!"
        ))

    def unlock_locker(self):
        attempt_seq = self.capture_lip_sequence("Attempting Unlock...")
        if not attempt_seq:
            return

        max_conf, avg_conf = lstm_confidence(attempt_seq)
        gate = max_conf >= UNLOCK_THRESHOLD

        print(f"LSTM | Max confidence: {max_conf:.4f} | Avg confidence: {avg_conf:.4f} | "
              f"Threshold: {UNLOCK_THRESHOLD} | Gate: {'PASS ✓' if gate else 'FAIL ✗'}")

        if gate:
            self.status_var.set("Status: UNLOCKED ✓")
            self.status_label.configure(fg="#00e676")
            self.root.after(10, lambda: messagebox.showinfo(
                "Access Granted",
                f"Lip pattern recognised!\n"
                f"LSTM Confidence: {max_conf*100:.1f}%  (threshold ≥ {UNLOCK_THRESHOLD*100:.0f}%)"
            ))
        else:
            self.status_var.set("Status: LOCKED ❌")
            self.status_label.configure(fg="#ff1744")
            self.root.after(10, lambda: messagebox.showerror(
                "Access Denied",
                f"Lip pattern not recognised!\n"
                f"LSTM Confidence: {max_conf*100:.1f}%  (need ≥ {UNLOCK_THRESHOLD*100:.0f}%)"
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = LockerUI(root)
    root.mainloop()

