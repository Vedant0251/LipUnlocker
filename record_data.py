import cv2
import dlib
import numpy as np
import os
import time
import urllib.request
import bz2
import glob

# Ensure directories exist
os.makedirs('data/password', exist_ok=True)
os.makedirs('data/random', exist_ok=True)

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(PREDICTOR_PATH):
    print("Downloading shape_predictor_68_face_landmarks.dat...")
    urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", PREDICTOR_PATH + ".bz2")
    with bz2.BZ2File(PREDICTOR_PATH + ".bz2", "rb") as source, open(PREDICTOR_PATH, "wb") as target:
        target.write(source.read())
    os.remove(PREDICTOR_PATH + ".bz2")

print("Loading dlib detector and predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

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

def draw_landmarks_on_image(image, landmarks, color=(0, 255, 0), thickness=2):
    outer = [i for i in range(48, 60)]
    inner = [i for i in range(60, 68)]
    for lips in [outer, inner]:
        for i in range(len(lips)):
            p1 = landmarks.part(lips[i])
            p2 = landmarks.part(lips[(i + 1) % len(lips)])
            cv2.line(image, (p1.x, p1.y), (p2.x, p2.y), color, thickness)
    return image

def record_sample(label, time_limit_seconds=30): # Shortened to 30s to make recording convenient but dense!
    existing_files = glob.glob(f'data/{label}/*.npy')
    sample_num = len(existing_files) + 1
    file_path = f'data/{label}/{sample_num}.npy'
    
    print(f"\\n---> Get ready! Recording FULL 40-POINT MULTI-DIMENSIONAL {label.upper()} sample #{sample_num} in 2 seconds...")
    cv2.waitKey(2000)
    
    frames_data = []
    print(f"Recording strictly for {time_limit_seconds} seconds...")
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit_seconds:
        ret, img = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        features = np.zeros(40).tolist()
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            features = get_dlib_lip_features(landmarks)
            img = draw_landmarks_on_image(img, landmarks, color=(0, 0, 255))
            
        frames_data.append(features)
        
        time_left = max(0, int(time_limit_seconds - (time.time() - start_time)))
        cv2.putText(img, f"RECORDING {label.upper()}: {time_left}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Live Camera', img)
        cv2.waitKey(1)
        
    np.save(file_path, np.array(frames_data))
    print(f"Saved 40-dimensional recording to [{file_path}]")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

print("\\n" + "="*50)
print("ADVANCED 40-POINT MULTI-DIMENSIONAL LIVE CAMERA READY")
print("Press 'P' to record a PASSWORD sample (30s burst)")
print("Press 'R' to record a RANDOM sample (30s burst)")
print("Press 'Q' to quit")
print("="*50)

while True:
    ret, img = cap.read()
    if not ret: break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        img = draw_landmarks_on_image(img, landmarks, color=(0, 255, 0))
        
    cv2.putText(img, "Press P (Password) | R (Random) | Q (Quit)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Live Camera', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('p'):
        record_sample('password')
    elif key == ord('r'):
        record_sample('random')

cap.release()
cv2.destroyAllWindows()
