from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import os
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import subprocess
import shutil
import time

app = Flask(__name__)
CORS(app)

# ===== FACE DATA CONFIG =====
KNOWN_FACES_DIR = "known_faces"
UPLOADS_DIR = "uploads"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

FACE_MATCH_THRESHOLD = 0.5
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

known_encodings = []
known_names = []


# ===== HELPER FUNCTIONS =====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_known_faces():
    """Load all known faces from the known_faces directory"""
    known_encodings.clear()
    known_names.clear()

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[WARN] {KNOWN_FACES_DIR} directory not found")
        return

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(person_dir, filename)
                try:
                    img = face_recognition.load_image_file(path)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        known_encodings.append(encs[0])
                        known_names.append(name)
                        print(f"[INFO] Loaded face: {name} from {filename}")
                except Exception as e:
                    print(f"[WARN] Could not load {path}: {e}")

    print(f"[INFO] Total loaded: {len(known_names)} known faces")


# Load faces on startup
load_known_faces()


# ===== ROUTES =====

@app.route('/')
def home():
    """Landing page"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition Server</title>
        <style>
            body {{ font-family: Arial; text-align: center; padding: 50px; background: linear-gradient(135deg,#667eea,#764ba2); min-height: 100vh; margin:0; }}
            .container {{ background:white; max-width:600px; margin:auto; padding:40px; border-radius:15px; box-shadow:0 10px 40px rgba(0,0,0,0.3); }}
            h1 {{ color:#333; }}
            .btn {{ background:#667eea; color:white; padding:15px 30px; text-decoration:none; border-radius:8px; display:inline-block; margin-top:20px; }}
            .btn:hover {{ background:#5568d3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔐 Face Recognition Server</h1>
            <p>✅ Server Running</p>
            <p><strong>Loaded Faces:</strong> {len(known_names)}</p>
            <a href="/app" class="btn">🚀 Open Face Recognition App</a>
        </div>
    </body>
    </html>
    """


@app.route('/app')
def app_page():
    """Serve the main HTML interface (index.html)"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(current_dir, 'index.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>❌ index.html not found!</h1><p>Place index.html in the same folder as app.py</p>", 404


@app.route('/add_face', methods=['POST'])
def add_face():
    try:
        if 'image' not in request.files or 'name' not in request.form:
            return jsonify({"status": "error", "message": "Image or name missing"}), 400

        file = request.files['image']
        name = request.form['name'].strip()

        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Invalid file type"}), 400

        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(f"{name}_{timestamp}.jpg")
        filepath = os.path.join(person_dir, filename)
        file.save(filepath)

        img = face_recognition.load_image_file(filepath)
        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) != 1:
            os.remove(filepath)
            return jsonify({"status": "error", "message": "Image must contain exactly one face"}), 400

        known_encodings.append(face_encodings[0])
        known_names.append(name)

        print(f"[INFO] Added face for {name}")
        return jsonify({"status": "success", "message": f"Added {name}", "faces_count": len(known_names)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image"}), 400

        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Invalid file"}), 400

        temp_path = os.path.join(UPLOADS_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        file.save(temp_path)

        img = face_recognition.load_image_file(temp_path)
        face_encodings = face_recognition.face_encodings(img)
        os.remove(temp_path)

        if len(face_encodings) == 0:
            return jsonify({"status": "no_face", "message": "No face detected"})

        encoding = face_encodings[0]
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        if not len(face_distances):
            return jsonify({"status": "error", "message": "No known faces loaded"})

        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance < FACE_MATCH_THRESHOLD:
            name = known_names[best_match_index]
            confidence = round((1 - best_distance) * 100, 2)
            print(f"[GRANTED] {name} ({confidence}%)")
            return jsonify({"status": "recognized", "name": name, "confidence": confidence})
        else:
            print("[DENIED] Unknown person")
            return jsonify({"status": "unknown", "message": "Face not recognized"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/list_faces', methods=['GET'])
def list_faces():
    face_counts = {}
    for name in known_names:
        face_counts[name] = face_counts.get(name, 0) + 1
    return jsonify({"status": "success", "faces": face_counts, "total": len(known_names)})


@app.route('/delete_face', methods=['POST'])
def delete_face():
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"status": "error", "message": "Name missing"}), 400

        name = data['name'].strip()
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            load_known_faces()
            return jsonify({"status": "success", "message": f"Deleted {name}"})
        else:
            return jsonify({"status": "error", "message": f"No data for {name}"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/delete_all_faces', methods=['POST'])
def delete_all_faces():
    for name in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
    load_known_faces()
    return jsonify({"status": "success", "message": "All faces deleted"})


# ===== OPEN IN STANDALONE WINDOW =====
def open_window():
    time.sleep(2)
    try:
        # Use Edge (or switch to Chrome if preferred)
        browser_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        # browser_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        subprocess.Popen([browser_path, "--app=http://127.0.0.1:5000/app"])
    except Exception as e:
        print(f"[WARN] Could not open browser window: {e}")


# ===== START SERVER =====
if __name__ == '__main__':
    print("=" * 60)
    print("Face Recognition Server Starting...")
    print(f"Known faces directory: {KNOWN_FACES_DIR}")
    print(f"Uploads directory: {UPLOADS_DIR}")
    print(f"Loaded {len(known_names)} known faces")
    print("=" * 60)

    threading.Thread(target=open_window, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)