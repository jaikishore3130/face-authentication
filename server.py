from flask import Flask, request, jsonify
import requests 
import base64
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# Firebase initialization
cred = credentials.Certificate("C:\\Users\\dell\\loksabha-firestore-upload\\serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# AES key and IV
key = b'28212821282128212821282128212821'
iv = b'3031303130313031'

# InsightFace setup
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

# Decrypt face image
def decrypt_image(encrypted_data):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(encrypted_data), AES.block_size)
    return decrypted

# Get embedding
def get_face_embedding(image):
    faces = face_app.get(image)
    if faces:
        return faces[0].normed_embedding
    return None

import requests  # ← Don't forget this if not already imported

@app.route('/verify-face', methods=['POST'])
def verify_face():
    aadhaar = request.json.get('aadhaar')
    live_image_b64 = request.json.get('live_image')

    if not aadhaar or not live_image_b64:
        return jsonify({'error': 'Aadhaar number or live image missing'}), 400

    # Decode the live image
    live_image_bytes = base64.b64decode(live_image_b64)
    nparr = np.frombuffer(live_image_bytes, np.uint8)
    live_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get stored encrypted face image URL from Firestore
    doc = db.collection("voters").document(aadhaar).get()
    if not doc.exists:
        return jsonify({'error': 'Aadhaar not found'}), 404

    face_url = doc.to_dict().get("face")
    if not face_url:
        return jsonify({'error': 'Face data missing'}), 404

    response = requests.get(face_url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download face image'}), 500

    decrypted_bytes = decrypt_image(response.content)

    # Convert decrypted image to OpenCV format
    saved_image = cv2.imdecode(np.frombuffer(decrypted_bytes, np.uint8), cv2.IMREAD_COLOR)
    saved_embedding = get_face_embedding(saved_image)
    if saved_embedding is None:
        return jsonify({'error': 'No face in saved image'}), 500

    live_embedding = get_face_embedding(live_image)
    if live_embedding is None:
        return jsonify({'error': 'No face detected in live image'}), 500

    similarity = 1 - cosine(live_embedding, saved_embedding)

    # Encode the decrypted image to base64 to send back to Flutter
    _, buffer = cv2.imencode('.jpg', saved_image)
    encrypted_face_b64 = base64.b64encode(buffer).decode('utf-8')

    if similarity > 0.4:
        return jsonify({
            'status': 'Face Verified',
            'similarity': float(similarity),
            'encrypted_image': encrypted_face_b64  # 👈 send base64 image
        }), 200
    else:
        return jsonify({
            'status': 'Face Not Verified',
            'similarity': float(similarity),
            'encrypted_image': encrypted_face_b64  # 👈 send base64 image anyway
        }), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
