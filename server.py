from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

app = Flask(__name__)

# InsightFace setup
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

# Get face embedding
def get_face_embedding(image):
    faces = face_app.get(image)
    if faces:
        return faces[0].normed_embedding
    return None

@app.route('/verify-face', methods=['POST'])
def verify_face():
    try:
        saved_face_b64 = request.json.get('saved_face')
        live_face_b64 = request.json.get('live_face')

        if not saved_face_b64 or not live_face_b64:
            return jsonify({'error': 'Both saved_face and live_face are required'}), 400

        # Decode base64 images
        saved_face_bytes = base64.b64decode(saved_face_b64)
        live_face_bytes = base64.b64decode(live_face_b64)

        saved_image = cv2.imdecode(np.frombuffer(saved_face_bytes, np.uint8), cv2.IMREAD_COLOR)
        live_image = cv2.imdecode(np.frombuffer(live_face_bytes, np.uint8), cv2.IMREAD_COLOR)

        saved_embedding = get_face_embedding(saved_image)
        live_embedding = get_face_embedding(live_image)

        if saved_embedding is None or live_embedding is None:
            return jsonify({'error': 'Face not detected in one or both images'}), 400

        similarity = 1 - cosine(live_embedding, saved_embedding)

        if similarity > 0.8:
            return jsonify({'status': 'Face Verified', 'similarity': float(similarity)}), 200
        else:
            return jsonify({'status': 'Face Not Verified', 'similarity': float(similarity)}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
