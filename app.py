from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from multiprocessing import Pool, TimeoutError
import shutil

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_similarity(image1_path, image2_path):
    try:
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0

        img1 = cv2.resize(img1, (150, 150))  # Resize for faster processing
        img2 = cv2.resize(img2, (150, 150))

        score, _ = ssim(img1, img2, full=True)
        return score
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0

def process_image(args):
    reference_path, file_path = args
    similarity = calculate_similarity(reference_path, file_path)
    return file_path, similarity

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Validate files
            if 'reference_image' not in request.files or 'image_folder[]' not in request.files:
                return jsonify({"error": "Missing required files."}), 400

            reference_file = request.files['reference_image']
            files = request.files.getlist('image_folder[]')

            if reference_file and allowed_file(reference_file.filename):
                reference_filename = secure_filename(reference_file.filename)
                reference_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
                reference_file.save(reference_path)

                folder_path = os.path.join(app.config['UPLOAD_FOLDER'], 'folder')
                os.makedirs(folder_path, exist_ok=True)

                for file in files:
                    if file and allowed_file(file.filename):
                        file.save(os.path.join(folder_path, secure_filename(file.filename)))

                file_paths = [
                    os.path.join(root, file)
                    for root, _, files in os.walk(folder_path)
                    for file in files if allowed_file(file)
                ]

                with Pool() as pool:
                    try:
                        results = pool.map_async(
                            process_image, 
                            [(reference_path, file_path) for file_path in file_paths]
                        ).get(timeout=120)  # Timeout set to 120 seconds
                    except TimeoutError:
                        return jsonify({"error": "Image recognition timed out."}), 408

                results.sort(key=lambda x: x[1], reverse=True)

                # Save results for display
                saved_results = []
                for idx, (file_path, similarity) in enumerate(results[:5]):
                    dst_path = os.path.join(app.config['RESULT_FOLDER'], f'result_{idx + 1}.jpg')
                    cv2.imwrite(dst_path, cv2.imread(file_path))
                    saved_results.append({
                        'file_path': dst_path,
                        'similarity': similarity
                    })

                return render_template('results.html', results=saved_results)

            return jsonify({"error": "Invalid reference image."}), 20000
        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({"error": "An error occurred during processing. Please try again."}), 20000

    return render_template('index.html')

@app.route('/results-data', methods=['GET'])
def results_data():
    try:
        # Replace this with actual logic to read and serve result data
        results = [
            {"file_path": "/static/results/result_1.jpg", "similarity": 0.90},
            {"file_path": "/static/results/result_2.jpg", "similarity": 0.85},
            {"file_path": "/static/results/result_3.jpg", "similarity": 0.80},
            {"file_path": "/static/results/result_4.jpg", "similarity": 0.75},
            {"file_path": "/static/results/result_5.jpg", "similarity": 0.72}
        ]
        return jsonify({"results": results})
    except Exception as e:
        print(f"Error loading results data: {e}")
        return jsonify({"error": "Failed to load results data."}), 20000
    
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "Processing...", "progress": "3/10 images completed"})



if __name__ == '__main__':
    app.run(debug=True)
