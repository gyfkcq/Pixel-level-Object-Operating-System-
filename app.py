from flask import Flask, request, jsonify, send_from_directory
from grasp_vis import infer_grasp
import os
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
import json
import numpy as np

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用 CORS

UPLOAD_FOLDER = '/tmp/grasp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/get_grasp_data')
def get_grasp_data():
    score_threshold = float(request.args.get('threshold', 0.5))
    
    # 使用默认参数调用infer_grasp
    result = infer_grasp(
        model_name='GraspNet',
        log_dir='eyeglasses',
        npoint=2048,
        category='eyeglasses',
        input_h5_path='/16T/guoyuefan/grasp_data/eyeglasses/001402.h5',
        score_threshold=score_threshold
    )
    
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logger.debug(f"Received request data: {data}")

        model_name = data.get('model', 'GraspNet')
        log_dir = data.get('log_dir', 'eyeglasses')
        npoint = data.get('npoint', 2048)
        category = data.get('category', 'eyeglasses')
        input_h5_path = data.get('input_h5_path')
        score_threshold = float(data.get('score_threshold', 0.5))

        # 自动拼接权重文件路径
        if model_name == 'DGCNN':
            pth_path = f'/home/guoyuefan/pth/DGCNN_{category}.pth'
        else:
            pth_path = f'/home/guoyuefan/pth/pointnet_{category}.pth'

        logger.debug(f"Auto-selected pth_path: {pth_path}")

        if not input_h5_path:
            return jsonify({'error': 'input_h5_path is required'}), 400

        if not os.path.exists(input_h5_path):
            return jsonify({'error': f'input_h5_path not found: {input_h5_path}'}), 400
        if not os.path.exists(pth_path):
            return jsonify({'error': f'pth_path not found: {pth_path}'}), 400

        result = infer_grasp(
            model_name=model_name,
            log_dir=log_dir,
            npoint=npoint,
            category=category,
            input_h5_path=input_h5_path,
            pth_path=pth_path,
            score_threshold=score_threshold
        )
        logger.debug("Successfully generated grasp predictions")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.endswith('.h5'):
        return jsonify({'error': 'Only .h5 files are allowed'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    return jsonify({'path': save_path})

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=8000, debug=True)
