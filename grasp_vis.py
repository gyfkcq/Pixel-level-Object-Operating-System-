# -*- coding: utf-8 -*-
# grasp_vis.py

import numpy as np
import h5py as h5
import argparse
from scipy.spatial.transform import Rotation as R
import importlib
import torch
from pathlib import Path
import datetime
import os
import sys
import json
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    if m.__class__.__name__.find('ReLU') != -1:
        m.inplace = True

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def infer_grasp(model_name, log_dir, npoint, category, input_h5_path, pth_path=None, score_threshold=0.5):
    '''Load model and return grasp prediction results'''
    exp_dir = Path('./log/log') / log_dir
    checkpoints_dir = exp_dir / 'checkpoints'
    MODEL = importlib.import_module(model_name)
    model = MODEL.get_model().cuda()
    model.apply(inplace_relu)

    # 加这段逻辑：支持自定义 pth 路径
    if pth_path is None:
        pth_path = str(checkpoints_dir / 'best_model.pth')
    else:
        pth_path = os.path.abspath(pth_path)

    checkpoint = torch.load(pth_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad(), h5.File(input_h5_path, 'r') as f:
        input_points = f['camcs_per_point'][:]
        input_points[:, 0:3] = pc_normalize(input_points[:, 0:3])
        choice = np.random.choice(len(input_points), npoint, replace=True)
        input_points = input_points[choice, :]

        input_tensor = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0).cuda()
        pred = model(input_tensor)

        pred_seg = torch.argmax(pred['grasp_seg_per_point'], dim=-1)
        pred_quats = pred['quats_per_point']

        # 先创建张量，再移动到GPU
        quats_score = torch.tensor(f['quats_per_point'][:, -1])[choice].unsqueeze(0)
        quats_score = quats_score.cuda()  # 移动到GPU
        
        # 分别处理分割和分数阈值
        seg_mask = (pred_seg == 1)
        score_mask = (quats_score >= score_threshold)  # 使用传入的阈值
        grasp_mask = seg_mask & score_mask

        grasp_points = input_tensor[0][grasp_mask[0]].cpu().numpy()
        grasp_quats = pred_quats[0][grasp_mask[0]].cpu().numpy()
        grasp_scores = quats_score[0][grasp_mask[0]].cpu().numpy()

        # 对抓取点按分数排序
        sorted_indices = np.argsort(grasp_scores)[::-1]
        top_indices = sorted_indices[:50]  # 只返回前50个最高分数的抓取点

        # 返回结果
        return {
            'grasp_mask_points': grasp_points.tolist(),
            'grasp_mask_scores': grasp_scores.tolist(),
            'grasp_mask_quats': grasp_quats.tolist(),
            'input_points': input_points[:, :3].tolist(),
            'score_threshold': score_threshold,
            'top_grasp_points': grasp_points[top_indices].tolist(),  # 添加前50个最高分数的抓取点
            'top_quats': grasp_quats[top_indices].tolist(),  # 添加对应的四元数
            'top_scores': grasp_scores[top_indices].tolist()  # 添加对应的分数
        }

# CLI test
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='GraspNet', help='model name')
    parser.add_argument('--log_dir', type=str, default='eyeglasses', help='log path')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--category', type=str, default='eyeglasses', help='category name')
    args = parser.parse_args()

    input_h5_path = f'/16T/guoyuefan/grasp_data/{args.category}/001402.h5'
    result = infer_grasp(args.model, args.log_dir, args.npoint, args.category, input_h5_path)

    # Force stdout encoding to UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))

