# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from tqdm import tqdm
import sys

from .configs.config import get_cfg_defaults
from .datasets.creation.util import get_arcface_input, get_center, draw_on
from .utils import util
from .utils.landmark_detector import LandmarksDetector, detectors

sys.path.append("MICA")

def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def process_single(arcface, app, img, image_size=224, draw_bbox=False):
    dst = Path(arcface)
    dst.mkdir(parents=True, exist_ok=True)
    name = "test"
    # img = cv2.imread(image_path)
    bboxes, kpss = app.detect(img)
    if bboxes.shape[0] == 0:
        logger.error(f"[ERROR] Face not detected")
    i = get_center(bboxes, img)
    bbox = bboxes[i, 0:4]
    det_score = bboxes[i, 4]
    kps = None
    if kpss is not None:
        kps = kpss[i]
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    blob, aimg = get_arcface_input(face, img)
    file = str(Path(dst, name))
    np.save(file, blob)

    process = file + ".npy"
    cv2.imwrite(file + ".jpg", face_align.norm_crop(img, landmark=face.kps, image_size=image_size))
    if draw_bbox:
        dimg = draw_on(img, [face])
        cv2.imwrite(file + "_bbox.jpg", dimg)

    return process


def to_batch(path):
    src = path.replace("npy", "jpg")
    if not os.path.exists(src):
        src = path.replace("npy", "png")

    image = imread(src)[:, :, :3]
    image = image / 255.0
    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).cuda()[None]

    arcface = np.load(path)
    arcface = torch.tensor(arcface).cuda()[None]

    return image, arcface

def MICA(pretrained, device):
    cfg = get_cfg_defaults()
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    
    checkpoint = torch.load(pretrained)
    if "arcface" in checkpoint:
        mica.arcface.load_state_dict(checkpoint["arcface"])
    if "flameModel" in checkpoint:
        mica.flameModel.load_state_dict(checkpoint["flameModel"])
        
    mica.eval()

    faces = mica.flameModel.generator.faces_tensor.cpu()

    app = LandmarksDetector(model=detectors.RETINAFACE)

    return mica, faces, app

def main(input_img, arcface, pretrained, device, output_path):
    mica, faces, app = MICA(pretrained, device)
    with torch.no_grad():
        logger.info(f'Processing has started...')
        path = process_single(arcface, app, input_img, draw_bbox=False)
        name = Path(path).stem
        images, arcface = to_batch(path)
        codedict = mica.encode(images, arcface)
        opdict = mica.decode(codedict)
        meshes = opdict['pred_canonical_shape_vertices']
        code = opdict['pred_shape_code']
        lmk = mica.flame.compute_landmarks(meshes)

        mesh = meshes[0]
        landmark_51 = lmk[0, 17:]
        landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

        dst = Path(output_path, name)
        dst.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
        np.save(f'{dst}/identity', code[0].cpu().numpy())
        np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
        np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

        logger.info(f'Processing finished. Results has been saved in {output_path}')

if __name__ == '__main__':
    deterministic(42)
    main(input_img='demo/input/carell.jpg', arcface='demo/arcface', pretrained='data/pretrained/mica.tar', device='cuda:0', output_path='demo/output')