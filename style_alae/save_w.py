"""
    This is a simple script that can save a z.npy which can be loaded into the latent space explorer
"""
import argparse
import math

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image
import style_alae.align_faces as af
import dlib

parser = argparse.ArgumentParser(description="Image to latent vector converter.")
parser.add_argument("--enc", action="store", type=str, help="Path to Gz/Encoder model")
parser.add_argument("--img", action="store", type=str, help="Path to input image")
parser.add_argument("--img2", action="store", type=str, default=None, help="Path to second input image (when morphing)")
parser.add_argument("--phase", action="store", type=float, help="Model phase", default=4.0)

parser.add_argument("--fix_contrast", action="store_true", default=False,
                    help="If true, makes sure that the colors in the image span from 0-255")
parser.add_argument("--ALAE_align", action="store_true", default=False,
                    help="Uses ALAE face alignment")
parser.add_argument("--ALAE_align_large", action="store_true", default=False,
                    help="Uses ALAE face alignment for larger images")
args = parser.parse_args()

resolution = 4 * (2**math.ceil(args.phase))
fname_enc = args.enc

morphing = args.img2 is not None

Gz = torch.load(fname_enc, map_location=torch.device('cpu'))
Gz.eval()

predictor_path = "style_alae/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

w_size = Gz.w_size
real_resolution = resolution


def load_process_img(fname):
    global args

    frame = Image.open(fname)
    if not (args.ALAE_align or args.ALAE_align_large):
        frame = np.array(frame)

        dets = detector(frame, 1)

        num_faces = len(dets)
        if num_faces == 0:
            print("No faces found!")
            exit()

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(frame, detection))

        crop_region_size = 0
        frame = dlib.get_face_chip(frame, faces[0], size=(64 + crop_region_size * 2))

        if crop_region_size != 0:
            frame = frame[crop_region_size:-crop_region_size, crop_region_size:-crop_region_size]
    else:
        frame = np.array(af.align_from_PIL(frame, large=args.ALAE_align_large, output_size=resolution, transform_size=resolution))


    if args.fix_contrast:
        frame = frame.astype(np.float32)
        imin, imax = np.min(frame), np.max(frame)
        frame -= imin
        frame *= 255.0 / (imax - imin)
        frame = frame.astype(np.uint8)

    frame = Image.fromarray(frame)
    input_frame = tvF.scale(frame, real_resolution)
    input_frame = tvF.to_tensor(input_frame).float()

    # input_frame = input_frame.permute(2, 0, 1)
    input_frame = input_frame.unsqueeze(0)
    # input_frame /= 255.0
    return input_frame


x1 = load_process_img(args.img)

z = Gz(x1, phase=args.phase)

if morphing:
    x2 = load_process_img(args.img2)
    z2 = Gz(x2, phase=args.phase)
    # z = 0.5*(z + z2)

    z = 0.5*(z + z2)

z = z.detach().numpy()
np.save("w.npy", z)
