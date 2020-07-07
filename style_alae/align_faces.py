# This is an altered version of the code at https://github.com/podgorskiy/ALAE/blob/master/align_faces.py
#
# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
import dlib
from PIL import Image
import PIL
import scipy
import scipy.ndimage

# lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
# 69	111	108	111	88	136	72	152	105	152
# 44	51	83	51	63	76	47	92	80	92


predictor_path = 'style_alae/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def align(img, parts, output_size=64, transform_size=128, enable_padding=True):
    # Parse landmarks.
    lm = np.array(parts)
    lm_chin          = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)


    x *= (np.hypot(*eye_to_eye) * 1.6410 + np.hypot(*eye_to_mouth) * 1.560) / 2.0

    y = np.flipud(x) * [-1, 1]

    c = eye_avg + eye_to_mouth * 0.317
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

    qsize = np.hypot(*x) * 2

    img = Image.fromarray(img)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    return img


def align_and_save(img, parts, dst_dir='cropped_imgs', output_size=1024, transform_size=4096, item_idx=0, enable_padding=True):
    align(img, parts,  output_size=1024, transform_size=4096, enable_padding=True)

    # Save aligned image.
    dst_subdir = dst_dir
    os.makedirs(dst_subdir, exist_ok=True)
    img.save(os.path.join(dst_subdir, '%06d.png' % item_idx))


def align_from_PIL(img, output_size=64, transform_size=128, enable_padding=True):
    img = np.asarray(img)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    dets = detector(img, 0)
    if len(dets) == 0:
        return None
    d = dets[0]
    shape = predictor(img, d)
    parts = shape.parts()
    parts = [[part.x, part.y] for part in parts]

    out = align(img, parts, output_size, transform_size, enable_padding)
    return out

if __name__ == "__main__":
    item_idx = 0

    use_1024 = False
    input_dir = "/run/media/gerben/LinuxData/data/ffhq_thumbnails/thumbnails128x128/"
    output_dir = "/run/media/gerben/LinuxData/data/ffhq_thumbnails/aligned64/"

    for filename in os.listdir(input_dir):
        img = np.asarray(Image.open(os.path.join(input_dir, filename)))
        if img.shape[2] == 4:
            img = img[:, :, :3]

        dets = detector(img, 0)
        print("Number of faces detected: {}".format(len(dets)))

        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

            shape = predictor(img, d)

            parts = shape.parts()

            parts = [[part.x, part.y] for part in parts]

            if use_1024:
                align(img, parts, dst_dir=output_dir, output_size=1024, transform_size=4098, item_idx=item_idx)
            else:
                align(img, parts, dst_dir=output_dir, output_size=64, transform_size=128, item_idx=item_idx)

            item_idx += 1