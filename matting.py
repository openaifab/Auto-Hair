import torch
import net
import cv2
import os
import numpy as np
from deploy import inference_img_whole
from PIL import Image

def loading_model():
    # init model
    resume = "stage1_sad_54.4.pth"
    model = net.VGG16(stage = 1)
    ckpt = torch.load(resume, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im

model = loading_model()
print("matting model loading")

def matting_result(pic_input, tri_input):
    model_input = model
    original_im = np.array(pic_input)[:, :, :3]
    trimap_im = np.array(tri_input)    
    if len(trimap_im.shape)>2:
        trimap_im = trimap_im[:, :, 0]
    with torch.no_grad():
        alpha = inference_img_whole(model_input, original_im, trimap_im)
    
    alpha[trimap_im == 0] = 0.0
    alpha[trimap_im == 255] = 1.0
    h, w = original_im.shape[:2]
    new_bg = np.array(np.full((h,w,3), 255), dtype='uint8')
    im = composite4(original_im, new_bg, alpha, w, h)
    im = Image.fromarray(im)
    return im
