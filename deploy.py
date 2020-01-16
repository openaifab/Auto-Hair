import torch
import net
import cv2
import os
from torchvision import transforms
import numpy as np

def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad

# inference once for image, return numpy
def inference_once(model, scale_img, scale_trimap, aligned=True):
    size_h = 320
    size_w = 320
    cuda = False
    stage = 1
    if aligned:
        assert(scale_img.shape[0] == size_h)
        assert(scale_img.shape[1] == size_w)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)
    # first, 0-255 to 0-1
    # second, x-mean/std and HWC to CHW
    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    scale_grad = compute_gradient(scale_img)
    #tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_trimap = torch.from_numpy(scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])
    tensor_grad = torch.from_numpy(scale_grad.astype(np.float32)[np.newaxis, np.newaxis, :, :])

    if cuda:
        tensor_img = tensor_img.cuda()
        tensor_trimap = tensor_trimap.cuda()
        tensor_grad = tensor_grad.cuda()
    #print('Img Shape:{} Trimap Shape:{}'.format(img.shape, trimap.shape))

    input_t = torch.cat((tensor_img, tensor_trimap / 255.), 1)

    # forward
    if stage <= 1:
        # stage 1
        pred_mattes, _ = model(input_t)
    else:
        # stage 2, 3
        _, pred_mattes = model(input_t)
    pred_mattes = pred_mattes.data
    if cuda:
        pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]
    return pred_mattes

# forward a whole image
def inference_img_whole(model, img, trimap):
    h, w, c = img.shape
    max_size = 1600
    new_h = min(max_size, h - (h % 32))
    new_w = min(max_size, w - (w % 32))

    # resize for network input, to Tensor
    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_once(model, scale_img, scale_trimap, aligned=False)

    # resize to origin size
    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(origin_pred_mattes.shape == trimap.shape)
    return origin_pred_mattes
