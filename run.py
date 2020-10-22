###########
# IMPORTS #
###########
from efficientnet_pytorch import EfficientNet
from torch import nn
from glob import glob
import pydicom
import numpy as np
import torch 
import cv2
import os.path as osp
import argparse
import io
import sys

parser = argparse.ArgumentParser()
parser.add_argument('dcm',type=str)

args = parser.parse_args()
_nb_classes = 1
_imsize = 224

##################
# DATA FUNCTIONS #
##################
def channels_last_to_first(img):
    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,1,2)
    return img 

def preprocess_input(img): 
    # assume image is RGB 
    img_min = float(np.min(img)) ; img_max = float(np.max(img))
    img = (img - img_min) / (img_max - img_min)
    means = [0.485,0.456,0.406] #ImageNet mean
    stds = [0.229,0.224,0.225] #ImageNet std
    img[0] -= means[0]; img[1] -= means[1]; img[2] -= means[2] 
    img[0] /= stds[0]; img[1] /= stds[1]; img[2] /= stds[2]
    return img

def pad_and_resize_image(img, size):
    """
    Resizes image to new_length x new_length and pads with 0.
    """
    # First, get rid of any black border
    nonzeros = np.nonzero(img)
    x1 = np.min(nonzeros[0]); x2 = np.max(nonzeros[0])
    y1 = np.min(nonzeros[1]); y2 = np.max(nonzeros[1])
    img = img[x1:x2, y1:y2, ...]
    pad_x  = img.shape[0] < img.shape[1]
    pad_y  = img.shape[1] < img.shape[0]
    no_pad = img.shape[0] == img.shape[1]
    if no_pad: return cv2.resize(img, (size,size))
    grayscale = len(img.shape) == 2
    square_size = np.max(img.shape[:2])
    x, y = img.shape[:2]
    if pad_x:
        x_diff = int((img.shape[1] - img.shape[0]) / 2)
        y_diff = 0
    elif pad_y:
        x_diff = 0
        y_diff = int((img.shape[0] - img.shape[1]) / 2)
    if grayscale:
        img = np.expand_dims(img, axis=-1)
    pad_list = ((x_diff, square_size-x-x_diff), (y_diff, square_size-y-y_diff), (0,0))
    img = np.pad(img, pad_list, 'constant', constant_values=0)
    assert img.shape[0] == img.shape[1]
    img = cv2.resize(img, (size, size))
    assert size == img.shape[0] == img.shape[1]
    return img

def get_image_from_dicom(dicom_file):
    dcm = pydicom.read_file(dicom_file)
    array = dcm.pixel_array
    try:
        array *= int(dcm.RescaleSlope)
        array += int(dcm.ResscaleIntercept)
    except:
        pass
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        array = np.invert(array.astype('uint16'))
    array = array.astype('float32')
    array -= np.min(array)
    array = array / np.max(array)
    array *= 255.
    return array.astype('uint8')

def new_model(checkpoint):
    device = torch.device('cpu')
    model = EfficientNet.from_pretrained('efficientnet-b0').to(device)
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, _nb_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()  

##########
# SCRIPT #
##########   
try:
    im = get_image_from_dicom(args.dcm)
except:
    print('Could not load [{}] ...'.format(osp.split(args.dcm)[1]))
sys.stdout = io.StringIO()
sys.stdout = sys.__stdout__
im = pad_and_resize_image(im, 224)
cv2.imwrite(args.dcm.replace('.dcm','.png').replace('dicom','png'), im)
img = cv2.imread(args.dcm.replace('.dcm','.png').replace('dicom','png'))
models = glob(osp.join('model_weights','*'))
img = preprocess_input(img)
img = channels_last_to_first(img)
img = torch.from_numpy(img).type('torch.FloatTensor').unsqueeze(0)
outputs = []
for cp in models:
    model = new_model(cp)
    with torch.no_grad():
        outputs.append(torch.sigmoid(model(img)))

sys.stdout = sys.__stdout__
print(f'Model output for file: {osp.split(args.dcm)[1]}')
if np.mean(outputs) > 0.5:
    print(f'\tCOVID-19 pneumonia: DETECTED')
else:
    print(f'\tCOVID-19 pneumonia: NOT DETECTED')



