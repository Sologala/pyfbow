import os
import sys
# if os.path.dirname(os.path.realpath(__file__)) == os.getcwd():
#    sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cv2
# import data
import pyfbow as bow
import time
import glob
import argparse
import numpy as np

k = 10
L = 6
nthreads = 1
maxIters = 0
verbose = True


def get_detector_config_string(detector):
    '''
    Get unique config string for detector config
    '''
    get_func_names = sorted(
        [att for att in detector.__dir__() if att.startswith('get')])
    get_vals = [getattr(detector, func_name)() for func_name in get_func_names]
    get_vals_str = [str(val) if type(
        val) != float else "{:.2f}".format(val) for val in get_vals]
    return '_'.join(get_vals_str)


parser = argparse.ArgumentParser(
    description='This script creates a vocabulary from a folder')
# go_pro_icebergs_config.yaml
parser.add_argument('-i', '--image', help='location of folder with images')
parser.add_argument('-v', '--voc', help='location of folder with images')

args = parser.parse_args()

imgpath = args.image

print("Image path :", imgpath)

voc = bow.Vocabulary(k, L, nthreads, maxIters, verbose)
voc.readFromFile(args.voc)
detector = cv2.ORB_create(nfeatures=2000)

img = cv2.imread(imgpath, cv2.IMREAD_COLOR)

if img is None:
    print("Couldn't read image: ", image_name)
    pass

gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp, des = detector.detectAndCompute(gr, mask=None)
print("Detected {} features".format(len(des)))
# dess = des[0].transpose()
# print(dess.shape)
out = voc.transform(des)

print(des.shape, len(out.keys()))
