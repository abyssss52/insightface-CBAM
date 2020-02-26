import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/r100-arcface-ms1m_05822x_all/model,2', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=1, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.41, type=float, help='ver dist threshold')
parser.add_argument('--ori_face', default='/home/night/PycharmProjects/face/insightface-CBAM/test_face_image/ori_face/1.jpg', help='path of the original face image')
parser.add_argument('--mask_face', default='/home/night/PycharmProjects/face/insightface-CBAM/test_face_image/ori_face/2.jpg', help='path of the face image with mask')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread(args.ori_face)
img = model.get_input(img)
f1 = model.get_feature(img)
#print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
img = cv2.imread(args.mask_face)
img = model.get_input(img)
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
