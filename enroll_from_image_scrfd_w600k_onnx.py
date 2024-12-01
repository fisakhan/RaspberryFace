#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Asif Khan
# June 13, 2024

'''
ENROL A PERSON (NEW IDENTITY) FROM FACE IMAGE.
NAME OF THE IMAGE: FIRST_MIDDLE_LAST.PNG
REQUIREMENTS: ONLY ONE FACE IN THE IMAGE
FRAMEWORK: tflite, both for detection and feature vector generation
         : can be onnx, but commented (tflite if faster)
SCRFD: float16 Quantized
w600k_r50: float32 Quantized

TODO: what if this id is already enrolled???
    what if there are multiple faces in the image???
    what if name is different but person is already enrolled???
    what if name is same but person is different???
'''
import cv2
import sys
import copy
import time
import argparse
import numpy as np
import pandas as pd
import tkinter as tk

#import tensorflow as tf
import tflite_runtime.interpreter as tflite
from scrfd.scrfd_tflite import SCRFD# face detector

# SETTINGS FOR FACE RECOGNITION MODEL
vectorizer = 'BFL'# use https://github.com/deepinsight/insightface
dir_models = './models'
if vectorizer == 'BFL':
    #from vectorizer.onnx_insightface import norm_crop as alignment
    from onnx_insightface import norm_crop as alignment
    from onnx_insightface import VectorizerModel
    vectorizer_model = VectorizerModel(dir_models+'/'+"onnx/w600k_r50.onnx")

# SETTINGS FOR FACE DETECTION MODEL
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--model", type=str, default='tflite/scrfd_500m_bnkps_480x640_float16_quant.tflite')
    parser.add_argument("--input_size", type=str, default='480,640')
    parser.add_argument("--score_th", type=float, default=0.5)
    parser.add_argument("--nms_th", type=float, default=0.4)
    args = parser.parse_args()
    return args
args = get_args()

# Load SCRFD face detector model
print("Loading face detector...")
detector = SCRFD(model_file='./models/'+args.model, nms_thresh=args.nms_th)
detector.prepare(-1)

# determine resolution of input image
input_size = [int(i) for i in args.input_size.split(',')]

# Load BFL face recognition model 
print("Loading face recognizer...")
interpreter = tflite.Interpreter(model_path="./models/tflite/w600k_r50_float32.tflite", num_threads=3)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# LOAD ENROLLED IDENTITIES
print("Loading database...")
df_names = pd.read_pickle("./id_names.pkl")
enrolled_fvectors = np.load("./scrfd_500m_bnkps_480x640_model_float16_quant_w600k_r50_float32_features.npy")

# READ INPUT IMAGE
image_BGR = cv2.imread(args.path)# BGR
if image_BGR is None:
    print("Cannot read the image at this path.")
    sys.exit(1)

#image_RGB = image_BGR[...,::-1]# convert to RGB image
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
# cv2.imshow("img_BGR", image_BGR)# make sure to process RGB
# cv2.imshow("img_RGB", image_RGB)# make sure to process RGB


# Infer Face Detection
bboxes, keypoints = detector.detect(
    image_RGB,
    args.score_th,
    input_size=(input_size[1], input_size[0]),
)

if (len(bboxes) > 0):# if a face is detected
    
    # which face to process?
    indx = 0# take only first face
    
    # Generate RGB crop (112x112) as per insightface
    landmarks = keypoints[indx]# take only first face
    recog_crop, _ = alignment(image_RGB, landmarks)
    #cv2.imshow("recog_crop", recog_crop)# make sure to process RGB
    
    # Generate feature vector as per insightface (w600k_r50.onnx)
    
    # ONNX
    #feature_vector_onnx = vectorizer_model.forward([recog_crop,])[0]
    
    # TFLITE    
    recog_crop = (recog_crop-127.5)/127.5
        
    recog_crop = np.array(recog_crop, dtype=np.float32)
    input_data = input_data = np.expand_dims(recog_crop, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()   
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data/np.linalg.norm(output_data)
    feature_vector_tflite = output_data[0]
    
    # write identity name and feature vector to .pkl and .npy files
    fv_list = []
    #fv_list.append(feature_vector_onnx)
    fv_list.append(feature_vector_tflite)
    
    temp_names = args.path.split('/')[-1].split('.')[0].split('_')
    if len(temp_names) == 2:
        names = list([temp_names[0], "", temp_names[1], -1])
    elif len(temp_names) == 3:
        names = list([temp_names[0], temp_names[1], temp_names[2], -1])
    elif len(temp_names) >= 4:
        extra_names = "_".join(temp_names[2:-1])
        names = list([temp_names[0], temp_names[1], extra_names, -1])
    else:
        print("Name is either too short or too long.")
        sys.exit(1)
    #print(names)
    #print(df_names.head(40))
    
    df_names.loc[len(df_names.index)] = names
    df_names.to_pickle("./id_names.pkl")
    #fvs = np.vstack([enrolled_fvectors, feature_vector_onnx])# ONNX
    fvs = np.vstack([enrolled_fvectors, feature_vector_tflite])# TFLITE
    np.save("./scrfd_500m_bnkps_480x640_model_float16_quant_w600k_r50_float32_features.npy", fvs)
    
    print("{} {} successfully enrolled.".format(names[0], names[1]))
    
else:
    print("No face detected.")

cv2.waitKey(0) 
cv2.destroyAllWindows()
