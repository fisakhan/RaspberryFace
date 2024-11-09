import onnxruntime as ort
import numpy as np
import cv2
from skimage import transform as trans
from numpy.linalg import norm as l2norm
#from toolkit.global_parameters import download_model

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, None

input_mean = 127.5
input_std = 127.5

input_size = (112, 112)
output_names = ['683']
input_name = 'input.1'

class VectorizerModel:

    def __init__(self, model_file=None, session=None, intraT=4, interT=1):
    
        # set path to face recognition (feature vector generation) model
        #model_path = download_model(f'{dir_models}/onnx/w600k_r50.onnx')
        #model_path = f'{dir_models}/onnx/w600k_r50.onnx'
        model_path = model_file
        #self.model = ort.InferenceSession((model_path),
        #                                      providers=['CUDAExecutionProvider',])
        #self.model = ort.InferenceSession((model_path),
        #                                      providers=['CPUExecutionProvider',])
        
        # Load model
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intraT# set optimum number of threads
        opts.inter_op_num_threads = interT# set optimum number of threads
        self.model = ort.InferenceSession((model_path), sess_options=opts)

    def forward(self, crops):
        '''
        Inference to generate feature vector
        '''

        blob = cv2.dnn.blobFromImages(crops, 1.0 / input_std, input_size,
                                      (input_mean, input_mean, input_mean), swapRB=True)
        
        net_out = self.model.run(output_names, {input_name: blob})[0].flatten()
        feature_vectors = net_out/l2norm(net_out)
    
        return [feature_vectors,]
