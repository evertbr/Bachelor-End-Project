import pickle
from copy import deepcopy
import math
import numpy as np

def openPickle(filename : str) -> dict:
    """Open Pickle file and store it in a dictionary"""
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)

    return dictionary
#
OP_keypoints = openPickle('OPSHN.pickle')
AP_keypoints = openPickle('APSH2.pickle')
annotations = openPickle('AnnsSH.pickle')

def removeEvenFrames(keypointsdict : dict) -> dict:
    """"We only annotated every 2 frames. Thus, we must remove some frames from the OpenPose/AlphaPose estimates."""
    duplicate = deepcopy(keypointsdict)
    for seq in duplicate.keys():
        for framenr, frame in enumerate(duplicate[seq]):
            if framenr % 2 != 0:
                del keypointsdict[seq][frame]

    return keypointsdict
#
OP_keypoints = removeEvenFrames(OP_keypoints)
AP_keypoints = removeEvenFrames(AP_keypoints)


def KS(GT : list, pred : list, ktype : int) -> float:
    """Determines the keypoint similarity for one keypoint instance of type ktype
    ktype: keypoint type. 0 for shoulder, 1 for hip
    GT: ground truth coordinates of keypoint (x, y, occluded)
    pred: predicted coordinates of keypoint (x, y, confidence)"""
    s = 0.53  # Heuristic scale factor. Might change later
    dist = math.dist(GT[:1], pred[:1])
    k_list = [0.079, 0.107]  # COCO per-joint constants [shoulder, hip]
    k = k_list[ktype]
    ks = math.exp(-(dist**2)/(2*s**2*k**2))

    return ks

def OKS(GT_list, pred_list) :
    ks_arr = np.empty(len(pred_list))
    occluded_arr = np.empty(len(pred_list))
    for i, GT in enumerate(GT_list):
        if GT_list.index(GT) <= 1:  # Assign joint type
            ktype = 0
        else:
            ktype = 1
        ks = KS(GT, pred_list[i], ktype)
        ks_arr[i] = ks
        occluded_arr[i] = GT[2] > 1 # Indicates whether the joint is occluded

    oks = 0
    for j in range(len(ks_arr)):
        oks += ks_arr[i] * occluded_arr[j]  # Numerator

    oks = oks/sum(occluded_arr)

    return oks

### The ground truth annotation format is not the same as the predictions' format.
### Following code snippet fixes it.

for i in range(len(annotations)):
    seq = annotations[i]
    for picnr in seq.keys():
        for persnr, person in enumerate(seq[picnr]):
            seq[picnr][persnr] = [person[0][j:j+3] for j in range(0, len(person[0]), 3)]


print("OpenPose:", OP_keypoints[0][0][0], "\n\nAlphaPose:", AP_keypoints[0]['0.jpg'][0], "\n\nGround Truth:", annotations[0][1][0])

luukdjong = OKS(annotations[0][1][0], OP_keypoints[0][0][0])


