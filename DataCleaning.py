import pickle
from copy import deepcopy
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def openPickle(filename : str) -> dict:
    """Open Pickle file and store it in a dictionary"""
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)

    return dictionary
#
OP_keypoints = openPickle('OPSHN2.pickle')
AP_keypoints = openPickle('APKPFINAL.pickle')
annotations = openPickle('AnnsSH.pickle')

# Fix AP keypoints naming
for seq in AP_keypoints.keys():
    for pic in list(AP_keypoints[seq].keys()):
        picnr = int(pic.split('.')[0])
        AP_keypoints[seq][picnr] = AP_keypoints[seq][pic]
        del AP_keypoints[seq][pic]


def removeEvenFrames(keypointsdict : dict) -> dict:
    """"We only annotated every 2 frames. Thus, we must remove some frames from the OpenPose/AlphaPose estimates."""
    duplicate = deepcopy(keypointsdict)
    for seq in duplicate.keys():
        for framenr, frame in enumerate(duplicate[seq]):
            if framenr % 2 != 0:
                del keypointsdict[seq][frame]
            else:
                if framenr > 0:  # 0/2 = 0
                    # Fix order
                    keypointsdict[seq][int(frame/2)] = keypointsdict[seq][frame]
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


# print("OpenPose:", OP_keypoints[0][0][0], "\n\nAlphaPose:", AP_keypoints[0][0][0], "\n\nGround Truth:", annotations[0][1][0])

# luukdjong = OKS(annotations[0][1][0], OP_keypoints[0][0][0])
# print(luukdjong)

# LSH_matrix = np.arange(len(annotations[0][1])**2)
# LSH_matrix = LSH_matrix.reshape((len(annotations[0][1]), len(annotations[0][1])))

def shiftKeys(keypoints):
    """Due to human error, one frame was not annotated. We remove estimations for this frame and shift the estimation
    dictionary to the left"""
    del keypoints[2][10]
    keys_to_shift = sorted(k for k in keypoints[2].keys() if k > 10)  # Find keys > 10

    for key in keys_to_shift:
        keypoints[2][key - 1] = keypoints[2].pop(key)

    return keypoints

AP_keypoints = shiftKeys(AP_keypoints)
OP_keypoints = shiftKeys(OP_keypoints)

def fillArray(array : np.array, ind, joints, jointnr) -> None:
    """This function helps deal with the fact that the number of estimated joints is not necessarily the same
    as the number of annotated joints"""
    try:
        array[ind] = joints[ind][jointnr]
    except:
        pass

def outersubtract(groundtruths, preds):
    """From an array of joint prediction and ground truths, form a np array used for bipartite matching
    Groundtruths: list with GT location arrays for multiple joints for multiple people
    Estimations: list with prediction locations for multiple joints for multiple people"""

    # We pad the estimation with less joints to get compatible shapes
    maxlen = max(len(groundtruths[0]), len(preds[0]))
    if len(groundtruths[0]) == maxlen:
        arrays_to_pad = preds
        lendiff = maxlen - len(preds[0])
    else:
        arrays_to_pad = groundtruths
        lendiff = maxlen - len(groundtruths[0])


    output = np.zeros((len(groundtruths), maxlen, maxlen, 2))  # Amount of joints, max amount of players, x and y coord

    for jointnr in range(len(groundtruths)):
        arrays_to_pad[jointnr] = np.pad(arrays_to_pad[jointnr], ((lendiff,0),(0,0)), 'constant', constant_values=np.NaN)
        gtcoords = groundtruths[jointnr][:, :2]
        predcoords = preds[jointnr][:, :2]
        for i, gtcoord in enumerate(gtcoords):
            for j, predcoord in enumerate(predcoords):
                # schouten = output[0][0]
                distsq = (gtcoord - predcoord)**2
                output[jointnr][i][j] = distsq

            # dists = np.subtract.outer(gtcoords, predcoords)  # Distances between joint pred and gt
            # ph = np.dstack((gtcoords, predcoords))

    return output

AP_BPMs = list()  # List with AlphaPose bipartite matrices
OP_BPMs = list()

for i in range(len(annotations)):
    seq = annotations[i]
    seq_op = OP_keypoints[i]
    seq_ap = AP_keypoints[i]

    for picnr in seq_op.keys():

        # Initialize annotation arrays per joint
        LSH_anns = np.zeros((len(seq[picnr+1]), 3))
        RSH_anns = np.zeros((len(seq[picnr+1]), 3))
        LH_anns = np.zeros((len(seq[picnr+1]), 3))
        RH_anns = np.zeros((len(seq[picnr+1]), 3))
        # LSH_anns, RSH_anns, LH_anns, RH_anns = ([] for i in range(4))
        # Initialize OpenPose predictions per joint
        LSH_OP = np.zeros((len(seq_op[picnr]), 3))
        RSH_OP = np.zeros((len(seq_op[picnr]), 3))
        LH_OP = np.zeros((len(seq_op[picnr]), 3))
        RH_OP = np.zeros((len(seq_op[picnr]), 3))
        # LSH_OP, RSH_OP, LH_OP, RH_OP = ([] for i in range(4))

        # Initialize AlphaPose predictions per joint
        LSH_AP = np.zeros((len(seq_ap[picnr]), 3))
        RSH_AP = np.zeros((len(seq_ap[picnr]), 3))
        LH_AP = np.zeros((len(seq_ap[picnr]), 3))
        RH_AP = np.zeros((len(seq_ap[picnr]), 3))

        most_people = max(len(seq[picnr+1]), len(seq_op[picnr]), len(seq_ap[picnr]))
        for persnr in range(most_people):
            anns_joints = seq[picnr+1]
            OP_joints = seq_op[picnr]
            AP_joints = seq_ap[picnr]

            # LSH_anns[persnr] = anns_joints[persnr][0]
            fillArray(LSH_anns, persnr, anns_joints, 0)
            fillArray(RSH_anns, persnr, anns_joints, 1)
            fillArray(LH_anns, persnr, anns_joints, 2)
            fillArray(RH_anns, persnr, anns_joints, 3)

            fillArray(LSH_OP, persnr, OP_joints, 0)
            fillArray(RSH_OP, persnr, OP_joints, 1)
            fillArray(LH_OP, persnr, OP_joints, 2)
            fillArray(RH_OP, persnr, OP_joints, 3)

            fillArray(LSH_AP, persnr, AP_joints, 0)
            fillArray(RSH_AP, persnr, AP_joints, 1)
            fillArray(LH_AP, persnr, AP_joints,  2)
            fillArray(RH_AP, persnr, AP_joints,  3)

        AP_preds = [LSH_AP, RSH_AP, LH_AP, RH_AP]
        OP_preds = [LSH_OP, RSH_OP, LH_OP, RH_OP]
        anns = [LSH_anns, RSH_anns, LH_anns, RH_anns]

        AP_BPM = outersubtract(anns, AP_preds)
        OP_BPM = outersubtract(anns, OP_preds)

        AP_BPMs.append(AP_BPM)
        OP_BPMs.append(OP_BPM)



print(AP_BPMs)