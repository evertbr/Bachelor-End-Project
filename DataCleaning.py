import pickle
from copy import deepcopy
import numpy as np


def openPickle(filename: str) -> dict:
    """Open Pickle file and store it in a dictionary"""
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)

    return dictionary


OP_keypoints = openPickle('OPSHFINALFINAL.pickle')  # x, y, confidence
AP_keypoints = openPickle('APSHFINALFINAL.pickle')  # x, y, confidence
annotations = openPickle('AnnsSHmetArea.pickle')  # x, y, occluded, area

# Fix AP keypoints naming
for seq in AP_keypoints.keys():
    for pic in list(AP_keypoints[seq].keys()):
        picnr = int(pic.split('.')[0])
        AP_keypoints[seq][picnr] = AP_keypoints[seq][pic]
        del AP_keypoints[seq][pic]


def removeEvenFrames(keypointsdict: dict) -> dict:
    """We only annotated every 2 frames. Thus, we must remove some frames from the OpenPose/AlphaPose estimates."""
    duplicate = deepcopy(keypointsdict)
    for seq in duplicate.keys():
        for framenr, frame in enumerate(duplicate[seq]):
            if framenr % 2 != 0:
                del keypointsdict[seq][frame]
            else:
                if framenr > 0:  # 0/2 = 0
                    # Fix order
                    keypointsdict[seq][int(frame / 2)] = keypointsdict[seq][frame]
                    del keypointsdict[seq][frame]

    return keypointsdict


OP_keypoints = removeEvenFrames(OP_keypoints)
AP_keypoints = removeEvenFrames(AP_keypoints)


def KS(dists: np.array, s) -> float:
    """Determines the keypoint similarity for one keypoint instance of type ktype
    dist: the Euclidian distances from prediction to ground truth for one person's joints
    jointtype: keypoint type. [left shoulder, right shoulder, left hip, right hip]
    s: area of the person object"""
    k = np.array([0.079, 0.107, 0.079, 0.107])  # COCO per-joint constants [Lshoulder, Rshoulder, Lhip, Rhip]

    num = -(dists)**2
    denom = np.dot(2*s**2,k **2)
    expo = num/denom
    ks = np.exp(expo)

    return ks


### The ground truth annotation format is not the same as the predictions' format.
### Following code snippet fixes it.

for i in range(len(annotations)):
    seq = annotations[i]
    for picnr in seq.keys():
        for persnr, person in enumerate(seq[picnr]):
            area = seq[picnr][persnr][2]
            seq[picnr][persnr] = [[person[0][j:j + 3] for j in range(0, len(person[0]), 3)], area]


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


def highestKS(prediction : np.array, gts : list, areas: list):
    """Given one prediction instance and a list of possible associated ground truth annotations,
    return the ground truth annotation with the highest keypoint similarity. Remove this ground truth annotation from
    the list of ground truth annotations."""
    predx = prediction[:, 0]
    predy = prediction[:, 1]
    maxks = np.array([-1, -1, -1, -1])
    maxks_ind = -1
    occ = -1

    for gtind, gt in enumerate(gts):
        distsq = (predx - gt[:, 0]) ** 2 + (predy - gt[:, 1]) ** 2
        dist = np.sqrt(distsq)
        area = areas[gtind]
        ks = KS(dist, area)
        if sum(maxks) < sum(ks):
            maxks = ks
            maxks_ind = gtind
            occ = gt[:, 2]  # Occlusion variable

    gts = np.delete(gts, maxks_ind, 0)

    return maxks, gts, occ

def fillArray(array: np.array, ind, joints, jointnr) -> None:
    """Fill an array of coordinates per joint. Deals with the fact that the number of estimated joints
    is not necessarily the same as the number of annotated joints by the try-except statement."""
    try:
        array[ind] = joints[ind][jointnr]
    except:
        pass


def calcKS(preds : np.array, gts, area) -> np.array:
    shortest = min(len(gts), len(preds))
    longest = max(len(gts), len(preds))
    ks_list = list()

    for prednr in range(shortest):
        # Predictions are handled in descending order of confidence
        ks, gts, occ = highestKS(preds[prednr], gts, area)  # gts decreases in size each iteration
        ks_list.append([ks, occ])
    return ks_list


def sortByConf(preds : np.array) -> list:
    """Sort predictions based on confidence score"""
    # Compute the sum of all third columns
    conf_sums = np.sum(preds[:, :, 2], axis=1)

    # Sorting indices
    sorted_by_conf_sums = np.argsort(conf_sums)[::-1]

    # Sort the array based on these indices
    preds = preds[sorted_by_conf_sums]
    return preds

def OKS(KSs : list) -> list:
    """Takes a list with keypoint similarities and occlusion variables of one picture. Returns the OKS per person."""
    OKS = np.zeros(len(KSs))

    for prednr in range(len(KSs)):
        pred = KSs[prednr]
        OKS_num = sum(pred[0] * (pred[1] > 0))
        OKS_denom = sum(pred[1] > 0)
        OKS[prednr] = OKS_num / OKS_denom

    return OKS


AP_OKS_list = []  # Will be filled with all OKS values for AP's predictions
OP_OKS_list = []

# MAIN LOOP
for i in range(len(OP_keypoints)):
    seq = annotations[i]
    seq_op = OP_keypoints[i]
    seq_ap = AP_keypoints[i]

    for picnr in seq_op.keys():

        # Initialize annotation arrays per joint
        LSH_anns = np.zeros((len(seq[picnr + 1]), 3))
        RSH_anns = np.zeros((len(seq[picnr + 1]), 3))
        LH_anns = np.zeros((len(seq[picnr + 1]), 3))
        RH_anns = np.zeros((len(seq[picnr + 1]), 3))

        # Initialize OpenPose predictions per joint
        LSH_OP = np.zeros((len(seq_op[picnr]), 3))
        RSH_OP = np.zeros((len(seq_op[picnr]), 3))
        LH_OP = np.zeros((len(seq_op[picnr]), 3))
        RH_OP = np.zeros((len(seq_op[picnr]), 3))

        # Initialize AlphaPose predictions per joint
        LSH_AP = np.zeros((len(seq_ap[picnr]), 3))
        RSH_AP = np.zeros((len(seq_ap[picnr]), 3))
        LH_AP = np.zeros((len(seq_ap[picnr]), 3))
        RH_AP = np.zeros((len(seq_ap[picnr]), 3))

        most_people = max(len(seq[picnr + 1]), len(seq_op[picnr]), len(seq_ap[picnr]))

        anns_joints = [x[0] for x in seq[picnr + 1]]
        anns_area = np.array([x[1] for x in seq[picnr + 1]])  # Save the areas of each person

        OP_joints = seq_op[picnr]
        AP_joints = seq_ap[picnr]

        for persnr in range(most_people):
            # Fill arrays for each joint for each dataset
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
            fillArray(LH_AP, persnr, AP_joints, 2)
            fillArray(RH_AP, persnr, AP_joints, 3)

        AP_preds = [LSH_AP, RSH_AP, LH_AP, RH_AP]
        OP_preds = [LSH_OP, RSH_OP, LH_OP, RH_OP]
        anns = [LSH_anns, RSH_anns, LH_anns, RH_anns]

        # Reshape coordinate arrays to (person, joint, coordinates). This eases the joint matching procedure
        AP_preds = np.transpose(AP_preds, (1, 0, 2))
        OP_preds = np.transpose(OP_preds, (1, 0, 2))
        anns = np.transpose(anns, (1, 0, 2))

        # Sort predictions by confidence score
        AP_preds = sortByConf(AP_preds)
        OP_preds = sortByConf(OP_preds)

        # Calculate keypoint similarities for AlphaPose and OpenPose
        AP_KS = calcKS(AP_preds, anns, anns_area)
        OP_KS = calcKS(OP_preds, anns, anns_area)

        AP_OKS = OKS(AP_KS)
        OP_OKS = OKS(OP_KS)

        AP_OKS_list.append(AP_OKS)
        OP_OKS_list.append(OP_OKS)

print('zijn we al kampioen?')