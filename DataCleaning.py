import pickle
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left

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
    k = np.array([0.079, 0.079, 0.107, 0.107])  # COCO per-joint constants [Lshoulder, Rshoulder, Lhip, Rhip]

    num = -(dists) ** 2
    denom = np.dot(2 * (s ** 2), k ** 2)
    expo = num / denom
    ks = np.exp(expo)

    assert np.all(0 <= ks)
    assert np.all(ks <= 1)

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


def highestKS(prediction: np.array, gts: list, areas: list):
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


def calcKS(preds: np.array, gts, area) -> np.array:
    """Given a list of predictions in order of confidence, calculate the KS for all people in a picture.
    If there are more predictions than ground truths, the redundant preds are labeled false negative"""
    ks_arr = np.zeros((len(gts), 2, 4))
    ks_arr.fill(-1)
    diff = len(gts) - len(preds)
    confs = np.sum(preds[:, :, 2], axis=1)  # Summed confidence for each body prediction

    for prednr in range(len(preds)):
        # Predictions are handled in descending order of confidence
        if gts.size != 0:
            ks, gts, occ = highestKS(preds[prednr], gts, area)  # gts decreases in size each iteration
            ks_arr[prednr] = [ks, occ]

    return ks_arr, diff, confs


def sortByConf(preds: np.array) -> list:
    """Sort predictions based on confidence score"""
    # Compute the sum of all third columns
    conf_sums = np.sum(preds[:, :, 2], axis=1)

    # Sorting indices
    sorted_by_conf_sums = np.argsort(conf_sums)[::-1]

    # Sort the array based on these indices
    preds = preds[sorted_by_conf_sums]
    return preds


def OKS(KSs: list) -> list:
    """Takes a list with keypoint similarities and occlusion variables of one picture. Returns the OKS per person."""
    OKS = np.zeros(len(KSs))  # TODO: Moet dit niet langer? hallo? is dit onzin? haal ik mn bep?

    for prednr in range(len(KSs)):
        pred = KSs[prednr]
        OKS_num = sum(pred[0] * (pred[1] > 0))
        OKS_denom = sum(pred[1] > 0)
        oks = OKS_num / OKS_denom
        OKS[prednr] = oks if oks != 0 else 0  # Account for dividing over 0 errors

    return OKS

AP_OKS_list = []  # Will be filled with all OKS values for AP's predictions
OP_OKS_list = []

AP_diffs = []  # List containing the difference between the amount of gt annotations and detected person
OP_diffs = []

AP_ranked_OKS_list = []
OP_ranked_OKS_list = []
tellert = 0
# MAIN LOOP
for i in range(len(OP_keypoints)):
    seq = annotations[i]
    seq_op = OP_keypoints[i]
    seq_ap = AP_keypoints[i]
    ###TODO: CHECK IF THE LENGTHS ARE NOT EQUAL. IF THEY ARE, INITIALIZATION MIGHT BE DONE MORE EFFICIENTLY
    for picnr in seq_op.keys():

        # Initialize annotation arrays per joint
        # Note that the ground truth has a slightly different format, which explains the + 1 in indices
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

        most_people = max(len(seq[picnr + 1]), len(seq_op[picnr]), len(seq_ap[picnr]))  # FIXME: same issue as line 169

        anns_joints = [x[0] for x in seq[picnr + 1]]
        anns_area = np.array([x[1] for x in seq[picnr + 1]])  # Save the areas of each person

        OP_joints = seq_op[picnr]
        AP_joints = seq_ap[picnr]

        for persnr in range(most_people):
            # TODO: mooi is anders

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

        # Sort predictions by confidence score per person
        AP_preds = sortByConf(AP_preds)
        OP_preds = sortByConf(OP_preds)

        # VANAF HIER PAS ECHTE BEREKENINGEN. ERVOOR LOUTER DATA CLEANEN/GOED ZETTEN.
        # Calculate keypoint similarities for AlphaPose and OpenPose
        AP_KS = calcKS(AP_preds, anns, anns_area)
        OP_KS = calcKS(OP_preds, anns, anns_area)

        AP_diff = AP_KS[1]
        OP_diff = OP_KS[1]
        AP_confs = AP_KS[2]
        OP_confs = OP_KS[2]

        AP_diffs.append(AP_diff)
        OP_diffs.append(OP_diff)

        AP_OKS = OKS(AP_KS[0])
        OP_OKS = OKS(OP_KS[0])

        if len(AP_confs) >= len(AP_OKS):
            AP_confs = AP_confs[:len(AP_OKS)]  # Only keep matched detections
        else:
            AP_OKS = AP_OKS[:len(AP_confs)]

        if len(OP_confs) >= len(OP_OKS):
            OP_confs = OP_confs[:len(OP_OKS)]  # Only keep matched detections
        else:
            OP_OKS = OP_OKS[:len(OP_confs)]

        AP_ranked_OKS = np.array((AP_confs, AP_OKS))
        OP_ranked_OKS = np.array((OP_confs, OP_OKS))

        AP_ranked_OKS_list.append(AP_ranked_OKS)
        OP_ranked_OKS_list.append(OP_ranked_OKS)

AP_conf_OKS = np.concatenate(AP_ranked_OKS_list, axis=1)
sort_AP_conf_OKS = np.argsort(AP_conf_OKS[0])[::-1]
AP_conf_OKS = AP_conf_OKS[:, sort_AP_conf_OKS]

OP_conf_OKS = np.concatenate(OP_ranked_OKS_list, axis=1)
sort_OP_conf_OKS = np.argsort(OP_conf_OKS[0])[::-1]
OP_conf_OKS = OP_conf_OKS[:, sort_OP_conf_OKS]

def avg_per_frame(OKS_list: list):
    """Find the average OKS per frame for a list of predictions"""
    avgs = np.zeros(len(OKS_list))
    for i in range(len(OKS_list)):
        avgs[i] = np.mean(OKS_list[i])

    return avgs


def len_per_frame(kp_list: list):
    """Find the amount of annotated person objects for a list of annotations"""
    lens = np.zeros(len(kp_list))
    for i in range(len(kp_list)):
        lens[i] = len(kp_list[i])

    return lens


AP_avgs = avg_per_frame(AP_OKS_list)
OP_avgs = avg_per_frame(OP_OKS_list)

AP_lens = len_per_frame(AP_OKS_list)
OP_lens = len_per_frame(OP_OKS_list)

anns_lens = []
for seq in annotations:
    for pic in annotations[seq]:
        anns_lens.append(len(annotations[seq][pic]))

AP_conf_OKS = AP_conf_OKS.T
OP_conf_OKS = OP_conf_OKS.T

num_annotated = np.sum(anns_lens)  # total amount of annotations

def precison_recall(conf_OKS, threshold, pos):
    """Calculate the precisions and recalls for different threshold values
    conf_OKS: numpy array containing summed confidence and OKS for all people from one model
    threshold: the OKS threshold value
    pos: total amount of positive annotations (people) in the ground truth set"""
    truepos = 0
    falsepos = 0

    precisions = np.empty(len(conf_OKS))
    recalls = np.empty(len(conf_OKS))

    for rank in range(len(conf_OKS)):
        conf, oks = conf_OKS[rank]
        if oks >= threshold:
            truepos += 1
        else:
            falsepos += 1

        precisions[rank] = truepos/ (truepos + falsepos)
        recalls[rank] = truepos / pos


    return precisions, recalls

recallpoints = np.arange(0,1.1, 0.01)

oks_thresholds = np.arange(0.5, 1, 0.05)
# blub = AP_conf_OKS.T[1]
# ledezma = blub >= 0.999
# saibari = ledezma.sum()

def plotPRcurve(precs, recs, recallthresholds, modelname, oksthreshold):
    """Plot the precision recall
    precs: np.array containing precision values
    recs: np.array containing recall values
    recallthresholds: recall values at which to measure the precision
    modelname: string containing the model name. Used for labelling only
    oksthreshold: the OKS threshold. Again, only for labelling"""

    precs_to_plot = np.empty(len(recallthresholds))
    recs_to_plot = np.empty(len(recallthresholds))
    precs_to_plot.fill(np.nan)
    recs_to_plot.fill(np.nan)
    j = 0

    for threshold in recallthresholds:
        try:
            idx = bisect_left(recs, threshold)
            recs_to_plot[j] = recs[idx]
            precs_to_plot[j] = precs[idx]
            j+=1
        except:
            break

    plt.plot(recs_to_plot, precs_to_plot)
    plt.title("PR-curves for " + modelname)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.legend([round(x, 2) for x in oks_thresholds])
    plt.savefig("./Graphs/Results/PR-curves/" + modelname + str(round(oksthreshold, 2)) + ".png")


def average_precision(precs, recs):
    """Calculate the approximated average precision"""
    APrec = 0

    for i in range(len(precs)):
        if i == 0:
            Delta_r = recs[i]
        else:
            Delta_r = recs[i] - recs[i-1]
        APrec += precs[i] * Delta_r

    return APrec

APAPRs = []
OPAPRs = []


# Create PR-curves and calculate the average precision for different oks threshold values
for oks_threshold in oks_thresholds:
    OP_precs, OP_recs = precison_recall(OP_conf_OKS, oks_threshold, num_annotated)
    # plotPRcurve(OP_precs, OP_recs, recallpoints, 'OpenPose', oks_threshold)
    AP_precs, AP_recs = precison_recall(AP_conf_OKS, oks_threshold, num_annotated)
    # plotPRcurve(AP_precs, AP_recs, recallpoints, 'AlphaPose', oks_threshold)
    OPAPR = average_precision(OP_precs, OP_recs)
    APAPR = average_precision(AP_precs, AP_recs)
    APAPRs.append(APAPR)
    OPAPRs.append(OPAPR)


plt.plot(oks_thresholds, APAPRs, color='orange', label='AlphaPose', marker='o')
plt.plot(oks_thresholds, OPAPRs, color='green', label='OpenPose', marker='x')
plt.xlabel('OKS threshold')
plt.ylabel('Average Precision')
plt.ylim(0,1)
plt.title("Average Precisions")
plt.legend()
plt.savefig("./Graphs/Results/averageprecisions.png")