import pickle
from copy import deepcopy

def openPickle(filename : str) -> dict:
    """Open Pickle file and store it in a dictionary"""
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)

    return dictionary

OP_keypoints = openPickle('OPSH.pickle')
AP_keypoints = openPickle('APSH.pickle')
annotations = openPickle('AnnsSH.pickle')
testanns = openPickle('OPSH.pickle')

def removeEvenFrames(keypointsdict : dict) -> dict:
    """"We only annotated every 2 frames. Thus, we must remove some frames from the OpenPose/AlphaPose estimates."""
    duplicate = deepcopy(testanns)
    for seq in duplicate.keys():
        for frame in duplicate[seq]:
            if frame % 2 != 0:
                del keypointsdict[seq][frame]

    return keypointsdict

testjoepels = removeEvenFrames(testanns)
print(testjoepels)