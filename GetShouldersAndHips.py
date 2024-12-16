import pickle

# Get the full keypoints dictionary
with open('OPKeypoints2.pickle', 'rb') as file:
    keypoints_dict = pickle.load(file)

def getShouldersandHips(keypointlist : list) -> list:
    """Takes a list of all COCO format keypoints and returns only shoulders and hips (x,y) coords with confidence score"""
    Rshoulder = keypointlist[6:9]
    Lshoulder = keypointlist[15:18]
    Rhip = keypointlist[27:30]
    Lhip = keypointlist[46:49]

    return [Rshoulder, Lshoulder, Rhip, Lhip]

# Trim down the full keypoints dictionary to only shoulders and hips, which will be used in the project.
SHdict = {}
for seqnr, seq in enumerate(keypoints_dict.values()):
    frameSH_dict = {}
    for framenr, frame in enumerate(seq.values()):
        SHlist = frame.apply(getShouldersandHips)  # List with shoulder and hips coordinates for this frame
        frameSH_dict[framenr] = SHlist

    SHdict[seqnr] = frameSH_dict

# Save the dictionary as a pickle file.
with open('OPSH.pickle', 'wb') as file:
    pickle.dump(SHdict, file)