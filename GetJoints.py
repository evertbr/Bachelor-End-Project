import os
import pandas as pd
import pickle

# Get the paths to all sequences and store them
def getJSONS(rootpath : str) -> list:
    """Gets all JSON files except the root file for a directory"""
    seqs = [x[0] for x in os.walk(rootpath)]  # List with all subdirectories with annotated sequences
    seqs = seqs[1:]  # Remove the root directory
    files = []

    for seq in seqs:
        seq_jsons = [seq + '/' + file for file in os.listdir(seq) if file.endswith('.json')]  # List with paths of all jsons for this sequence
        files.append(seq_jsons)  # Append to a list of lists with jsons of all sequences

    files = [x for x in files if x != []]  # Remove possible empty lists
    return files

# Define functions
def readyData(OpenPoseJSON) -> pd.Series:
    """Takes a .json file with OpenPose format estimations and returns a pandas Series with usable data"""
    OP_df = pd.read_json(OpenPoseJSON)  # Read in data
    OP_df.drop('version', axis=1, inplace=True)  # Drop useless version column
    people_col = OP_df['people']
    return people_col

def getPoseKeypoints(personDict: dict) -> list:
    """Takes a person's OpenPose estimations and returns a list with the estimated pose keypoints"""
    pose_keypoints = personDict['pose_keypoints_2d']
    return pose_keypoints

# Create the final dictionary
# keypoint_dict = {}
# for seqnr, sequence in enumerate(OP_jsons):  # Go through all sequences
#     frame_dict = {}
#     for framenr, frame in enumerate(sequence):  # Go through all json files in a sequence
#         poses = readyData(frame)
#         keypoints = poses.apply(getPoseKeypoints)  # Gets a pd Series with keypoints for all people in this frame
#         frame_dict[framenr] = keypoints
#     keypoint_dict[seqnr] = frame_dict
#
# # Save the dictionary as a pickle file. Note that it can't be saved as a json due to nestedness
# with open('OPKeypoints2.pickle', 'wb') as file:
#     pickle.dump(keypoint_dict, file)
