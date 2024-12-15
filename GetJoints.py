import os
import pandas as pd
import pickle

# Get the paths to all sequences and store them
OP_path = "../RESULTS/OpenPose/"  # Path to folder with results of applying OpenPose to our data
OP_seqs = [x[0] for x in os.walk(OP_path)]  # List with all subdirectories with annotated sequences
OP_seqs = OP_seqs[1:]  # Remove the root directory
OP_jsons = []

for seq in OP_seqs:
    seq_jsons = [seq + '/' + OP_json for OP_json in os.listdir(seq) if OP_json.endswith('.json')]  # List with paths of all jsons for this sequence
    OP_jsons.append(seq_jsons)  # Append to a list of lists with jsons of all sequences

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
keypoint_dict = {}
for seqnr, sequence in enumerate(OP_jsons):  # Go through all sequences
    for frame in sequence:  # Go through all json files in a sequence
        poses = readyData(frame)
        keypoints = poses.apply(getPoseKeypoints)  # Gets a pd Series with keypoints for all people in this frame
        keypoint_dict[seqnr] = keypoints

# Save the dictionary as a pickle file. Note that it can't be saved as a json due to nestedness
with open('OPKeypoints.pickle', 'wb') as file:
    pickle.dump(keypoint_dict, file)
