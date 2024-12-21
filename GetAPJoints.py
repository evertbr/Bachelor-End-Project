from GetOPJoints import getJSONS
import json
import pickle
from GetShouldersAndHips import getShouldersandHips

AP_path = "../RESULTS/AlphaPose"

paths = getJSONS(AP_path)
annotation_dict = {}

for path in paths:
    for filenr, jsonfile in enumerate(path):
        with open(jsonfile, 'r') as file:
            data = json.load(file)

            annotation_dict[filenr] = {}
            for estimate in data:
                image_id = estimate['image_id']
                keypoints = estimate['keypoints']
                joints = getShouldersandHips(keypoints)
                try:
                    annotation_dict[filenr][image_id].append(joints)
                except:
                    annotation_dict[filenr][image_id] = [joints]

with open('APSH2.pickle', 'wb') as file:
    pickle.dump(annotation_dict, file)
