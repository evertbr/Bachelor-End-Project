import json
from GetOPJoints import getJSONS
import pickle
from GetShouldersAndHips import getShouldersandHips

paths = '../annotated-data/'
annotationpaths = getJSONS(paths)
annotation_dict = {}  # [image_id : int, occluded : bool, keypoints: list[float]]

for filenr, path in enumerate(annotationpaths):
    for jsonfile in path:
        with open(jsonfile, 'r') as file:
            data = json.load(file)

            annotations = data['annotations']
            annotation_dict[filenr] = {}

            for annotation in annotations:
                image_id = annotation['image_id']
                occluded = annotation['attributes']['occluded']
                keypoints = annotation['keypoints']
                try:
                    annotation_dict[filenr][image_id].append([keypoints, occluded])
                except:
                    annotation_dict[filenr][image_id] = [[keypoints, occluded]]

with open('AnnsSH.pickle', 'wb') as file:
    pickle.dump(annotation_dict, file)
