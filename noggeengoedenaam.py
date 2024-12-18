import pandas as pd
import json
from GetJoints import getJSONS


paths = '../annotated-data/'
annotationpaths = getJSONS(paths)

for path in annotationpaths:
    with open(path, 'r') as file:
        data = json.load(file)
    
    annotations = data['annotations']
    annotation_list = []   # [image_id : int, occluded : bool, keypoints: list[float]]

    for annotation in annotations:
        image_id = annotation['image_id']
        occluded = annotation['attributes']['occluded']
        keypoints = annotation['keypoints']
        annotation_list.append([image_id, occluded, keypoints])

print(annotation_list)