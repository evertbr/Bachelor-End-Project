from PIL import Image
import os

path = "../annotated-data/pics/"
picdirs = [x[0] for x in os.walk(path)]  # List with all subdirectories with pictures
picdirs = picdirs[1:]  # Remove root dir
pic_paths = []

# Get a list of all picture paths
for seq in picdirs:
    seq_jsons = [seq + '/' + picname for picname in os.listdir(seq) if picname.endswith('.jpg')]  # List with paths of all jsons for this sequence
    pic_paths.append(seq_jsons)  # Append to a list of lists with jsons of all sequences

# Flip all pictures
for seqnr, seq in enumerate(pic_paths):
    for framenr, frame in enumerate(pic_paths[seqnr]):
        original = Image.open(frame)
        pic_paths[seqnr][framenr] = original.transpose(method=Image.FLIP_LEFT_RIGHT)  # Store flipped images in the original list for efficiency

for seqnr, seq in enumerate(pic_paths):
    for framenr, frame in enumerate(pic_paths[seqnr]):
        frame.save('FLIP' + str(seqnr *100 + framenr) + '.png')


