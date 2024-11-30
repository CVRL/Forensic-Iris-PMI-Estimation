import os
import random
from PIL import Image

root_dir = './pmi-generate-output/'
output_dir = 'pm-iris-synthetic-images/'

pmi_ranges = [(1, 24), (25, 48), (49, 72), (73, 96), (97, 120), (121, 144), (145, 168), (169, 192),
              (193, 216), (217, 240), (241, 264), (265, 288), (289, 312), (313, 336), (337, 360),
              (361, 384), (385, 408), (409, 1674)]

subdirs = sorted([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])

with open('synthetic_metadata.txt', 'w') as f:
    f.write('filename' + ',' + 'subject_id' + ',' + 'pmi' + '\n')

for idx, subdir in enumerate(subdirs):
    file_dir = os.path.join(root_dir, subdir)
    pmi_range = pmi_ranges[idx]
    filenames = sorted(os.listdir(file_dir))

    for filename in filenames:
        img_path = os.path.join(file_dir, filename)
        subject = filename[12:16]
        pmi = float(random.randint(pmi_range[0], pmi_range[1]))

        image = Image.open(img_path)
        image.save(os.path.join(output_dir, filename))

        with open('synthetic_metadata.txt', 'a') as f:
            f.write(filename + ',' + subject + ',' + str(pmi) + '\n')

        print(filename, subject, pmi)

