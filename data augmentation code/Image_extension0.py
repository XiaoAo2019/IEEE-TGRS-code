import os
from PIL import Image
import Image_extension as T

path_in_img = r" "
path_in_mask = r" "

file_in_img = os.listdir(path_in_img)
file_in_mask = os.listdir(path_in_mask)
num_file_in = len(file_in_img)
file_in_img.sort(key=lambda x: int(x[6:-4]))
file_in_mask.sort(key=lambda x: int(x[6:-4]))

m = 0
base_size = 256
crop_size = 256
min_size = int(0.5 * base_size)
max_size = int(1.5 * base_size)
rotate_prob = 1
flip_prob = 0.5
BSH_prob = 0.5

for j in range(0, 30):
    for i in range(0, num_file_in):

        m +=1

        img = Image.open(os.path.join(path_in_img, file_in_img[i]))
        target = Image.open(os.path.join(path_in_mask, file_in_mask[i]))

        trans = [T.RandomResize(min_size, max_size)]

        if rotate_prob > 0:
            trans.append(T.Rotate(rotate_prob))

        if flip_prob > 0:
            trans.append(T.RandomFlip(flip_prob))

        if BSH_prob > 0:
            trans.append(T.B_S_H(BSH_prob))

        trans.append(T.RandomCrop(crop_size))

        if rotate_prob > 0:
            trans.append(T.Rotate(rotate_prob))

        if flip_prob > 0:
            trans.append(T.RandomFlip(flip_prob))


        transforms = T.Compose(trans)

        img, target = transforms(img, target)

        img.save(r"../{}_{}.tif".format('train', str(m)))
        target.save(r"../{}_{}.png".format('train', str(m)))
