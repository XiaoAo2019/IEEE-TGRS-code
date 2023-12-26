import os
import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from osgeo import gdal
from src import unet_resnet50
import torch.nn.functional as F
import transforms as T


def detect_image(image):

    classes = 1
    weights_path = r"./"                                                                                                # load weights
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = unet_resnet50(num_classes=classes+1)

    # load weights
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict)
    model.to(device)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([#transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(image)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)


    model.eval()
    with torch.no_grad():
        # init model
        # channel, height, width = image.shape
        # init_img = torch.zeros((1, 3, height, width), device=device)
        # model(init_img)

        t_start = time.time()
        output = model(img.to(device))
        t_end = time.time()
        print("inference time: {}".format(t_end - t_start))
        output = output['out']

        pr = torch.squeeze(output, dim=0)
        pr = F.softmax(pr, dim=0).cpu().numpy()
        pr = pr.argmax(axis=0)


    image_array = np.array(image)
    image1 = np.transpose(image_array, (2, 0, 1))
    masked_image = image1.astype(np.uint32).copy()

    color = (255, 0, 0)                                                                                                 # color

    for c in range(3):
        alpha = 1
        masked_image[c, :, :] = np.where(pr == 1,
                                         masked_image[c, :, :] *
                                         (1 - alpha) + alpha * color[c],
                                         masked_image[c, :, :])
    image2 = np.transpose(masked_image, (1, 2, 0))
    r_image = Image.fromarray(np.uint8(image2))
    return r_image

def main():

    IMAGE_DIR = r'./'                                                                                                   # input file
    path_out = r'./'                                                                                                    # output file

    count = os.listdir(IMAGE_DIR)
    # count.sort(key=lambda x: int(x[4:-4]))
    for i in range(0, len(count)):
        path = os.path.join(IMAGE_DIR, count[i])
        data = Image.open(path).convert('RGB')
        r_image = detect_image(data)
        r_image.save(path_out+'{}.tif'.format(str(count[i])))


if __name__ == '__main__':
    main()