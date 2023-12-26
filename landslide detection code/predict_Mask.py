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


class GRID:
    # read image
    def load_image(self, filename):
        image = gdal.Open(filename)

        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height)

        del image

        return img_proj, img_geotrans, img_data

    # write image
    def write_image(self, filename, img_proj, img_geotrans, img_data):
        # if 'int8' in img_data.dtype.name:
        #     datatype = gdal.GDT_Byte
        # elif 'int16' in img_data.dtype.name:
        #     datatype = gdal.GDT_UInt16
        # else:
        #     datatype = gdal.GDT_Float32
        datatype = gdal.GDT_Byte

        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image


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


    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])

    image1 = np.transpose(image, (1, 2, 0))
    # image1 = data_transform(image).permute(1, 2, 0)
    image2 = data_transform(image1)
    img = torch.unsqueeze(image2, dim=0)



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

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)

        prediction[np.where(prediction > 0)] = 255
        mask = Image.fromarray(prediction)
        mask.putpalette([128, 0, 128,
                         255, 255, 0])

    return mask

def main():

    IMAGE_DIR = r'./'                                                                                                   # input file
    path_out = r'./'                                                                                                    # output file

    run = GRID()
    count = os.listdir(IMAGE_DIR)
    for i in range(0, len(count)):
        path = os.path.join(IMAGE_DIR, count[i])

        proj, geotrans, data = run.load_image(path)
        r_image = detect_image(data)

        r_image.save(path_out+'{}.png'.format(str(count[i])))
        # run.write_image(path_out + '{}.png'.format(str(count[i])), proj, geotrans, r_image)



if __name__ == '__main__':
    main()
