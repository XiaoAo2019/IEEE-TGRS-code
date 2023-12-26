import os
import time
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



def detect_images(images, model, device):
    model.eval()
    with torch.no_grad():
        t_start = time.time()
        outputs = model(images.to(device))
        t_end = time.time()
        print("Inference time: {:.2f} seconds".format(t_end - t_start))

        outputs = outputs['out']
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    return preds

def colorize_predictions(predictions, original_images):
    # color
    color_mapping = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 0.0, 255.0),
    }

    colored_images = []
    for pred, original_image in zip(predictions, original_images):
        colored_image = np.zeros_like(original_image, dtype=np.uint8)
        # colored_image = np.zeros((3, pred.shape[0], pred.shape[1]), dtype=np.uint8)
        for label, color in color_mapping.items():
            mask = pred == label
            for c in range(3):
                # colored_image[c, :, :] = np.where(mask, color[c], colored_image[c, :, :])
                colored_image[c, :, :] = np.where(mask, color[c], original_image[c, :, :])
        colored_images.append(colored_image)

    return colored_images


def main():
    IMAGE_DIR = r'./'                                                                                                   # input file
    path_out = r'./'                                                                                                    # output file
    weights_path = r'./'                                                                                                # load weight
    classes = 1
    batch_size = 8                                                                                                      # batch_size

    run = GRID()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = unet_resnet50(num_classes=classes + 1)
    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict)
    model.to(device)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    count = os.listdir(IMAGE_DIR)
    num_images = len(count)

    predictions = []
    original_images = []
    proj = []
    geotrans = []
    for i in range(0, num_images, batch_size):
        batch_paths = count[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            img_path = os.path.join(IMAGE_DIR, path)
            img_proj, img_geotrans, data = run.load_image(img_path)
            proj.append(img_proj)
            geotrans.append(img_geotrans)
            original_images.append(data.copy())

            image = np.transpose(data, (1, 2, 0))
            # original_images.append(image.copy())

            image = data_transform(image)
            image = torch.unsqueeze(image, dim=0)
            batch_images.append(image)

        batch_images = torch.cat(batch_images, dim=0)
        preds = detect_images(batch_images, model, device)
        predictions.extend(preds)

    colored_images = colorize_predictions(predictions, original_images)

    for j, (path, img_proj, img_geotrans) in enumerate(zip(count, proj, geotrans)):
        output_path = os.path.join(path_out, '{}.tif'.format(str(path)))
        run.write_image(output_path, img_proj, img_geotrans, colored_images[j])


if __name__ == '__main__':
    main()
