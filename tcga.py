### TCGA
import os

openslide_path = os.getcwd() + "\\openslide-win64-20171122\\bin"
os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']
import openslide as slide
from PIL import Image
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
from os import listdir, mkdir, path, makedirs
from os.path import join
import time, sys, warnings, glob
import threading
from tqdm import tqdm
import argparse
from histolab.slide import Slide
from histolab.tiler import ScoreTiler, GridTiler
from histolab.scorer import NucleiScorer, CellularityScorer
import histolab.filters.image_filters as imf
import histolab.filters.morphological_filters as mof
from histolab.masks import BiggestTissueBoxMask, TissueMask, BinaryMask
from functools import reduce
import tcga_util as tu
warnings.simplefilter('ignore')


def thres_saturation(img, t=15):
    # typical t = 15
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t


def crop_slide(img, save_slide_path, position=(0, 0), step=(0, 0), patch_size=224):  # position given as (x, y)
    img = img.read_region((position[0] * 4, position[1] * 4), 1, (patch_size, patch_size))
    img = np.array(img)[..., :3]
    if thres_saturation(img, 30):
        patch_name = "{}_{}".format(step[0], step[1])
        io.imsave(join(save_slide_path, patch_name + ".jpg"), img_as_ubyte(img))


def slide_to_patch(out_base, img_slides, step, folder='test'):
    makedirs(out_base, exist_ok=True)
    patch_size = 224
    step_size = step
    for s in range(len(img_slides)):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        bag_path = join(out_base, img_name)
        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)
        dimension = img.level_dimensions[1]  # given as width, height
        if folder == 'test':
            thumbnail = np.array(img.get_thumbnail((int(dimension[0]) / 7, int(dimension[1]) / 7)))[..., :3]
        else:
            thumbnail = np.array(img.get_thumbnail((int(dimension[0]) / 28, int(dimension[1]) / 28)))[..., :3]
        io.imsave(join(folder, 'thumbnails', img_name + ".png"), img_as_ubyte(thumbnail))
        step_y_max = int(np.floor(dimension[1] / step_size))  # rows
        step_x_max = int(np.floor(dimension[0] / step_size))  # columns
        for j in range(step_y_max):  # rows
            for i in range(step_x_max):  # columns
                crop_slide(img, bag_path, (i * step_size, j * step_size), step=(j, i), patch_size=patch_size)
            sys.stdout.write('\r Cropped: {}/{} -- {}/{}'.format(s + 1, len(img_slides), j + 1, step_y_max))


def run_slide():
    parser = argparse.ArgumentParser(description='Generate patches from testing slides')
    parser.add_argument('--dataset', type=str, default='tcga', help='Dataset name [tcga]')
    args = parser.parse_args()
    if args.dataset == 'tcga':
        path_base = ('test/input')  # input dir
        out_base = ('test/patches')  # output
        folder = 'test'
        makedirs('test/thumbnails', exist_ok=True)

    all_slides = glob.glob(join(path_base, '*.svs')) + glob.glob(join(path_base, '*.tif'))
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=224)
    args = parser.parse_args()

    print('Cropping patches, please be patient')
    step = args.patch_size - args.overlap
    slide_to_patch(out_base, all_slides, step, folder)


# 前面的没什么用

def histolab_func():
    input_path = r".\test\input\TCGA-06-0125-01A-01-TS1.5c704cf9-fb9e-46ce-b53b-5b3b3cc908f7.svs"
    processed_path = r".\test\processed"

    tcga_slide = Slide(input_path, processed_path=processed_path, use_largeimage=True, )
    # tcga_slide.show()
    scored_tiles_extractor = ScoreTiler(
        scorer=CellularityScorer(),
        tile_size=(224, 224),
        n_tiles=200,
        level=0,
        check_tissue=True,
        tissue_percent=90.0,
        pixel_overlap=0,  # default
        prefix="scored/",  # save tiles in the "scored" subdirectory of slide's processed_path
        suffix=".png"  # default
    )

    extraction_mask = tu.MyMask()
    extraction_mask.custom_filters = [
        tu.BluePenFilter(),  # 对蓝色墨迹进行去除
        imf.RgbToGrayscale(),
        imf.OtsuThreshold(),  # 阈值分割
        mof.BinaryErosion(disk_size=1),  # 腐蚀
        mof.BinaryDilation(disk_size=1),  # 膨胀
        mof.RemoveSmallHoles(area_threshold=500),
        mof.RemoveSmallObjects(min_size=1500), ]
    summary_filename = "summary_ovarian_tiles3.csv"
    SUMMARY_PATH = os.path.join(processed_path, summary_filename)
    locate_img = scored_tiles_extractor.locate_tiles(
        slide=tcga_slide,
        scale_factor=24,  # default
        alpha=224,  # default
        outline="red",
        extraction_mask=extraction_mask  # default

    )
    locate_img.show()
    locate_img.save(os.path.join(processed_path, "locate.png"))

    scored_tiles_extractor.extract(tcga_slide, report_path=SUMMARY_PATH,  extraction_mask=extraction_mask)


if __name__ == '__main__':
    # run_slide()
    histolab_func()
