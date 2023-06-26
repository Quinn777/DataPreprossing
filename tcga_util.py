import PIL
from histolab.filters.image_filters import ImageFilter, Filter, Compose
from histolab.util import np_to_pil
from histolab.masks import BinaryMask
from histolab.filters.compositions import FiltersComposition
from histolab.slide import Slide
from functools import reduce
import numpy as np
from functools import lru_cache
from typing import Iterable, List, Union


def apply_mask_image(img: PIL.Image.Image, mask: np.ndarray) -> PIL.Image.Image:
    img_arr = np.array(img)

    if mask.ndim == 2 and img_arr.ndim != 2:
        n_channels = img_arr.shape[2]
        for channel_i in range(n_channels):
            img_arr[~mask] = [255, 255, 255]  # mask为0的位置转为255
    else:
        img_arr[~mask] = [255, 255, 255]
    return np_to_pil(img_arr)


class BluePenFilter(ImageFilter):
    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return self.blue_pen_filter(img)

    def blue_pen_filter(self, img: PIL.Image.Image) -> PIL.Image.Image:
        """Filter out blue pen marks from a diagnostic slide.

        The resulting mask is a composition of green filters with different thresholds
        for the RGB channels.

        Parameters
        ---------
        img : PIL.Image.Image
            Input RGB image

        Returns
        -------
        PIL.Image.Image
            Input image with the blue pen marks filtered out.
        """
        parameters = [
            {"red_thresh": 60, "green_thresh": 120, "blue_thresh": 190},
            {"red_thresh": 120, "green_thresh": 170, "blue_thresh": 200},
            {"red_thresh": 175, "green_thresh": 210, "blue_thresh": 230},
            {"red_thresh": 145, "green_thresh": 180, "blue_thresh": 210},
            {"red_thresh": 37, "green_thresh": 95, "blue_thresh": 160},
            {"red_thresh": 30, "green_thresh": 65, "blue_thresh": 130},
            {"red_thresh": 130, "green_thresh": 155, "blue_thresh": 180},
            {"red_thresh": 40, "green_thresh": 35, "blue_thresh": 85},
            {"red_thresh": 30, "green_thresh": 20, "blue_thresh": 65},
            {"red_thresh": 90, "green_thresh": 90, "blue_thresh": 140},
            {"red_thresh": 60, "green_thresh": 60, "blue_thresh": 120},
            {"red_thresh": 110, "green_thresh": 110, "blue_thresh": 175},
        ]

        blue_pen_filter_img = reduce(
            (lambda x, y: x & y), [self.blue_filter(img, **param) for param in parameters]
        )
        return apply_mask_image(img, blue_pen_filter_img)

    def blue_filter(self,
                    img: PIL.Image.Image, red_thresh: int, green_thresh: int, blue_thresh: int
                    ) -> np.ndarray:
        """Filter out blueish colors in an RGB image.

        Create a mask to filter out blueish colors, where the mask is based on a pixel
        being above a red channel threshold value, above a green channel threshold value,
        and below a blue channel threshold value.

        Parameters
        ----------
        img : PIL.Image.Image
            Input RGB image
        red_thresh : int
            Red channel lower threshold value.
        green_thresh : int
            Green channel lower threshold value.
        blue_thresh : int
            Blue channel upper threshold value.

        Returns
        -------
        np.ndarray
            Boolean NumPy array representing the mask.
        """
        if np.array(img).ndim != 3:
            raise ValueError("Input must be 3D.")
        if not (
                0 <= red_thresh <= 255 and 0 <= green_thresh <= 255 and 0 <= blue_thresh <= 255
        ):
            raise ValueError("RGB Thresholds must be in range [0, 255]")
        img_arr = np.array(img)
        red = img_arr[:, :, 0] > red_thresh
        green = img_arr[:, :, 1] > green_thresh
        blue = img_arr[:, :, 2] < blue_thresh
        return red | green | blue


class MyMask(BinaryMask):
    def __init__(self, *filters: Iterable[Filter]) -> None:
        self.custom_filters = filters

    @lru_cache(maxsize=100)
    def _mask(self, slide):
        thumb = slide.thumbnail
        if len(self.custom_filters) == 0:
            composition = FiltersComposition(Slide)
        else:
            composition = FiltersComposition(Compose, *self.custom_filters)

        thumb_mask = composition.tissue_mask_filters(thumb)

        return thumb_mask
