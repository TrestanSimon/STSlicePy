from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import convolve
from matplotlib import colors
from sunpy.coordinates import Helioprojective
from sunpy.map import MapSequence
from tqdm import tqdm
from sunslicepy.rasterize import bresenham_line, dda_line
from sunslicepy.slice import GenericSlice


class PointsRibbon(GenericSlice):
    def _get_observer(self, skycoords_input):
        return skycoords_input[0][0].observer

    def _get_slice(self, curve_skycoords) -> (np.ndarray, np.ndarray):
        intensity_cube = np.array([map_s.data for map_s in self.map_sequence])
        curve_len = len(curve_skycoords)
        width = len(curve_skycoords[0])
        curve_px = np.empty((self.frame_n, curve_len, 2), dtype=int)
        intensity = np.empty((self.frame_n, curve_len), dtype=float)

        for f in tqdm(range(self.frame_n), unit='frames'):
            xf, yf = self.map_sequence[f].world_to_pixel(curve_skycoords)
            xf, yf = np.round(xf), np.round(yf)
            for i in range(curve_len):
                xi, yi = xf[i].value, yf[i].value
                curve_px[f][i] = int(yi[0]), int(xi[0])
                xi = [int(p) for p in xi]
                yi = [int(p) for p in yi]
                int_sum = 0
                for j in range(width):
                    int_sum += intensity_cube[f][yi[j]][xi[j]]
                intensity[f][i] = int_sum / len(xi)
        return curve_px, intensity

    def _get_curve_ds(self, curve_skycoords) -> np.ndarray[u.Quantity]:
        curve_ds = np.empty(self.curve_len, dtype=u.Quantity)
        curve_ds[0] = 0 * u.arcsec
        for i in range(self.curve_len - 1):
            curve_ds[i + 1] = curve_ds[i] + curve_skycoords[i + 1][0].separation(curve_skycoords[i][0])
        return curve_ds
