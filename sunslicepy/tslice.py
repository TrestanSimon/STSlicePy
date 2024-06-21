from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective
from sunpy.map import MapSequence
from tqdm import tqdm


class GenericSlice(ABC):
    def __init__(
            self,
            seq_input: MapSequence,
            skycoords_input: list[SkyCoord] or np.ndarray[SkyCoord],
    ):
        self.map_sequence = seq_input
        self.map_sequence.all_maps_same_shape()
        self.frame_n = len(self.map_sequence)
        self.time = [smap.date.datetime for smap in self.map_sequence]

        self.spatial_units = self.map_sequence[0].spatial_units
        self.colormap = self.map_sequence[0].cmap

        # Necessary when points are not on disk
        with Helioprojective.assume_spherical_screen(
                center=skycoords_input[0].observer,
                only_off_disk=True
        ):
            self.curve_px, self.intensity = self._get_slice(skycoords_input)
            self.curve_len = len(self.curve_px[0])
            self.curve_ds = self._get_curve_ds(skycoords_input)

    @abstractmethod
    def _get_slice(self, curve_skycoords) -> (np.ndarray, np.ndarray):
        """"""

    @abstractmethod
    def _get_curve_ds(self, curve_skycoords) -> np.ndarray:
        """"""

    def peek(self, norm='log'):
        plt.pcolormesh(
            self.time, [ds.value for ds in self.curve_ds], self.intensity.T,
            cmap=self.colormap, norm=norm
        )
        plt.show()

    @property
    def running_difference(self):
        difference = np.empty((self.frame_n-1, self.curve_len), dtype=float)
        for f in range(self.frame_n-1):
            for i in range(self.curve_len):
                difference[f][i] = self.intensity[f+1][i] - self.intensity[f][i]
        return difference


class PointsSlice(GenericSlice):
    def _get_slice(self, curve_skycoords):
        intensity_cube = np.array([map_s.data for map_s in self.map_sequence])
        curve_len = len(curve_skycoords)
        curve_px = np.empty((self.frame_n, curve_len, 2), dtype=int)
        intensity = np.empty((self.frame_n, curve_len), dtype=float)

        for f in tqdm(range(self.frame_n), unit='frames'):
            xf, yf = self.map_sequence[f].world_to_pixel(curve_skycoords)
            xf, yf = np.round(xf), np.round(yf)
            for i in range(curve_len):
                xi, yi = int(xf[i].value), int(yf[i].value)
                curve_px[f][i] = yi, xi
                intensity[f][i] = intensity_cube[f][yi][xi]

        return curve_px, intensity

    def _get_curve_ds(self, curve_skycoords):
        # Calculate distances along curve
        curve_ds = np.empty(self.curve_len, dtype=u.Quantity)
        curve_ds[0] = 0 * u.arcsec
        for i in range(self.curve_len - 1):
            curve_ds[i + 1] = curve_ds[i] + curve_skycoords[i + 1].separation(curve_skycoords[i])
        return curve_ds


class BreSlice(GenericSlice):
    def _get_slice(self, curve_skycoords):
        intensity_cube = np.array([map_s.data for map_s in self.map_sequence])
        intensity = None
        curve_px = None
        coords_n = len(curve_skycoords)  # 2

        for f in tqdm(range(self.frame_n), unit='frames'):
            xp, yp = self.map_sequence[f].world_to_pixel(curve_skycoords)
            coords = np.abs(np.array([
                [int(xi.value) for xi in xp],
                [int(yi.value) for yi in yp]]).T)
            for i in range(coords_n):
                coords[i] = int(np.round(coords[i][0])), int(np.round(coords[i][1]))

            x0, y0 = coords[0]
            x1, y1 = coords[1]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)

            reciprocal = False
            if dy > dx:
                reciprocal = True
                dx, dy = dy, dx
                x0, y0 = y0, x0
                x1, y1 = y1, x1

            D = 2*dy - dx
            x = np.empty(dx, dtype=int)
            y = np.empty(dx, dtype=int)
            xi, yi = x0, y0
            for i in range(dx):
                if D > 0:
                    yi += 1 if yi < y1 else -1
                    D += 2*(dy - dx)
                else:
                    D += 2*dy
                xi += 1 if xi < x1 else -1
                x[i], y[i] = xi, yi

            if reciprocal:
                x, y = y, x
            if curve_px is None:
                curve_px = np.empty((self.frame_n, dx, 2), dtype=int)
            if intensity is None:
                intensity = np.empty((self.frame_n, dx), dtype=float)

            for i in range(dx):
                curve_px[f][i] = y[i], x[i]
                intensity[f][i] = intensity_cube[f][y[i]][x[i]]
        return curve_px, intensity

    def _get_curve_ds(self, curve_skycoords):
        curve_ds = np.empty(self.curve_len, dtype=u.Quantity)
        curve_ds[0] = 0 * u.arcsec

        intensity_coords = self.map_sequence[0].pixel_to_world(*(self.curve_px[0].T * u.pix))
        for i in range(self.curve_len - 1):
            curve_ds[i + 1] = curve_ds[i] + intensity_coords[i + 1].separation(intensity_coords[i])
        return curve_ds
