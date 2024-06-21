import datetime
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import Helioprojective
from sunpy.map import MapSequence
import matplotlib.pyplot as plt
from tqdm import tqdm


class Slice:
    def __init__(
            self,
            seq_input: MapSequence,
            curve_input: list or np.ndarray,
    ):
        self.map_sequence = seq_input
        self.map_sequence.all_maps_same_shape()

        self.frame_n = len(self.map_sequence)
        self.curve_n = len(curve_input)

        self.spatial_units = self.map_sequence[0].spatial_units
        self.colormap = self.map_sequence[0].cmap

        self.time = [smap.date.datetime for smap in self.map_sequence]
        self.curve_px = np.empty((self.frame_n, self.curve_n, 2), dtype=float)
        self.curve_ds = np.empty(self.curve_n, dtype=u.Quantity)
        self.intensity = np.zeros((self.frame_n, self.curve_n), dtype=float)

        intensity_cube = np.array([map_s.data for map_s in self.map_sequence])

        curve_components = curve_input.T
        curve_skycoords = SkyCoord(
            curve_components[0], curve_components[1],
            frame=Helioprojective(obstime=self.map_sequence[0].date, observer='earth')
        )

        # Necessary when points are not on disk
        with (Helioprojective.assume_spherical_screen(
                center=curve_skycoords[0].observer,
                only_off_disk=True
        )):
            # World to pixel by frame
            for f in tqdm(range(self.frame_n), unit='frames'):
                x_p, y_p = self.map_sequence[f].world_to_pixel(curve_skycoords)
                coords = np.abs(np.array([x_p.value, y_p.value]).T)
                for i in range(self.curve_n):
                    self.curve_px[f][i][1], self.curve_px[f][i][0] = coords[i]

            # Calculate distances along curve
            self.curve_ds[0] = 0 * u.arcsec
            for i in range(self.curve_n-1):
                self.curve_ds[i+1] = self.curve_ds[i] + curve_skycoords[i+1].separation(curve_skycoords[i])

        self._line_draw(intensity_cube)
        self.intensity = self.intensity.T

    def _line_draw(self, data_cube):
        for f in range(self.frame_n):
            for i in range(self.curve_n):
                x_i = int(np.floor(self.curve_px[f][i][0]))
                y_i = int(np.floor(self.curve_px[f][i][1]))
                self.intensity[f][i] = data_cube[f][x_i][y_i]

    def peek(self, norm='log'):
        plt.pcolormesh(
            self.time, [ds.value for ds in self.curve_ds], self.intensity,
            cmap=self.colormap, norm=norm
        )
        plt.show()

    def running_difference(self):
        pass
