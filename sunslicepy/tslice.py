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
        self.intensity = np.zeros((self.frame_n, self.curve_n), dtype=float)
        self.curve_axis = np.empty(self.curve_n, dtype=u.Quantity)

        intensity_cube = np.array([map_s.data for map_s in self.map_sequence])

        coords = curve_input.T

        curve_p = SkyCoord(
            coords[0], coords[1],
            frame=Helioprojective(obstime=self.map_sequence[0].date, observer='earth')
        )

        curve_i_raw = np.empty(2, dtype=u.Quantity)
        curve_i = np.empty((self.frame_n, self.curve_n, 2), dtype=int)

        with (Helioprojective.assume_spherical_screen(
                center=curve_p[0].observer,
                only_off_disk=True
        )):
            for j in tqdm(range(self.frame_n)):
                curve_i_raw[0], curve_i_raw[1] = self.map_sequence[j].world_to_pixel(
                    curve_p
                )
                curve_i_raw = [np.abs(row.value) for row in curve_i_raw]
                for i in range(self.curve_n):
                    curve_i[j][i][1] = curve_i_raw[0][i]
                    curve_i[j][i][0] = curve_i_raw[1][i]

            self.curve_axis[0] = 0 * u.arcsec
            for i in range(self.curve_n-1):
                self.curve_axis[i+1] = self.curve_axis[i] + curve_p[i+1].separation(curve_p[i])

        for j in range(self.frame_n):
            for i in range(self.curve_n):
                x0 = int(np.floor(curve_i[j][i][0]))
                y0 = int(np.floor(curve_i[j][i][1]))
                self.intensity[j][i] = intensity_cube[j][x0][y0]

        self.intensity = self.intensity.T

    def peek(self, norm='log'):
        plt.pcolormesh(
            self.time, [ds.value for ds in self.curve_axis], self.intensity,
            cmap=self.colormap, norm=norm
        )
        plt.show()

    def running_difference(self):
        pass
