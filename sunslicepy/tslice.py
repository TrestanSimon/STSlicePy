import datetime

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington, Helioprojective
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import sunpy.map

import tqdm

import matplotlib.pyplot as plt


class Slice:
    def __init__(
            self,
            seq: sunpy.map.MapSequence,
            curve: list or np.ndarray,
    ):
        self.map_sequence = seq
        self.map_sequence.all_maps_same_shape()
        frame_count = len(self.map_sequence)
        data_slab = [smap.data for smap in self.map_sequence]

        self.spatial_units = self.map_sequence[0].spatial_units
        self.colormap = self.map_sequence[0].cmap

        curve_len = len(curve)

        curve_p = np.empty(curve_len, dtype=SkyCoord)
        curve_c = np.empty(curve_len, dtype=SkyCoord)
        curve_c_comp = np.empty((curve_len, 2), dtype=u.Quantity)
        for i in tqdm.tqdm(range(curve_len)):
            curve_p[i] = SkyCoord(
                curve[i][0], curve[i][1],
                frame=Helioprojective(obstime=self.map_sequence[0].date, observer='earth')
            )
            curve_c[i] = curve_p[i].transform_to(
                HeliographicCarrington(obstime=self.map_sequence[0].date, observer='earth')
            )
            curve_c_comp[i][0] = curve_c[i].lon
            curve_c_comp[i][1] = curve_c[i].lat

        curve_skycoord = np.empty((frame_count, curve_len), dtype=SkyCoord)
        curve_indices_raw = np.empty((curve_len, 2), dtype=u.Quantity)
        curve_indices = np.empty((frame_count, curve_len, 2), dtype=int)

        for i in tqdm.tqdm(range(curve_len)):
            for j in range(frame_count):
                curve_skycoord[j][i] = SkyCoord(
                    curve_c_comp[i][0], curve_c_comp[i][1],
                    frame=HeliographicCarrington(obstime=self.map_sequence[j].date, observer='earth')
                )

        for i in tqdm.tqdm(range(curve_len)):
            for j in range(frame_count):
                curve_indices_raw[i][0], curve_indices_raw[i][1] = self.map_sequence[j].world_to_pixel(
                    curve_skycoord[j][i]
                )
                curve_indices[j][i][1] = np.abs(curve_indices_raw[i][0].value)
                curve_indices[j][i][0] = np.abs(curve_indices_raw[i][1].value)

        self.arc_axis = np.empty(curve_len, dtype=u.Quantity)
        self.arc_axis[0] = 0 * u.arcsec
        for i in range(curve_len - 1):
            self.arc_axis[i+1] = self.arc_axis[i] + curve_p[i+1].separation(curve_p[i])

        self.time = np.empty(frame_count, dtype=datetime.datetime)
        self.intensity = np.zeros((frame_count, curve_len), dtype=float)

        for j in range(0, frame_count):
            self.time[j] = self.map_sequence[j].date.datetime
            for i in range(curve_len):
                x0 = int(np.floor(curve_indices[j][i][0]))
                #x1 = int(np.ceil(curve_indices[j][i][0]))
                y0 = int(np.floor(curve_indices[j][i][1]))
                """
                y1 = int(np.ceil(curve_indices[j][i][1]))
                dx = curve_indices[j][i][0] - x0
                dy = curve_indices[j][i][1] - y0
                ds = np.sqrt(dx**2 + dy**2)"""

                # self.intensity[j][i] = (1 - ds)*data_slab[j][x0][y0] + ds*data_slab[j][x1][y1]
                self.intensity[j][i] = data_slab[j][x0][y0]

        self.intensity = self.intensity.T

    def peek(self):
        plt.pcolormesh(
            self.time, [ds.value for ds in self.arc_axis], self.intensity,
            cmap=self.colormap, norm='log'
        )
        plt.show()

    def running_difference(self):
        pass
