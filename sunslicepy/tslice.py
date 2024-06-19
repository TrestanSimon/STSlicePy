import datetime

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map

import tqdm

import matplotlib
import matplotlib.pyplot as plt


class Slice:
    def __init__(
            self,
            seq: sunpy.map.MapSequence,
            curve: list,
    ):
        self.map_sequence = seq
        self.map_sequence.all_maps_same_shape()
        frame_count = len(self.map_sequence)
        data_slab = [smap.data for smap in seq]

        self.spatial_units = self.map_sequence[0].spatial_units

        if curve[0] is not u.Quantity:
            print('Warning')
            curve = [c * self.spatial_units[0] for c in curve]

        self.colormap = self.map_sequence[0].cmap

        cf = self.map_sequence[0].coordinate_frame

        curve_len = len(curve)

        curve_skycoord = np.empty(curve_len, dtype=SkyCoord)
        curve_indices_raw = np.empty((curve_len, 2), dtype=u.Quantity)
        curve_indices = np.empty((curve_len, 2), dtype=int)

        for i in range(curve_len):
            curve_skycoord[i] = SkyCoord(curve[i][0], curve[i][1], frame=cf)
            curve_indices_raw[i][0], curve_indices_raw[i][1] = self.map_sequence[0].world_to_pixel(
                curve_skycoord[i]
            )
            curve_indices[i][0] = int(np.abs(np.floor(curve_indices_raw[i][0].value)))
            curve_indices[i][1] = int(np.abs(np.floor(curve_indices_raw[i][1].value)))

        self.arc_axis = np.empty(curve_len, dtype=u.Quantity)
        self.arc_axis[0] = 0 * u.arcsec
        for i in range(curve_len - 1):
            self.arc_axis[i+1] = self.arc_axis[i] + curve_skycoord[i+1].separation(curve_skycoord[i])

        print(self.arc_axis)
        self.time = np.empty(frame_count, dtype=datetime.datetime)
        self.intensity = np.zeros((frame_count, len(curve_indices)), dtype=float)

        for frame_index in tqdm.tqdm(range(0, frame_count)):
            self.time[frame_index] = self.map_sequence[frame_index].date.datetime
            for i in range(len(curve_indices)):
                self.intensity[frame_index][i] = data_slab[frame_index][curve_indices[i][0]][curve_indices[i][1]]

        self.intensity = self.intensity.T

    def peek(self):
        line_coords = np.arange(0, np.shape(self.intensity)[0])
        plt.pcolormesh(
            self.time, [ds.value for ds in self.arc_axis], self.intensity,
            cmap=self.colormap, norm='log'
        )
        plt.show()

    def running_difference(self):
        pass
