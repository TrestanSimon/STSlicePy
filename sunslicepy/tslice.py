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

        self.spatial_units = self.map_sequence[0].spatial_units

        if curve[0] is not u.Quantity:
            print('Warning')
            curve = [c * self.spatial_units[0] for c in curve]

        self.colormap = self.map_sequence[0].cmap

        cf = self.map_sequence[0].coordinate_frame
        curve_indices = [SkyCoord(pair[0], pair[1], frame=cf) for pair in curve]

        self.time = np.empty(frame_count, dtype=datetime.datetime)
        self.intensity = np.zeros((frame_count, len(curve_indices)), dtype=float)

        for frame_index in tqdm.tqdm(range(0, frame_count)):
            self.time[frame_index] = self.map_sequence[frame_index].date.datetime
            for i in range(0, len(curve_indices)):
                self.intensity[frame_index][i] = sunpy.map.sample_at_coords(self.map_sequence[frame_index], curve_indices[i]).value

        self.intensity = self.intensity.T

        """for frame_index in range(len(self._map_seq)):
            self.time.append(self._map_seq[frame_index].date.datetime)
            self._curve_coords.append(
                SkyCoord(
                    curve[0],
                    curve[1],
                    unit=(unit, unit),
                    frame=self._map_seq[frame_index].coordinate_frame
                )
            )
            self._intensity_coords.append(sunpy.map.pixelate_coord_path(
                self._map_seq[frame_index], self._curve_coords[frame_index]
            ))
            self._intensity.append(sunpy.map.sample_at_coords(
                self._map_seq[frame_index], self._intensity_coords[frame_index]
            ))
        print(self._curve_coords[0])
        self._intensity = np.transpose(self._intensity)
        self._curve = curve"""

    def peek(self):
        line_coords = np.arange(0, np.shape(self.intensity)[0])
        plt.pcolormesh(
            self.time, line_coords, self.intensity,
            cmap=self.colormap, norm='log'
        )
        plt.show()

    def running_difference(self):
        pass
