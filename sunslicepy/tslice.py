import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map


class Slice:
    def __init__(self, map_array, curve):
        self._map_seq = sunpy.map.Map(map_array, sequence=True)
        self._map_seq.all_maps_same_shape()
        # Sort sequence by time

        self._curve_coords = []
        self._intensity_coords = []
        self._intensity = []
        for map_index in range(len(self._map_seq)):
            self._curve_coords.append(
                SkyCoord(
                    curve[0],
                    curve[1],
                    unit=(u.arcsec, u.arcsec),
                    frame=self._map_seq[map_index].coordinate_frame
                )
            )
            self._intensity_coords.append(sunpy.map.pixelate_coord_path(
                self._map_seq[map_index], self._curve_coords[map_index]
            ))
            self._intensity.append(sunpy.map.sample_at_coords(
                self._map_seq[map_index], self._intensity_coords[map_index]
            ))
        self._curve = curve

    def plot(self):
        """For testing"""
        plt.imshow(self._intensity)
        plt.show()

    def running_difference(self):
        pass
