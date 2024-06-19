import datetime
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicCarrington, Helioprojective
from sunpy.map import MapSequence
import matplotlib.pyplot as plt
from tqdm import tqdm


class Slice:
    def __init__(
            self,
            seq: MapSequence,
            curve: list or np.ndarray,
    ):
        self.map_sequence = seq
        self.map_sequence.all_maps_same_shape()
        
        frame_n = len(self.map_sequence)
        curve_n = len(curve)

        self.spatial_units = self.map_sequence[0].spatial_units
        self.colormap = self.map_sequence[0].cmap

        self.time = np.empty(frame_n, dtype=datetime.datetime)
        self.intensity = np.zeros((frame_n, curve_n), dtype=float)

        self.arc_axis = np.empty(curve_n, dtype=u.Quantity)
        
        intensity_array = np.array([map_s.data for map_s in self.map_sequence])

        # Curve helioprojective
        curve_p = np.empty(curve_n, dtype=SkyCoord)
        # Curve Carrington
        curve_c = np.empty(curve_n, dtype=SkyCoord)
        # Curve Carrington components
        curve_c_comp = np.empty((curve_n, 2), dtype=u.Quantity)
        
        for i in tqdm(range(curve_n)):
            curve_p[i] = SkyCoord(
                curve[i][0], curve[i][1],
                frame=Helioprojective(obstime=self.map_sequence[0].date, observer='earth')
            )
            curve_c[i] = curve_p[i].transform_to(
                HeliographicCarrington(obstime=self.map_sequence[0].date, observer='earth')
            )
            curve_c_comp[i][0] = curve_c[i].lon
            curve_c_comp[i][1] = curve_c[i].lat

        curve_skycoord = np.empty((frame_n, curve_n), dtype=SkyCoord)
        curve_indices_raw = np.empty((curve_n, 2), dtype=u.Quantity)
        curve_indices = np.empty((frame_n, curve_n, 2), dtype=int)

        for i in tqdm(range(curve_n)):
            for j in range(frame_n):
                curve_skycoord[j][i] = SkyCoord(
                    curve_c_comp[i][0], curve_c_comp[i][1],
                    frame=HeliographicCarrington(obstime=self.map_sequence[j].date, observer='earth')
                )

        for i in tqdm(range(curve_n)):
            for j in range(frame_n):
                curve_indices_raw[i][0], curve_indices_raw[i][1] = self.map_sequence[j].world_to_pixel(
                    curve_skycoord[j][i]
                )
                curve_indices[j][i][1] = np.abs(curve_indices_raw[i][0].value)
                curve_indices[j][i][0] = np.abs(curve_indices_raw[i][1].value)

        # Calculate helioprojective increments
        self.arc_axis[0] = 0 * u.arcsec
        for i in range(curve_n - 1):
            self.arc_axis[i+1] = self.arc_axis[i] + curve_p[i+1].separation(curve_p[i])

        for j in range(0, frame_n):
            self.time[j] = self.map_sequence[j].date.datetime
            for i in range(curve_n):
                x0 = int(np.floor(curve_indices[j][i][0]))
                y0 = int(np.floor(curve_indices[j][i][1]))
                """
                x1 = int(np.ceil(curve_indices[j][i][0]))
                y1 = int(np.ceil(curve_indices[j][i][1]))
                dx = curve_indices[j][i][0] - x0
                dy = curve_indices[j][i][1] - y0
                ds = np.sqrt(dx**2 + dy**2)"""

                self.intensity[j][i] = intensity_array[j][x0][y0]

        self.intensity = self.intensity.T

    def peek(self, norm='log'):
        plt.pcolormesh(
            self.time, [ds.value for ds in self.arc_axis], self.intensity,
            cmap=self.colormap, norm=norm
        )
        plt.show()

    def running_difference(self):
        pass
