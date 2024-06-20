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
        self.curve_axis = np.empty(curve_n, dtype=u.Quantity)

        intensity_cube = np.array([map_s.data for map_s in self.map_sequence])

        curve_p = np.empty(curve_n, dtype=SkyCoord)
        curve_c = np.empty(curve_n, dtype=SkyCoord)
        curve_c_comp = np.empty((curve_n, 2), dtype=u.Quantity)

        for i in tqdm(range(curve_n)):
            # Helioprojective skycoords at t = 0
            curve_p[i] = SkyCoord(
                curve[i][0], curve[i][1],
                frame=Helioprojective(obstime=self.map_sequence[0].date, observer='earth')
            )
            # Transform to Carrington coordinates:
            # Carrington skycoords at t = 0
            curve_c[i] = curve_p[i].transform_to(
                HeliographicCarrington(obstime=self.map_sequence[0].date, observer='earth')
            )
            # Extract Carrington lon and lat
            curve_c_comp[i][0] = curve_c[i].lon
            curve_c_comp[i][1] = curve_c[i].lat

        curve_skycoord = np.empty((frame_n, curve_n), dtype=SkyCoord)
        curve_indices_raw = np.empty((curve_n, 2), dtype=u.Quantity)
        curve_indices = np.empty((frame_n, curve_n, 2), dtype=int)

        # Helioprojective skycoords corresponding to extracted Carrington lon and lat for all t
        for i in tqdm(range(curve_n)):
            for j in range(frame_n):
                curve_skycoord[j][i] = SkyCoord(
                    curve_c_comp[i][0], curve_c_comp[i][1],
                    frame=HeliographicCarrington(obstime=self.map_sequence[j].date, observer='earth')
                )

        # World to pixel
        for i in tqdm(range(curve_n)):
            for j in range(frame_n):
                curve_indices_raw[i][0], curve_indices_raw[i][1] = self.map_sequence[j].world_to_pixel(
                    curve_skycoord[j][i]
                    # curve_p[i]
                )
                curve_indices[j][i][1] = np.abs(curve_indices_raw[i][0].value)
                curve_indices[j][i][0] = np.abs(curve_indices_raw[i][1].value)

        # Calculate helioprojective increments
        self.curve_axis[0] = 0 * u.arcsec
        for i in range(curve_n - 1):
            self.curve_axis[i+1] = self.curve_axis[i] + curve_skycoord[0][i+1].separation(curve_skycoord[0][i])

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
