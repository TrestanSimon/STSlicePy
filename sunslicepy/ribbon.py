import numpy as np
import astropy.units as u
from tqdm import tqdm
from sunslicepy.slice import GenericSlice


class PointsRibbon(GenericSlice):
    def observer(self):
        return self.skycoords_input[0][0].observer

    def _get_slice(self) -> (np.ndarray, np.ndarray):
        intensity_cube = np.array([map_s.data for map_s in self.map_sequence])
        curve_len = len(self.skycoords_input)
        curve_wid = len(self.skycoords_input[0])
        curve_px = np.empty((self.frame_n, curve_len, 2), dtype=int)
        intensity = np.empty((self.frame_n, curve_len), dtype=float)

        for f in tqdm(range(self.frame_n), unit='frames'):
            xf, yf = self.map_sequence[f].world_to_pixel(self.skycoords_input)
            xf, yf = np.round(xf), np.round(yf)
            for i in range(curve_len):
                xi = [int(p.value) for p in xf[i]]
                yi = [int(p.value) for p in yf[i]]
                curve_px[f][i] = yi[0], xi[0]
                int_sum = 0
                for j in range(curve_wid):
                    int_sum += intensity_cube[f][yi[j]][xi[j]]
                intensity[f][i] = int_sum / len(xi)
        return curve_px, intensity

    def _get_curve_ds(self) -> np.ndarray[u.Quantity]:
        curve_ds = np.empty(self.curve_len, dtype=u.Quantity)
        curve_ds[0] = 0 * u.arcsec
        for i in range(self.curve_len-1):
            curve_ds[i+1] = curve_ds[i] + self.skycoords_input[i+1][0].separation(self.skycoords_input[i][0])
        return curve_ds
