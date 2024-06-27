import numpy as np


def bresenham_line(x0, y0, x1, y1) -> np.ndarray:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    curve_px = np.empty((dx + 1, 2), dtype=int)

    reciprocal = False
    if dy > dx:
        reciprocal = True
        dx, dy = dy, dx
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    D = 2*dy - dx
    x = np.empty(dx + 1, dtype=int)
    y = np.empty(dx + 1, dtype=int)
    xi, yi = x0, y0
    for i in range(dx):
        if D > 0:
            yi += 1 if yi < y1 else -1
            D += 2*(dy - dx)
        else:
            D += 2*dy
        xi += 1 if xi < x1 else -1
        x[i], y[i] = xi, yi
    x[-1] = x1
    y[-1] = y1

    if reciprocal:
        x, y = y, x
    for i in range(dx + 1):
        curve_px[i] = x[i], y[i]
    return curve_px


def dda_line(x0, y0, x1, y1) -> np.ndarray:
    curve_px = None

    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    dxi = dx / steps
    dyi = dy / steps
    curve_len = steps + 1

    xi, yi = float(x0), float(y0)
    x = np.array([xi + i * dxi for i in range(curve_len)])
    y = np.array([yi + i * dyi for i in range(curve_len)])

    if curve_px is None:
        curve_px = np.empty((curve_len, 2), dtype=int)

    for i in range(curve_len):
        curve_px[i] = int(x[i]), int(y[i])
    return curve_px
