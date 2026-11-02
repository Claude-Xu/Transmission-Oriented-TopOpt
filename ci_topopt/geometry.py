"""Passive frame from the reference inverter geometry."""

import numpy as np


def passive_masks(coordinates, geometry):
    x, y = coordinates[:, :2].T * (10.0 / geometry["length"])
    if not geometry["passive_frame"]:
        return np.zeros(x.size, dtype=bool), np.zeros(x.size, dtype=bool)
    a = geometry["corner_size"] * 10.0 / geometry["length"]
    b = geometry["port_half_width"] * 10.0 / geometry["length"]
    width = geometry["frame_width"] * 10.0 / geometry["length"]
    radius = np.hypot(5.0, 5.0 + b / 2 - 7.7)
    inner, outer = max(0.0, radius - width)**2, (radius + width)**2
    solid = (((np.abs(x - 5) < b) & ((y <= 1) | (y >= 9)))
             | ((np.abs(y - 5) < b) & ((x <= 1) | (x >= 9)))
             | (((x <= a) | (x >= 10 - a)) & ((y <= a) | (y >= 10 - a))))
    void = np.zeros(x.size, dtype=bool)
    for cx, cy in ((-5, 7.7), (-5, 2.3), (15, 7.7), (15, 2.3),
                   (2.3, -5), (7.7, -5), (2.3, 15), (7.7, 15)):
        distance = (x - cx)**2 + (y - cy)**2
        solid |= (distance >= inner) & (distance <= outer)
        void |= distance < inner
    solid &= ~void
    return solid, void
