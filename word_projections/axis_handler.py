import numpy as np

from .errors import AxisNotFound400

class AxisHandler:
    def __init__(self, model, axis_defs):
        self.axes = {}

        for axis in axis_defs:
            word_pairs = axis_defs[axis]
            all_axis_vecs = [model.get_word_vec(wp[0]) - model.get_word_vec(wp[1]) for wp in word_pairs]
            self.axes[axis] = np.average(all_axis_vecs, axis=0)

    def list_axes(self):
        return list(self.axes.keys())

    def get_axis_vec(self, axis):
        if axis not in self.axes:
            raise AxisNotFound400

        return self.axes[axis]
