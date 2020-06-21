#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class PixelInpainting:
    """
    Object that depicts basic information of a pixel.
    """

    def __init__(self, i, j, value, confidence, in_margin_crown=False, in_computation_margin_crown=False):

        self.i, self.j = i, j  # line, column
        self.value = value
        self.confidence = confidence
        self.data = 0
        self.priority = 0
        self.in_margin_crown = in_margin_crown
        self.in_computation_margin_crown = in_computation_margin_crown
        self.in_crown = self.in_margin_crown * self.in_computation_margin_crown

    def get_4neighbours_values(self, max_i, max_j):
        coords = [
            [self.i, self.j - 1],  # left
            [self.i - 1, self.j],  # top
            [self.i, self.j + 1],  # right
            [self.i + 1, self.j]  # bottom
        ]

        for k, coord in enumerate(coords):  # exclude pixels out of the matrix
            if not (0 <= coord[0] <= max_i and 0 <= coord[1] <= max_j):
                del coords[k]

        return coords

    def compute_priority(self):
        self.priority = self.confidence * self.data

    def get_8neighbours_values(self, max_i, max_j):
        coords = [
            [self.i - 1, self.j - 1],  # top-left
            [self.i - 1, self.j],  # top
            [self.i - 1, self.j + 1],  # top-right
            [self.i, self.j - 1],  # left
            [self.i, self.j + 1],  # right
            [self.i + 1, self.j - 1],  # bottom-left
            [self.i + 1, self.j],  # bottom
            [self.i + 1, self.j + 1]  # bottom-right
        ]

        for k, coord in enumerate(coords):  # exclude pixels out of the matrix
            if 0 <= coord[0] <= max_i and 0 <= coord[1] <= max_j:
                del coords[k]

        return coords

    # getters
    def get_confidence(self): return self.confidence
    def get_i(self): return self.i
    def get_j(self): return self.j
    def get_value(self): return self.value
    def get_priority(self): return self.priority
    def get_in_margin_crown(self): return self.in_margin_crown
    def get_in_computation_margin_crown(self): return self.in_computation_margin_crown
    def is_in_crown(self): return self.in_crown

    # setters
    def set_confidence(self, new_confidence):
        self.confidence = new_confidence
        self.compute_priority()

    def set_data(self, new_data):
        self.data = new_data
        self.compute_priority()

    def set_value(self, new_value):
        self.value = new_value
