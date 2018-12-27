#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### SYSTEM IMPORTS ###
import sys
import os
import inspect

### ADDING PARENT FOLDER PATH ###
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
classes_path = parent_folder_path + "/classes"
if not parent_folder_path in sys.path :
    sys.path.append(parent_folder_path)
if not classes_path in sys.path :
    sys.path.append(classes_path)




class Pixel :
    """ Object that depicts basic information of a pixel """
    
    def __init__(self, i, j, value, confidence, in_margin_crown=False, in_computation_margin_crown=False):
        
        self.i = i # Line number in the matrix
        self.j = j # Column number in the matrix
        self.value = value
        self.confidence = confidence
        self.data = 0
        self.priority = 0
        self.in_margin_crown = in_margin_crown
        self.in_computation_margin_crown = in_computation_margin_crown
        self.in_crown = (self.in_margin_crown)*(self.in_computation_margin_crown)


    def get_4neighbours_values(self, max_i_coordinate, max_j_coordinate):
        
        neighbours_values = []
        
        coordinate1 = [self.i, self.j - 1] # left
        coordinate2 = [self.i - 1, self.j] # top
        coordinate3 = [self.i, self.j + 1] # right
        coordinate4 = [self.i + 1, self.j] # bottom
        coordinates = [coordinate1, coordinate2, coordinate3, coordinate4]
        
        for coordinate in coordinates :
            # exclude pixels out of the matrix
            if (coordinate[0] >= 0 and coordinate[1] >= 0) and (coordinate[0] <= max_i_coordinate and coordinate[1] <= max_j_coordinate) :
                neighbours_values.append(coordinate)
        
        return neighbours_values
    
    
    def compute_priority(self):
        self.priority = self.confidence * self.data
    
    
    def get_8neighbours_values(self, max_i_coordinate, max_j_coordinate):
        
        neighbours_values = []
        
        coordinate1 = [self.i - 1, self.j - 1] # top-left
        coordinate2 = [self.i - 1, self.j    ] # top
        coordinate3 = [self.i - 1, self.j + 1] # top-right
        coordinate4 = [self.i    , self.j - 1] # left
        coordinate5 = [self.i    , self.j + 1] # right
        coordinate6 = [self.i + 1, self.j - 1] # bottom-left
        coordinate7 = [self.i + 1, self.j    ] # bottom
        coordinate8 = [self.i + 1, self.j + 1] # bottom-right
        coordinates = [coordinate1, coordinate2, coordinate3, coordinate4, coordinate5, coordinate6, coordinate7, coordinate8]
        
        for coordinate in coordinates :
            # exclude pixels out of the matrix
            if (coordinate[0] >= 0 and coordinate[1] >= 0) and (coordinate[0] <= max_i_coordinate and coordinate[1] <= max_j_coordinate) :
                neighbours_values.append(coordinate)
        
        return neighbours_values
    
    
    # GETTERS #
    def get_confidence(self) :
        return self.confidence
    
    def get_i(self) :
        return self.i
    
    def get_j(self) :
        return self.j
    
    def get_value(self):
        return self.value
    
    def get_priority(self):
        return self.priority
    
    def get_in_margin_crown(self):
        return self.in_margin_crown
    
    def get_in_computation_margin_crown(self):
        return self.in_computation_margin_crown
    
    def is_in_crown(self):
        return self.in_crown
    
    
    
    # SETTERS #
    
    def set_confidence(self, new_confidence) :
        self.confidence = new_confidence
        self.compute_priority()
    
    def set_data(self, new_data):
        self.data = new_data
        self.compute_priority()
    
    def set_value(self, new_value):
        self.value = new_value
    