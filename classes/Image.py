#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### SYSTEM IMPORTS ###
import sys
import os
import inspect


### ADDING PARENT FOLDER PATH ###
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # class directory path
project_path = parent_folder_path + "/.."

if not parent_folder_path in sys.path :
    sys.path.append(parent_folder_path)
if not project_path in sys.path :
    sys.path.append(project_path)


### TOOLS IMPORTS ###
import numpy as np

from scipy.ndimage import convolve
from scipy.ndimage import zoom
from scipy.misc import imsave
import copy

from sklearn.cluster import KMeans

import warnings
import time


""" DEBUGGING """
import pdb
import cProfile
import re


### CLASS IMPORTS ###
from Pixel import Pixel
from useful_functions import viewimage



class Image :
    """ Object that depicts basic information of an image """
    
    def __init__(self, image, mask, is_color_image, patch_size, computation_patch_size, which_optimization_method, number_of_clusters, use_threshold) :
        
        
        # basic data
        self.image = image
        self.mask = mask
        self.is_color_image = is_color_image
        
        self.nb_lines = image.shape[0]
        self.nb_columns = image.shape[1]
        empty_matrix = []
        
        # width and height of patches (computed now to save operations)
        self.width_patch  = patch_size
        self.height_patch = patch_size
        self.width_computation_patch  = computation_patch_size
        self.height_computation_patch = computation_patch_size
        
        # width and height margins (computed now to save operations)
        self.width_margin  = int((self.width_patch - 1)/2)
        self.height_margin = int((self.height_patch - 1)/2)
        self.computation_height_margin = int((self.height_computation_patch - 1)/2)
        self.computation_width_margin  = int((self.width_computation_patch - 1)/2)
        
        # patch size (computed now to save operations)
        self.patch_size = (self.height_patch)*(self.width_patch)
        
        # computation patch size (computed now to save operations)
        self.computation_patch_size = (self.height_computation_patch)*(self.width_computation_patch)
        
        
        self.priorities = []
        
        self.operations = 0
        self.count = 0
        
        self.min_distance = 3*(255**2)*max(self.width_patch, self.width_computation_patch)*max(self.height_patch, self.height_computation_patch) # compute maximal distance between patches
        self.count += 3

        # related to the computations of the distances (optimization)
        self.use_threshold = use_threshold
        
        
        # optimization method attributes
        if (which_optimization_method == 1 or which_optimization_method == 2) : # clustering methods 1 or 2 (on pixels or on patches)
            self.use_clustering = True
            self.clustering_method = which_optimization_method
            self.nb_clusters = number_of_clusters
        else : # research mask method
            self.use_clustering = False
        
        # attributes for restricted patch search area
        
        first_line = np.where(mask == 1)[0][0]
        last_line = np.where(mask == 1)[0][-1]
        height_range = last_line - first_line
        search_height_margin = np.ceil(0.5 * height_range)
        
        self.first_line_search = int(max(first_line - search_height_margin, self.height_margin))
        self.last_line_search = int(min(last_line + search_height_margin, self.nb_lines - self.height_margin))
        
        first_column = np.where(mask.transpose() == 1)[0][0]
        last_column = np.where(mask.transpose() == 1)[0][-1]
        width_range = last_column - first_column 
        search_width_margin = np.ceil(0.5 * width_range)
        
        self.first_column_search = int(max(first_column - search_width_margin, self.width_margin))
        self.last_column_search = int(min(last_column + search_width_margin, self.nb_columns - self.width_margin))
        
        """ 
        # attributes for restricted patch search area
        self.zoom_factor = 2 # 200% of the original mask
        
        # make sure the research mask is not too small
        first_line = np.where(self.mask == 1)[0][0]
        last_line = np.where(self.mask == 1)[0][-1]
        first_column = np.where(self.mask.transpose() == 1)[0][0]
        last_column = np.where(self.mask.transpose() == 1)[0][-1]
        
        height_range = last_line - first_line
        width_range = last_column - first_column
        if (height_range < 50) or (width_range < 50) :
            self.zoom_factor = 1 + 50/min(height_range/2, width_range/2)
        
        # we zoom the initial mask and then we clip in order to get back to the initial shape
        self.research_mask = zoom(self.mask, self.zoom_factor)
        self.research_mask_height = self.research_mask.shape[0]
        self.research_mask_width = self.research_mask.shape[1]

        height_reduction = (self.research_mask_height - image.shape[0])//2
        width_reduction = (self.research_mask_width - image.shape[1])//2

        if ((self.research_mask_height - image.shape[0])%2 == 0) :
            self.research_mask = self.research_mask[height_reduction:(self.research_mask_height - height_reduction), :]
        else :
            self.research_mask = self.research_mask[(height_reduction + 1):(self.research_mask_height - height_reduction), :]

        if ((self.research_mask_width - image.shape[1])%2 == 0) :
            self.research_mask = self.research_mask[:, width_reduction:(self.research_mask_width - width_reduction)]
        else :
            self.research_mask = self.research_mask[:, (width_reduction + 1):(self.research_mask_width - width_reduction)]
        """
        
        
        # build all pixels objects and store them in pixels_matrix
        
        ### FIRST CASE : FILLING PATCH SMALLER THAN COMPUTATION PATCH ###
        if self.height_patch < self.height_computation_patch :
        
            for i in range (self.nb_lines) :
                new_line = []
                for j in range (self.nb_columns) :
                    confidence = 1 - self.mask[i, j] # mask[i,j]=1 => pixel blanc => inconnu => confidence=0
                    value = self.image[i, j]*confidence
                    
                    if   (0 <= i < self.height_margin) or (self.nb_lines - self.height_margin <= i < self.nb_lines) or (0 <= j < self.width_margin) or (self.nb_columns - self.width_margin <= j < self.nb_columns):
                        new_line.append(Pixel(i, j, value, confidence, in_margin_crown = True, in_computation_margin_crown=True))
                    
                    elif (self.height_margin <= i < self.computation_height_margin) or (self.nb_lines - self.computation_height_margin <= i < self.nb_lines - self.height_margin) or (self.width_margin <= j < self.computation_width_margin) or (self.nb_columns - self.computation_width_margin <= j < self.nb_columns - self.width_margin):
                        new_line.append(Pixel(i, j, value, confidence, in_computation_margin_crown=True))
                    
                    else :
                        new_line.append(Pixel(i, j, value, confidence))
                
                empty_matrix.append(new_line)
        
        
        ### SECOND CASE : FILLING PATCH BIGGER THAN COMPUTATION PATCH ###
        elif self.height_patch > self.height_computation_patch :
            
            for i in range (self.nb_lines) :
                new_line = []
                for j in range (self.nb_columns) :
                    confidence = 1 - self.mask[i, j] # mask[i,j]=1 => pixel blanc => inconnu => confidence=0
                    value = self.image[i, j]*confidence
                
                    if   (0 <= i < self.computation_height_margin) or (self.nb_lines - self.computation_height_margin <= i < self.nb_lines) or (0 <= j < self.computation_width_margin) or (self.nb_columns - self.computation_width_margin <= j < self.nb_columns):
                        new_line.append(Pixel(i, j, value, confidence, in_margin_crown = True, in_computation_margin_crown=True))
                    
                    elif (self.computation_height_margin <= i < self.height_margin) or (self.nb_lines - self.height_margin <= i < self.nb_lines - self.computation_height_margin) or (self.computation_width_margin <= j < self.width_margin) or (self.nb_columns - self.width_margin <= j < self.nb_columns - self.computation_width_margin):
                        new_line.append(Pixel(i, j, value, confidence, in_computation_margin_crown=True))
                    
                    else :
                        new_line.append(Pixel(i, j, value, confidence))
            
                empty_matrix.append(new_line)
        
        
        ### THIRD CASE : COMPUTATION PATCH AND FILLIN PATCH HAVE THE SAME SIZE ###
        else :
            
            for i in range (self.nb_lines) :
                new_line = []
                for j in range (self.nb_columns) :
                    confidence = 1 - self.mask[i, j] # mask[i,j]=1 => pixel blanc => inconnu => confidence=0
                    value = self.image[i, j]*confidence
                
                    if   (0 <= i < self.computation_height_margin) or (self.nb_lines - self.computation_height_margin <= i < self.nb_lines) or (0 <= j < self.computation_width_margin) or (self.nb_columns - self.computation_width_margin <= j < self.nb_columns):
                        new_line.append(Pixel(i, j, value, confidence, in_margin_crown = True, in_computation_margin_crown=True))
                    
                    else :
                        new_line.append(Pixel(i, j, value, confidence))
                    
                empty_matrix.append(new_line)
        
        
        self.pixels_matrix = np.matrix(empty_matrix)
        
        
        ### BUILD THE PATCHES MATRICES ###
        # we store all patches in a matrix of lists in order to avoid operation wastes
        patches_matrix = []
        computation_patches_matrix = []
        
        print("DEBUT DE LA PHASE DE CALCUL DES PATCHS")
        for i in range (self.nb_lines):
            new_line_filling_patch = []
            new_line_computation_patch = []
            for j in range (self.nb_columns) :
                pixel = self.pixels_matrix[i,j]
                filling_patch = self.compute_patch(pixel, shorten = False)
                computation_patch = self.compute_patch(pixel, shorten = True)
                new_line_filling_patch.append(filling_patch)
                new_line_computation_patch.append(computation_patch)
                
            patches_matrix.append(new_line_filling_patch)
            computation_patches_matrix.append(new_line_computation_patch)
        
        self.patches_matrix = np.array(patches_matrix)
        self.computation_patches_matrix = np.array(computation_patches_matrix)
        print("FIN DE LA PHASE DE CALCUL DES PATCHS")

        ### COMPUTE THE FILLFRONT ###
        
        self.fillFront_pixels = []
        self.compute_fillFront()
    
    

    def start_filling(self, display_frequency):
        """ start procedure to fill the target zone """
        self.display_frequency = display_frequency
        display_result_count = 0
        
        if self.use_clustering :
            
            if self.clustering_method == 1 :
                self.data = self.image.reshape(self.image.shape[0]*self.image.shape[1], 3)
            
            elif self.clustering_method == 2 :
                list_result = self.get_vectorized_patches_list()
                self.data = list_result[0]
                patches_coordinates = list_result[1]
            
            
            self.clusters = KMeans(n_clusters = self.nb_clusters, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(self.data)
            self.centroids = self.clusters.cluster_centers_
            
            """
            if self.clustering_method == 2 :
                centroids_copy = self.centroids.deepcopy()
            else :
                centroids_copy = self.centroids.copy()
            
            self.reshaped_centroids_list = []
            
            for centroid in centroids_copy : # NB : the centroid is flat !
                        # reshape the the centroid
                        reshaped_centroid = centroid.reshape(self.height_patch, self.width_patch, 3)
                        self.reshaped_centroids_list.append(reshaped_centroid)
            """
            
            if self.clustering_method == 1 :
                self.clusters_labels = self.clusters.labels_.reshape(self.image.shape[0], self.image.shape[1])
            
            elif self.clustering_method == 2 :
                
                # initializing label values to "nan"
                self.clusters_labels = np.zeros((self.image.shape[0],self.image.shape[1]), dtype=int)
                for i in range (self.image.shape[0]):
                    for j in range (self.image.shape[1]):
                        self.clusters_labels[i,j] = np.nan
                
                # set the real label of the patch at each coordinates
                for index in range (len(patches_coordinates)):
                    
                    label = self.clusters.labels_[index]
                    coordinates = patches_coordinates[index]
                    i_index = coordinates[0]
                    j_index = coordinates[1]
                    self.clusters_labels[i_index,j_index] = label
        
        else :
        
            while len(self.fillFront_pixels) > 0 : # while the fillFront is not empty
                
                display_result_count += 1
                self.count = 0
                
                # step 1 : compute priorities
                priorities_list = self.compute_priorities()
                
                # step 2 : find highest priority pixel
                highest_priority_pixel = self.find_highest_priority_pixel(priorities_list)[0] # [0] because we take only the pixel, not the tuple
                
                # step 3 : fill the patch AND update confidences
                self.fill_patch(highest_priority_pixel)
                
                # step 4 : compute new fillfront
                self.compute_fillFront()
                
                # display step result
                if not (type(self.display_frequency) == str) : # meaning its value is not "only at the end" or "None"
                    if (display_result_count % self.display_frequency) == 0 :
                        self.display_step_result(True)
                
                self.operations += self.count
            
            if not (self.display_frequency == "None"):
                self.display_step_result(True)
    
    
    
    
    def start_filling_with_data(self, display_frequency):
        """ start procedure to fill the target zone and return statistical data """
        self.display_frequency = display_frequency
        display_result_count = 0
        
        # notice starting time
        starting_time = time.time()
        
        
        if self.use_clustering :
            
            if self.clustering_method == 1 :
                self.data = self.image.reshape(self.image.shape[0]*self.image.shape[1], 3)
            
            elif self.clustering_method == 2 :
                list_result = self.get_vectorized_patches_list()
                self.data = list_result[0]
                patches_coordinates = list_result[1]
            
            print("START CLUSTERING")
            self.clusters = KMeans(n_clusters = self.nb_clusters, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(self.data)
            self.centroids = self.clusters.cluster_centers_
            print("END CLUSTERING")
            
            """
            if self.clustering_method == 2 :
                
                centroids_copy = copy.deepcopy(self.centroids)
                self.reshaped_centroids_list = []
                for centroid in centroids_copy : # NB : le centroid est à plat !
                            # reshape the the centroid
                            reshaped_centroid = centroid.reshape(self.height_patch, self.width_patch, 3)
                            self.reshaped_centroids_list.append(reshaped_centroid)
            
            else :
                centroids_copy = copy.deepcopy(self.centroids)
            """
            
            if self.clustering_method == 1 :
                self.clusters_labels = self.clusters.labels_.reshape(self.image.shape[0], self.image.shape[1])
            
            elif self.clustering_method == 2 :
                
                # initializing label values to "nan"
                self.clusters_labels = np.empty((self.image.shape[0],self.image.shape[1]), dtype=float)
                self.clusters_labels[:] = np.nan
                
                # set the label of the patch at each coordinate
                for index in range (len(patches_coordinates)):
                    
                    label = self.clusters.labels_[index]
                    coordinates = patches_coordinates[index]
                    i_index = coordinates[0]
                    j_index = coordinates[1]
                    self.clusters_labels[i_index,j_index] = label
        
        
        
        while len(self.fillFront_pixels) > 0 : # while the fillFront is not empty
            
            display_result_count += 1
            self.count = 0
            
            # step 1 : compute priorities
            priorities_list = self.compute_priorities()
            
            # step 2 : find highest priority pixel
            highest_priority_pixel = self.find_highest_priority_pixel(priorities_list)[0] # [0] because we take only the pixel, not the tuple
            
            # step 3 : fill the patch AND update confidences
            self.fill_patch(highest_priority_pixel)
            
            # step 4 : compute new fillfront
            self.compute_fillFront()
            
            # display step result
            if not (type(self.display_frequency) == str) : # meaning its value is not "only at the end" or "None"
                if (display_result_count % self.display_frequency) == 0 :
                    self.display_step_result(True)
            
            self.operations += self.count
            
        if not (self.display_frequency == "None"):
            self.display_step_result(True)
        
        # notice ending time
        ending_time = time.time()
        total_duration = ending_time - starting_time
        
        ### results dictionnary ###
        self.display_step_result(view=False)
        size_of_image = (self.image.shape[0])*(self.image.shape[1])
        data_dict = {"final_image" : self.image, "total_duration" : total_duration, "size_of_image" : size_of_image, "number_of_operations" : self.operations}
        
        return data_dict # AJOUTER LES DONNEES DU CLUSTERING
    
    
    
    def compute_priorities(self) :
        """ compute priorities for all pixels in the fillFront """
        
        priorities = []
        
        for pixel in self.fillFront_pixels :
            priority = self.compute_priority(pixel)
            priorities.append([pixel,priority])
            
            self.count += 1
        
        return priorities
    
    
    
    def find_highest_priority_pixel(self, priorities_list):
        """ returns a tuple that contains the fillFront pixel holding the highest priority """
        
        def takeSecond(elem):
            return elem[1]
        
        sorted_priorities = sorted(priorities_list, reverse=True, key=takeSecond)
        best_tuple  = sorted_priorities[0]
        
        self.count += 1
        
        return best_tuple
    
    
    
    def fill_patch(self, pixel):
        """ fill the patch which center is the given parameter """

        main_patch = self.get_patch(pixel, shorten=False)[0]
        main_computation_patch = self.get_patch(pixel, shorten=True)[0]
                
        min_patch_pixel = self.pixels_matrix[self.height_margin,self.width_margin]
        
        minimum_distance = copy.copy(self.min_distance)
        new_distance = copy.copy(self.min_distance)

        
        #search for the pixel which patch minimizes the distance with the target patch
        if self.use_clustering :
            
            cluster_number = self.find_cluster_number(pixel)
            
            for i in range (self.height_margin, self.nb_lines - self.height_margin):
                for j in range (self.width_margin, self.nb_columns - self.width_margin):
                    
                    if self.clusters_labels[i, j] == cluster_number :
                        
                        new_pixel = self.pixels_matrix[i,j]
                        
                        # full patch
                        patch_and_boolean = self.get_patch(new_pixel, shorten=False)
                        is_target_zone = patch_and_boolean[1]
                        
                        # shortened patch, just for computations
                        computation_patch_and_boolean = self.get_patch(new_pixel, shorten=True)
                        new_computation_patch = computation_patch_and_boolean[0]
                        
                        if not is_target_zone :
                            new_distance = self.compute_distance(main_computation_patch, new_computation_patch, new_distance)
                            if new_distance < minimum_distance :
                                minimum_distance = new_distance
                                min_patch_pixel = new_pixel

        
        else :
            """
            for i in range (self.height_margin, self.nb_lines - self.height_margin) :
                for j in range (self.width_margin, self.nb_columns - self.width_margin) :
                    
                    if self.research_mask[i, j] == 1 : """ # with the second method for the research_mask
            
            for i in range (self.first_line_search, self.last_line_search):
                for j in range (self.first_column_search, self.last_column_search):
                    new_pixel = self.pixels_matrix[i,j]
                    
                    # full patch
                    patch_and_boolean = self.get_patch(new_pixel, shorten=False)
                    is_target_zone = patch_and_boolean[1]
                    
                    # shortened patch, just for computations
                    computation_patch_and_boolean = self.get_patch(new_pixel, shorten=True)
                    new_computation_patch = computation_patch_and_boolean[0]
                    
                    if not is_target_zone :
                        new_distance = self.compute_distance(main_computation_patch, new_computation_patch, new_distance)
                        if new_distance < minimum_distance :
                            minimum_distance = new_distance
                            min_patch_pixel = new_pixel
        
        
        # transfer values from the best patch found
        self.transfer_values(pixel, min_patch_pixel)

    
    
    def compute_fillFront(self) :
        """ search for pixels of the fillFront """
        
        ## STEP 1 : Search for the first pixel of the FillFront
        
        found = False # states if the first pixel of the fillfront has been found
        i = 0
        while ((not found) and (i < self.nb_lines)) :
            for j in range (self.nb_columns) :
                confidence = self.pixels_matrix[i,j].get_confidence()
                if (confidence == 0) and (not found) : # hidden part of the picture
                    start_pixel = self.pixels_matrix[i, j]
                    found = True
            i += 1
        
        if not found :
            self.fillFront_pixels = []
            return 0
        
        
        # STEP 2 : Search for pixels of the FillFront
        
        considered_pixels = []
        self.fillFront_pixels = []
        self.fillFront_pixels = self.search_fillFront_pixels(start_pixel, self.fillFront_pixels, considered_pixels)
        
    
    
    def display_step_result(self,view):
        """ display image with highlighted fillfront """
        
        # update image
        for i in range (self.nb_lines) :
            for j in range (self.nb_columns) :
                self.image[i,j] = self.pixels_matrix[i,j].get_value()
        
        # highlight the fillFront in white
        new_image = self.image.copy()
        for pixel in self.fillFront_pixels :
            i_pixel = pixel.get_i()
            j_pixel = pixel.get_j()
            new_image[i_pixel,j_pixel] = np.float32(255)
        
        if view :
            # display image
            viewimage(new_image,normalise=False)
        
        
        for i in range (self.nb_lines) :
            for j in range (self.nb_columns) :
                self.image[i,j] = self.pixels_matrix[i,j].get_value()
        
    
    
    ####################
    ### TOOLS METHODS ###
    ####################
    
    
    def get_vectorized_patches_list(self):
        """ returns a list containing all vectorized patches of the image """
        
        vectorized_patches = []
        patches_coordinates = []
        
        add_patch_to_list = vectorized_patches.append
        add_coordinates_to_list = patches_coordinates.append
        
        for i in range (self.height_margin, self.nb_lines - self.height_margin):
            for j in range (self.width_margin, self.nb_columns - self.width_margin):
                
                pixel = self.pixels_matrix[i,j]
                
                List = self.get_patch(pixel)
                patches_list = List[0]
                is_target_zone = List[1]
                
                if not is_target_zone :
                    vectorized_patch = self.vectorize_patch(patches_list)
                    coordinates = [pixel.get_i(), pixel.get_j()]
                    add_patch_to_list(vectorized_patch)
                    add_coordinates_to_list(coordinates)
                    
        return [vectorized_patches, patches_coordinates]
    
    
    def vectorize_patch(self, patch) :
        """ turns a patch into a vector following this scheme :
            [ pixel0_value0, pixel0_value1, pixel0_value2, pixel1_value0, pixel1_value1, pixel1_value2, ..., pixeln_value0, pixeln_value1, pixeln_value2 ]
            with value0, value1 and value2 corresponding to the 3 values of a pixel in RGB space
            """
        
        patch_size = len(patch)
        vectorized_patch = np.zeros(patch_size*3)
        
        index = 0
        
        for pixel in patch :
            
            if pixel == "out of bound with the image" :
                vectorized_patch[index]   = np.nan
                vectorized_patch[index+1] = np.nan
                vectorized_patch[index+2] = np.nan
                index += 3
            else :
                values = pixel.get_value()
                vectorized_patch[index]   = values[0]
                vectorized_patch[index+1] = values[1]
                vectorized_patch[index+2] = values[2]
        
        return vectorized_patch
    
    
    
    def compute_priority(self, pixel) :
        
        confidence_term = self.compute_confidence(pixel)
        data_term = self.compute_data(pixel)
        
        priority = confidence_term * data_term
        self.count += 1
        
        return priority
    
    
    def compute_confidence(self, central_pixel):
        """ compute confidence of the central pixel of the patch """
        
        i_target = central_pixel.get_i()
        j_target = central_pixel.get_j()
        
        new_confidence = 0
        
        for k in range(-self.height_margin, self.height_margin + 1) :
            for l in range(-self.width_margin, self.width_margin + 1) :
                if ((0 <= i_target + k < self.nb_lines) and (0 <= j_target + l < self.nb_columns)) :
                    current_pixel = self.pixels_matrix[i_target + k, j_target + l]
                    new_confidence += current_pixel.get_confidence()
        
        new_confidence = float(new_confidence)/(float(self.width_patch)*float(self.height_patch))
        self.count += 2
        
        return new_confidence
    
    
    def compute_data(self, pixel) :
        """ compute the data on linear structures around the pixel """
        
        pixel_coords = [pixel.get_i(), pixel.get_j()]
        
        # normalization factor
        alpha = 255
        
        # computation of the tangent vector
        k = self.fillFront_pixels.index(pixel) # returns the position of pixel in fillFront_pixels
        previous_neighbour_coords = [self.fillFront_pixels[k-1].get_i(), self.fillFront_pixels[k-1].get_j()]
        next_neighbour_coords = [self.fillFront_pixels[(k+1)%len(self.fillFront_pixels)].get_i(), self.fillFront_pixels[(k+1)%len(self.fillFront_pixels)].get_j()]
        
        tangent = [n - p for n, p in zip(next_neighbour_coords, previous_neighbour_coords)]
        tangent_norm = (tangent[0]**2 + tangent[1]**2)**0.5
        
        if tangent_norm == 0 :
            return 1
        
        tangent[0] = tangent[0]/tangent_norm
        tangent[1] = tangent[1]/tangent_norm
        
        self.count += 8
        
        # computation of the gradient
        
        # if we are on the edge of the image
        a = max(pixel_coords[0] - self.height_margin, 0)
        b = min(pixel_coords[0] + self.height_margin+1, self.nb_lines)
        c = max(pixel_coords[1] - self.width_margin, 0)
        d = min(pixel_coords[1] + self.width_margin+1, self.nb_columns)
        patch = self.pixels_matrix[a:b, c:d]
        
        if self.is_color_image : # we build 3 gradients (one for each color) and sum them
            
            temp0 = np.zeros_like(patch)
            temp1 = np.zeros_like(patch)
            temp2 = np.zeros_like(patch)
            
            for i in range(patch.shape[0]) :
                for j in range(patch.shape[1]) :
                    
                    if (patch[i, j].get_confidence() == 0) :
                        temp0[i, j] = np.nan
                        temp1[i, j] = np.nan
                        temp2[i, j] = np.nan
                    else :
                        temp0[i, j] = patch[i, j].get_value()[0]
                        temp1[i, j] = patch[i, j].get_value()[1]
                        temp2[i, j] = patch[i, j].get_value()[2]
            
            grad_y0, grad_x0 = np.gradient(temp0, axis=(0, 1))
            grad_y1, grad_x1 = np.gradient(temp1, axis=(0, 1))
            grad_y2, grad_x2 = np.gradient(temp2, axis=(0, 1))
            
            grad_y = (grad_y0 + grad_y1 + grad_y2)/3
            grad_x = (grad_x0 + grad_x1 + grad_x2)/3
            
            self.count += 5
            
        else :
            
            temp = np.zeros_like(patch)
            
            for i in range(patch.shape[0]) :
                for j in range(patch.shape[1]) :
                
                    if (patch[i, j].get_confidence() == 0) :
                        temp[i, j] = np.nan
                    else :
                        temp[i, j] = patch[i, j].get_value()
            
            grad_y, grad_x = np.gradient(temp, axis=(0, 1))
            
            self.count += 1
        
        
        grad_mod = np.zeros_like(grad_y)
        
        for i in range(grad_mod.shape[0]) :
            for j in range(grad_mod.shape[1]) :
                
                if ( not(np.isnan(grad_x[i, j])) and not(np.isnan(grad_y[i, j])) ) :
                    grad_mod[i, j] = np.sqrt(grad_x[i, j]**2 + grad_y[i, j]**2)
                    self.count += 3
                else :
                    grad_mod[i, j] = -1
        
        k = np.nanargmax(np.ravel(grad_mod), axis=0) # we keep the gradient with the highest module
        self.count += 1
        
        grad = np.array([np.ravel(grad_y)[k], np.ravel(grad_x)[k]])
        
        # computation of the data term
        data_term = abs((grad[0]*tangent[0] + grad[1]*tangent[1]))/alpha
        self.count += 3
        
        return data_term
    
    
    def find_cluster_number(self, pixel):
        """ returns the number of cluster of the pixel (which value is unknown), taking into account the pixels of its patch """
        
        patch_list = self.get_patch(pixel, shorten=False)[0] # get the patch
        
        if self.clustering_method == 1 :
        
            # compute the mean pixel of the patch
            mean = 0
            nb_of_filled_pixels = 0
            for i in range (len(patch_list)):
                current_pixel = patch_list[i]
                if not (current_pixel == "out of bound with the image"):
                    if current_pixel.get_confidence() > 0 : # checking that it is not an unfilled pixel
                        mean += current_pixel.get_value()
                        nb_of_filled_pixels += 1
            
            mean = mean/nb_of_filled_pixels
            self.count += 1
            
            #mean = list(mean)
            list_mean = [mean] # format adapted to kmeans.predict() function
            
            cluster_number = self.clusters.predict(list_mean)[0]
        
        
        elif self.clustering_method == 2 :
            
            # get the vectorized patch of this pixel
            vectorized_patch = self.vectorize_patch(patch_list)
            
            nb_centroids = len(self.centroids)
            minimum_distance = copy.copy(self.min_distance)
            
            for index in range (nb_centroids):
                
                centroid = self.centroids[index]
                distance = self.compute_distance_patch_centroid(vectorized_patch, centroid)
                
                if distance < minimum_distance :
                    
                    minimum_distance = distance
                    cluster_number = copy.copy(index)
            
            """
            patch_list
            self.reshaped_centroids_list
            centroids_copy = self.centroids.deepcopy()
            pixel_patch_vectorized = []
            
            for pixel in patch_list :
                confidence = pixel
            
            reshaped_centroids_copy = self.reshaped_centroids_list.deepcopy()
            centroids_with_nan = []
            
            # build a matrix from the pixel
            for index in range(len(patch_list)) :
                
                pixel = patch_list[index]
                
                if pixel == "out of bound with the image" :
                    centroids_copy
                
            
            # commencer par déterminer les coordonnees manquantes
            for pixel in patch_list :
                
                if pixel.get_confidence() == 0 :
                    pass
            """
        
        return cluster_number
    
    
    
    def compute_distance_patch_centroid(self, vectorized_patch, centroid):
        
        distance = 0
        for index in range(len(vectorized_patch)):
            value1 = vectorized_patch[index]
            value2 = vectorized_patch[index]
            
            if not np.isnan(value1) and not np.isnan(value2):
                distance += (value2 - value1)**2
        return distance
    
    
    
    def get_patch(self,pixel, shorten=False):
        
        i_pixel = pixel.get_i()
        j_pixel = pixel.get_j()
        
        if shorten :
            return self.computation_patches_matrix[i_pixel,j_pixel]
        else :
            return self.patches_matrix[i_pixel,j_pixel]
    
    
    
    def compute_patch(self,pixel, shorten=False):
        """ returns a list [patch_list, is_target_zone] containing : 
            - a list of all pixels of the patch which center is the parameter
            - a boolean that states if there are "target zone" pixels in the patch
            
            The "shorten" parameter states if the size of the patch must be the usual one or the "short" one."""
        
        i_center = pixel.get_i()
        j_center = pixel.get_j()
        is_target_zone = False # boolean to check if there is an unknown pixel
        
        if shorten :
                        
            patch_list = ["out of bound with the image"]*self.computation_patch_size
            
            i_begin = i_center - self.computation_height_margin
            i_end   = i_center + self.computation_height_margin + 1
            j_begin = j_center - self.computation_width_margin
            j_end   = j_center + self.computation_width_margin + 1
        
        
        else :
            
            patch_list = ["out of bound with the image"]*self.patch_size
            
            i_begin = i_center - self.height_margin
            i_end   = i_center + self.height_margin + 1
            j_begin = j_center - self.width_margin
            j_end   = j_center + self.width_margin + 1
        
        
        if not pixel.is_in_crown() : # the pixel is not on the border margin
            
            index = 0
            for i in range (i_begin, i_end):
                for j in range (j_begin, j_end):
                    
                    current_pixel = self.pixels_matrix[i,j]
                    patch_list[index] = current_pixel
                    
                    if not is_target_zone :
                        if self.mask[i,j] == 1 :
                            is_target_zone = True
                    
                    index += 1
        else :
            
            index = 0
            for i in range (i_begin, i_end):
                for j in range (j_begin, j_end):
                    
                    if (i < self.nb_lines and i >= 0 and j < self.nb_columns and j >= 0):
                        current_pixel = self.pixels_matrix[i,j]
                        patch_list[index] = current_pixel
                        if self.mask[i,j] == 1 :
                            is_target_zone = True
                        
                    index += 1

        
        # return the list of pixels + the boolean stating if there is an unknown pixel
        return [patch_list, is_target_zone]
    
        
    
    def transfer_values(self, target_patch_pixel, source_patch_pixel):
        """ transfer values of pixels from the source patch to the target patch """
        
        i_target = target_patch_pixel.get_i()
        j_target = target_patch_pixel.get_j()
        
        i_source = source_patch_pixel.get_i()
        j_source = source_patch_pixel.get_j()
        
        new_confidence = self.compute_confidence(target_patch_pixel)
        
        # transfer values
        for i in range(-self.height_margin, self.height_margin + 1):
            for j in range(-self.width_margin, self.width_margin + 1):
                if (0 <= i_target + i < self.nb_lines) and (0 <= j_target + j < self.nb_columns) and (0 <= i_source + i < self.nb_lines) and (0 <= j_source + j < self.nb_columns):
                    if (self.pixels_matrix[i_target + i, j_target + j].get_confidence() == 0): # fill only blank pixels
                        new_value = self.pixels_matrix[i_source + i, j_source + j].get_value() # get value at the source pixel
                        
                        # update matrix
                        self.pixels_matrix[i_target + i, j_target + j].set_value(new_value) # update value in the matrix
                        self.pixels_matrix[i_target + i, j_target + j].set_confidence(new_confidence) # update confidence in the matrix
                                            
                        # update image
                        self.image[i,j] = new_value
    
    
    
    def compute_distance(self, patch1, patch2, threshold):
        """ Computes euclidian distance between two patches and returns the minimum of this distance and a threshold, in order to reduce the number of operations """
        distance = 0
        
        if (len(patch1) != len(patch2)):
            raise Exception("Both patches don't have the same length (", len(patch1), " vs ", len(patch2), " ).")
        
        else :
            
            if self.is_color_image : # first case : color image
            
                for i in range (len(patch1)) :
                    
                    pixel1 = patch1[i]
                    pixel2 = patch2[i]
                    
                    if not(pixel1 == "out of bound with the image" or pixel2 == "out of bound with the image"):
                        if (pixel1.get_confidence() != 0) and (pixel2.get_confidence() != 0) :
                            gap0 = (pixel2.get_value()[0] - pixel1.get_value()[0])**2
                            gap1 = (pixel2.get_value()[1] - pixel1.get_value()[1])**2
                            gap2 = (pixel2.get_value()[2] - pixel1.get_value()[2])**2
                            gap = gap0 + gap1 + gap2
                            distance += gap
                            
                            self.count += 3
                        
                        if (distance > threshold) and self.use_threshold :
                            return threshold
            
            else : # second case : black and white image
                
                for i in range (len(patch1)) :
                    
                    pixel1 = patch1[i]
                    pixel2 = patch2[i]
                    
                    if not(pixel1 == "out of bound with the image" or pixel2 == "out of bound with the image"):
                        if (pixel1.get_confidence() != 0) and (pixel2.get_confidence() != 0) :
                            gap = (pixel2.get_value() - pixel1.get_value())**2
                            distance += gap
                            
                            self.count += 1
                        
                        if (distance > threshold) and self.use_threshold :
                            return threshold
                
           
        return distance
    
    
    
    def search_fillFront_pixels(self, pixel, fillFront_pixels, considered_pixels) :
        """ recursive function that searchs fillFront pixels in the neighbourhood of the parameter """
        
        if not(pixel in fillFront_pixels) and not(pixel in considered_pixels) and self.is_fillFront(pixel) :
            fillFront_pixels.append(pixel)
            considered_pixels.append(pixel)
            neighbours = self.get_4neighbours(pixel)
            for neighbour in neighbours :
                fillFront_pixels = self.search_fillFront_pixels(neighbour, fillFront_pixels, considered_pixels)
        
        return fillFront_pixels
    
    
    
    def is_fillFront(self, pixel) :
        """ returns boolean True if the pixel is in the fillfront """
        
        if (pixel.get_confidence() != 0): # not in the fillFront if its confidence is not zero
            return False
        else :
            neighbours = pixel.get_8neighbours_values(self.nb_lines-1, self.nb_columns-1)
            for neighbour in neighbours :
                if (self.pixels_matrix[neighbour[0], neighbour[1]].get_confidence() != 0) :
                    return True
            return False
    
    
    def get_4neighbours(self, pixel):
        """ returns top, bottom, left and right pixels next to the pixel given as parameter """
        
        pixel_neighbours = []
        values = pixel.get_4neighbours_values(self.nb_lines - 1, self.nb_columns - 1)
        
        for coordinates in values :
            i = coordinates[0]
            j = coordinates[1]
            pixel = self.pixels_matrix[i,j]
            pixel_neighbours.append(pixel)
        
        return pixel_neighbours
    
    
    def get_pixel(self, i, j) :
        return self.pixels_matrix[i,j]