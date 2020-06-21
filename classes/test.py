#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from classes.image_inpainting import ImageInpainting


class Test:

    def __init__(self, test_number, description, image_parent_directory, fixed_parameters, variable_parameter):
        """
        Description of parameters:
            - global_test_number : number of this test (so it appears in the title of the file where the test object
              will be saved )
            - image_parent_directory : parent directory of the image on which the tests will be applied
            - fixed_parameters : a dictionnary that contains all parameters that will remain constant during the test
            --> scheme :  {'image':[[...]], 'mask':[[...]], 'is_color_image':True, 'patch_size':7,
                           'computation_patch_size':7}
            - variable_parameter : a dictionnary that contains the name of the parameter and its values (as a list)
            --> scheme :  {'name' : 'number_of_clusters', 'values' : [1,2,3,4,5,6,7,8,9,10]}
        """

        print("\n")
        print("########### TESTING IMAGE ", image_parent_directory, " ###########")
        print("\n")

        self.results = []

        # practical information
        self.global_test_number = test_number
        self.description = description
        self.image_parent_directory = image_parent_directory

        # (image, mask, is_color_image, patch_size, computation_patch_size, number_of_clusters)

        fixed_parameters_keys = list(fixed_parameters.keys())

        # determine the variable parameter

        if type(variable_parameter["values"]) == list:
            self.variable_parameter_values = variable_parameter["values"]
            self.number_of_tests = len(variable_parameter["values"])
        else:
            raise Exception("variable_parameter['values'] is a ",
                            type(variable_parameter["values"], ", however it should be a list."))

        self.variable_parameter_name = variable_parameter["name"]

        if self.variable_parameter_name == "image":
            self.image_values = self.variable_parameter_values
        elif self.variable_parameter_name == "mask":
            self.mask_values = self.variable_parameter_values
        elif self.variable_parameter_name == "is_color_image":
            self.is_color_image_values = self.variable_parameter_values
        elif self.variable_parameter_name == "patch_size":
            self.patch_size_values = self.variable_parameter_values
        elif self.variable_parameter_name == "computation_patch_size":
            self.computation_patch_size_values = self.variable_parameter_values
        elif self.variable_parameter_name == "which_clustering_method":
            self.which_clustering_method_values = self.variable_parameter_values
        elif self.variable_parameter_name == "number_of_clusters":
            self.number_of_clusters_values = self.variable_parameter_values
        else:
            raise Exception("The dictionnary of variable parameter contains an unknown parameter name: ",
                            self.variable_parameter_name)

        # determine the fixed parameters

        if len(fixed_parameters_keys) != 6:
            raise Exception(
                "6 fixed parameters and 1 variable parameter expected. Current number of fixed parameters: ",
                len(fixed_parameters_keys))

        for fixed_parameter in fixed_parameters_keys:

            if fixed_parameter == "image":
                self.image_values = [fixed_parameters["image"]] * self.number_of_tests
            elif fixed_parameter == "mask":
                self.mask_values = [fixed_parameters["mask"]] * self.number_of_tests
            elif fixed_parameter == "is_color_image":
                self.is_color_image_values = [fixed_parameters["is_color_image"]] * self.number_of_tests
            elif fixed_parameter == "patch_size":
                self.patch_size_values = [fixed_parameters["patch_size"]] * self.number_of_tests
            elif fixed_parameter == "computation_patch_size":
                self.computation_patch_size_values = [fixed_parameters["computation_patch_size"]] * (
                    self.number_of_tests)
            elif fixed_parameter == "which_clustering_method":
                self.which_clustering_method_values = [fixed_parameters["which_clustering_method"]] * (
                    self.number_of_tests)
            elif fixed_parameter == "number_of_clusters":
                self.number_of_clusters_values = [fixed_parameters["number_of_clusters"]] * self.number_of_tests
            else:
                raise Exception("The dictionnary of fixed parameters contains an unknown parameter name : ",
                                fixed_parameter)

        print("Fixed parameters   : ", fixed_parameters_keys)
        print("Variable parameter : ", self.variable_parameter_name)
        print("     is_color_image              =  ", self.is_color_image_values)
        print("     patch size                  =  ", self.patch_size_values)
        print("     computation patch size      =  ", self.computation_patch_size_values)
        print("     which_clustering_method     =  ", self.which_clustering_method_values)
        print("     number_of_clusters          =  ", self.number_of_clusters_values)

    def start_inpainting_test(self):

        self.results = []  # scheme: [dict1, dict2, ..., dictN]  and dict1 = {'number_of_clusters': 0, 'results': {...}}

        for i in range(self.number_of_tests):

            print("\nTEST No", i)
            print("With ", self.variable_parameter_name, " equal to ", self.variable_parameter_values[i])

            image = self.image_values[i]
            mask = self.mask_values[i]
            is_color_image = self.is_color_image_values[i]
            patch_size = self.patch_size_values[i]
            computation_patch_size = self.computation_patch_size_values[i]
            which_clustering_method = self.which_clustering_method_values[i]
            number_of_clusters = self.number_of_clusters_values[i]

            inpainting_image = ImageInpainting(image, mask, is_color_image, patch_size, computation_patch_size,
                                               which_clustering_method, number_of_clusters, True)

            new_result = inpainting_image.start_filling_with_data(display_frequency="None")
            result_dict = {self.variable_parameter_name: self.variable_parameter_values[i], "results": new_result}
            self.results.append(result_dict)

            print("RESULTS :")
            for key in list(result_dict["results"].keys()):
                if not key == "final_image":
                    print("   ", key, " : ", result_dict["results"][key])

    def store_test_in_textfile(self):

        file_name = self.image_parent_directory + "/tests/" + str(self.global_test_number)

        with open(file_name, 'wb') as file:
            pickler = pickle.Pickler(file)
            pickler.dump(self)
