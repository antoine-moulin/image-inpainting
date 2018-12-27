#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### SYSTEM IMPORTS ###
import sys
import os
import inspect

### ADDING PARENT FOLDER PATH ###
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataset_path = parent_folder_path + "/dataset"

if not parent_folder_path in sys.path :
    sys.path.append(parent_folder_path)
if not classes_path in sys.path :
    sys.path.append(classes_path)


### TOOLS IMPORTS ###
import pickle
import matplotlib.pyplot as plt
import pdb


### CLASS IMPORTS ###
import Test as InpaintingTest # to avoid confusion with other python build-in librairies
from useful_functions import viewimage



#################
### FUNCTIONS ###
#################



def plot_operations(data_folder_name, test_number,variable_parameter_name):

    L = get_operations_and_variation(data_folder_name, test_number,variable_parameter_name)
    x = L[0]
    y = L[1]
    
    figure = plt.figure()
    
    plt.xlim(22,3)    
    plt.plot(x,y)
    
    plt.xlabel('taille du côté du patch de calcul (en pixels)')
    plt.ylabel('nombre d\'operations')
    plt.title('Nbre d\'operations en fonction du patch de calcul')
    
    figure.savefig("graphe_operations.pdf", bbox_inches='tight')
    
    plt.show()


def plot_duration(data_folder_name, test_number,variable_parameter_name):

    L = get_duration_and_variation(data_folder_name, test_number,variable_parameter_name)
    x = L[0]
    y = L[1]
    
    figure = plt.figure()
    
    plt.ylim(31000000,2000000)    
    plt.plot(x,y)
    
    plt.xlabel('taille du côté du patch de calcul (en pixels)')
    plt.ylabel('durée (en sec)')
    plt.title('Durée de l\'execution en fonction du patch de calcul')
    
    figure.savefig("graphe_duree2.pdf", bbox_inches='tight')
    
    plt.show()



def get_operations_and_variation(data_folder_name, test_number,variable_parameter_name):
    
    results = get_results_for_test(data_folder_name, test_number)
    
    operations = []
    variations = []
    
    for result_dict in results :
        
        variations.append(result_dict[variable_parameter_name])
        operations.append(result_dict["results"]["number_of_operations"])
    
    return [variations, operations]



def get_duration_and_variation(data_folder_name, test_number,variable_parameter_name):
    
    results = get_results_for_test(data_folder_name, test_number)
    
    durations = []
    variations = []
    
    for result_dict in results :
        
        variations.append(result_dict[variable_parameter_name])
        durations.append(result_dict["results"]["total_duration"])
    
    return [variations, durations]



def get_description_for_test(data_folder_name, test_number):
    
    test = get_test_object(data_folder_name, test_number)
    
    return test.description


def visualize_images_results(data_folder_name, test_number):
    
    results_list = get_results_for_test(data_folder_name, test_number)
    
    for result in results_list :
        
        result_dict = result["results"]
        image = result_dict["final_image"]
        
        viewimage(image, normalise=False)



def get_results_for_test(data_folder_name, test_number):
    
    test = get_test_object(data_folder_name, test_number)
    
    return test.results



def get_test_object(data_folder_name, test_number):
    global data_folders_paths
    
    # find full path of the data folder
    found = False
    for i in range (len(data_folders_paths)):
        
        liste = data_folders_paths[i].split("/")
        name = liste[len(liste) - 1]
        
        if data_folder_name == name:
            if not found :
                full_path = str(data_folders_paths[i])
                found = True
            else :
                raise Exception("Error : 2 corresponding files found")
    
    pickle_file_full_path = full_path + "/tests/" + test_number
    
    with open(pickle_file_full_path, 'rb') as pickle_file :
        
        depickler = pickle.Unpickler(pickle_file)
        
        test_object = depickler.load()
        
        return test_object
        
    


#######################
### TOOLS FUNCTIONS ###
#######################


def get_all_data_folders_names():
    global dataset_path
    
    # get all folders of the dataset directory
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir() ]
    
    return subfolders



#############################
### INSPECTING PARAMETERS ###
#############################

data_folders_paths = get_all_data_folders_names()







description = get_description_for_test("data1","11_14_9_38_47")
results = get_results_for_test("data1","11_14_9_38_47")
"""
L = get_duration_and_variation("data1","11_14_9_38_47","computation_patch_size")
x = L[0]
y = L[1]
"""

plot_operations("data1","11_14_9_38_47","computation_patch_size")
#print("description = ",description)
#print(results)





