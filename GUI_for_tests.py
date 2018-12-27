#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###############
### IMPORTS ###
###############


### SYSTEM IMPORTS ###
import sys
import os
import inspect


### ADDING PARENT FOLDER PATH ###
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
classes_path = parent_folder_path + "/classes"
images_path = parent_folder_path + "/images"
dataset_path = parent_folder_path + "/dataset"
if not parent_folder_path in sys.path :
    sys.path.append(parent_folder_path)
if not classes_path in sys.path :
    sys.path.append(classes_path)


### CLASS IMPORTS ###
import Image as ImageProject # to avoid confusion with other python build-in librairies
import Test as Test



### TOOLS IMPORTS ###
import numpy as np
import platform
import tempfile
import os
from scipy import ndimage as ndi
from skimage import io as skio
import cv2
import datetime
import pdb


### GUI IMPORTS ###
from tkinter import *
from tkinter import filedialog
from tkinter import ttk


### CONVENTIONS ###
canvas_width = 800
canvas_height = 650


### BUILD THE ROOT WINDOW ###

master = Tk() # must be done at the beginning so as to initialize specific variables such as IntVar() objects



#########################
### GLOBAL VARIABLES ####
#########################


# intvar objects
all_dataset_intvar = IntVar()
all_dataset_intvar.set(1)

# stringvar objects
dataset_directory_stringvar = StringVar()
dataset_directory_stringvar.set("default : "+ dataset_path)
variable_parameter_stringvar = StringVar()
variable_parameter_stringvar.set("number of clusters")
starting_value_stringvar = StringVar()
ending_value_stringvar = StringVar()
step_value_stringvar = StringVar()
number_of_clusters_stringvar = StringVar()
number_of_clusters_stringvar.set("1")
size_of_filling_patch_stringvar = StringVar()
size_of_filling_patch_stringvar.set("5")
size_of_computation_patch_stringvar = StringVar()
size_of_computation_patch_stringvar.set("5")
which_clustering_method_stringvar = StringVar()
which_clustering_method_stringvar.set("method 2 : on patches")
which_clustering_method = 2
description_stringvar = StringVar()
description_stringvar.set("")

# booleans
is_all_dataset = True
already_displayed  = False
already_displayed2 = False
already_displayed3 = False

# strings
dataset_directory = dataset_path

# other parameters
all_images_data = [] # list following this scheme : [[image0, mask0, parent_path0], [image1, mask1, parent_path1], ..., [imagen, maskn, parent_pathn]]
variable_parameter = "number_of_clusters"
starting_value = 3
ending_value = 9
step_value = 2
number_of_clusters = 1
size_of_filling_patch = 5
size_of_computation_patch = 5
which_clustering_method = 2
description = ""
test_number = "uncomputed"
values_variable_parameter = [3,5,7]


#################
### FUNCTIONS ###
#################


### INIT FUNCTIONS ###
def load_data():
    """ stock all images in variables """
    global dataset_directory, all_images_data
    
    # get all folders of the dataset directory
    subfolders = [f.path for f in os.scandir(dataset_directory) if f.is_dir() ]
    
    for folder in subfolders:
        
        subfiles_list = os.listdir(folder)
        if not (len(subfiles_list) == 0):
            image_name1 = "image.jpg"
            image_name2 = "image.jpeg"
            mask_name1 = "mask.jpg"
            mask_name2 = "mask.jpeg"
            
            if (image_name1 in subfiles_list) :
                is_image_found = True
                loaded_image = skio.imread(folder+"/"+image_name1, as_gray=False)
            elif (image_name2 in subfiles_list) :
                is_image_found = True
                loaded_image = skio.imread(folder+"/"+image_name2, as_gray=False)
            else :
                is_image_found = False
                print("WARNING : no file 'image.jpg' or 'image.jpeg' could be found in directory '", folder, "'")
                
            if (mask_name1 in subfiles_list) :
                is_mask_found = True
                loaded_mask = skio.imread(folder+"/"+mask_name1, as_gray=False)
            elif (mask_name2 in subfiles_list) :
                is_mask_found = True
                loaded_mask = skio.imread(folder+"/"+mask_name2, as_gray=False)
            else :
                is_mask_found = False
                print("WARNING : no file 'mask.jpg' or 'mask.jpeg' could be found in directory '", folder, "'")
            
            if is_image_found and is_mask_found :
                loaded_image = np.array(loaded_image)
                loaded_mask = np.array(loaded_mask)
                imfloat= np.float32(loaded_image)
                maskfloat = np.float32(loaded_mask)
                built_mask = maskfloat
                
                # process to avoid intermediate values in the mask
                for i in range (built_mask.shape[0]):
                    for j in range (built_mask.shape[1]):
                        if built_mask[i,j]<127:
                            built_mask[i,j] = 0
                        else :
                            built_mask[i,j] = 1
                
                all_images_data.append([folder, imfloat, built_mask])
                


### ENABLE FUNCTIONS ###

def enable_all_dataset():
    global all_dataset_intvar, is_all_dataset
    if all_dataset_intvar.get()==1:
        is_all_dataset=True
    else :
        is_all_dataset=False


### OTHER FUNCTIONS ###


def start_test():
    global all_images_data, test_number, description, variable_parameter, size_of_filling_patch, size_of_computation_patch, which_clustering_method, number_of_clusters, values_variable_parameter
    
    update_test_number()
    count = 0
    
    for image_data in all_images_data :
        
        count += 1
        update_progress_bar(count)
        
        fixed_parameters_dict = {}
        variable_parameter_dict = {}
        
        image_parent_directory = image_data[0]
        image = image_data[1]
        mask = image_data[2]
        
        # fixed parameters
        
        fixed_parameters_dict["image"] = image
        fixed_parameters_dict["mask"] = mask
        fixed_parameters_dict["is_color_image"] = True
        fixed_parameters_dict["patch_size"] = size_of_filling_patch
        fixed_parameters_dict["computation_patch_size"] = size_of_computation_patch
        fixed_parameters_dict["which_clustering_method"] = which_clustering_method
        fixed_parameters_dict["number_of_clusters"] = number_of_clusters
        fixed_parameters_dict.pop(variable_parameter)
        
        # variable parameter
        
        update_values_variable_parameter()
        
        variable_parameter_dict["name"] = variable_parameter
        variable_parameter_dict["values"] = values_variable_parameter
        
        # update description
        description = description_entry.get()
        print("description :", description)
        
        # start the test
        new_test = Test.Test(test_number, description, image_parent_directory, fixed_parameters_dict, variable_parameter_dict)
        new_test.start_inpainting_test()
        new_test.store_test_in_textfile()
        


def update_test_number():
    global test_number
    
    date_time = datetime.datetime.now()
    month = str(date_time.month)
    day = str(date_time.day)
    hour = str(date_time.hour)
    minute = str(date_time.minute)
    second = str(date_time.second)
    
    test_number = month + "_" + day + "_" + hour + "_" + minute + "_" + second


def update_values_variable_parameter():
    global values_variable_parameter, starting_value, ending_value, step_value, combobox_starting_value, combobox_ending_value, combobox_ending_value
    
    values_variable_parameter = []
    
    starting_value = int(combobox_starting_value.get())
    ending_value = int(combobox_ending_value.get())
    step_value = int(combobox_step_value.get())
    
    for i in range (starting_value, ending_value + step_value, step_value):
        values_variable_parameter.append(i)



def update_progress_bar(count):
    global all_images_data, progress_bar, already_displayed3
    
    if already_displayed3 :
        progress_bar.destroy()
    else :
        already_displayed3 = True
    
    advancement = "Analyzing image " + str(count) + " over " + str(len(all_images_data))
    progress_bar = Label(master, text = advancement)
    progress_bar.grid(row=11, column=2, sticky="nw")



def browse_dataset_directory():
    """ all the use to look for a new directory for the dataset """
    global dataset_directory, dataset_directory_stringvar
    
    dataset_directory = filedialog.askdirectory(initialdir = parent_folder_path)
    dataset_directory_stringvar.set(dataset_directory)
    
    load_data()


def select_variable_parameter(event):
    global variable_parameter_stringvar, variable_parameter
    
    variable_parameter = convert_syntax(variable_parameter_stringvar.get())
    
    if variable_parameter == "number_of_clusters":
        display_clusters_choices()
    elif variable_parameter == "patch_size":
        display_filling_patch_choices()
    elif variable_parameter == "computation_patch_size":
        display_computation_patch_choices()


def display_clusters_choices():
    global starting_values_tuple, ending_values_tuple, step_values_tuple
    
    starting_values_tuple = ('1','2','3','4','5','10','20','30','40','50','100')
    ending_values_tuple = ('1','2','3','4','5','10','20','30','40','50','100')
    step_values_tuple = ('1','2','3','4','5','10')
    
    display_variable_parameter_commands()
    display_fixed_parameters_commands()


def display_filling_patch_choices():
    global starting_values_tuple, ending_values_tuple, step_values_tuple
    
    starting_values_tuple = ('3','5','7', '9', '11','13','15','17','19','21','23','25')
    ending_values_tuple   = ('3','5','7', '9', '11','13','15','17','19','21','23','25')
    step_values_tuple= ('2','4','6', '8', '10')
    
    display_variable_parameter_commands()
    display_fixed_parameters_commands()


def display_computation_patch_choices():
    global starting_values_tuple, ending_values_tuple, step_values_tuple
    
    starting_values_tuple = ('3','5','7', '9', '11','13','15','17','19','21','23','25')
    ending_values_tuple   = ('3','5','7', '9', '11','13','15','17','19','21','23','25')
    step_values_tuple= ('2','4','6', '8', '10')
    
    display_variable_parameter_commands()
    display_fixed_parameters_commands()



def display_variable_parameter_commands():
    global variable_parameter_frame, starting_values_tuple, ending_values_tuple, step_values_tuple, starting_value_stringvar, ending_value_stringvar, step_value_stringvar, text1, text2, text3, combobox_starting_value, combobox_ending_value, combobox_step_value,already_displayed
    
    if already_displayed :
        # destroy previous widgets
        text1.destroy()
        text2.destroy()
        text3.destroy()
        combobox_starting_value.destroy()
        combobox_ending_value.destroy()
        combobox_step_value.destroy()
    
    else :
        already_displayed = True
    
    # set new widgets
    text1 = Label(variable_parameter_frame, text = "From :")
    text2 = Label(variable_parameter_frame, text = " to :")
    text3 = Label(variable_parameter_frame, text = " with step :")
    
    combobox_starting_value = ttk.Combobox(variable_parameter_frame, values = starting_values_tuple, width=5, textvariable = starting_value_stringvar)
    combobox_ending_value   = ttk.Combobox(variable_parameter_frame, values = ending_values_tuple  , width=5, textvariable = ending_value_stringvar  )
    combobox_step_value     = ttk.Combobox(variable_parameter_frame, values = step_values_tuple  , width=5, textvariable = step_value_stringvar  )
    
    combobox_starting_value.bind("<<ComboboxSelected>>", select_starting_value)
    combobox_ending_value.bind("<<ComboboxSelected>>", select_ending_value)
    combobox_step_value.bind("<<ComboboxSelected>>", select_step_value)
    
    
    text1.pack(side="left")
    combobox_starting_value.pack(side="left")
    text2.pack(side="left")
    combobox_ending_value.pack(side="left")
    text3.pack(side="left")
    combobox_step_value.pack(side="left")



def display_fixed_parameters_commands():
    global already_displayed2, variable_parameter, number_of_clusters_instructions, size_of_filling_patch_instructions, size_of_computation_patch_instructions, combobox_number_of_clusters, combobox_size_of_filling_patch, combobox_size_of_computation_patch, number_of_clusters_stringvar, size_of_filling_patch_stringvar, size_of_computation_patch_stringvar, which_clustering_method_stringvar, clustering_method_instructions, combobox_clustering_method
    
    if already_displayed2 :
        number_of_clusters_instructions.destroy()
        combobox_number_of_clusters.destroy()
        size_of_filling_patch_instructions.destroy()
        combobox_size_of_filling_patch.destroy()
        size_of_computation_patch_instructions.destroy()
        combobox_size_of_computation_patch.destroy()
    
    else :
        already_displayed2 = True
    
    ### number of clusters ###
    number_of_clusters_instructions = Label(master, text = "Please select the number of clusters :")
    text_font_number_of_clusters = ('1','2','3','4','5','10','20','30','40','50','100')
    combobox_number_of_clusters = ttk.Combobox(values = text_font_number_of_clusters, width=5, textvariable = number_of_clusters_stringvar)
    combobox_number_of_clusters.bind("<<ComboboxSelected>>", select_number_of_clusters)
    combobox_number_of_clusters.bind("<<KeyPress>>", select_number_of_clusters)
    #grid
    number_of_clusters_instructions.grid(row=2, column=2, sticky="nw")
    combobox_number_of_clusters.grid(row=3, column=2, sticky="nw")
    combobox_number_of_clusters.grid(row=3, column=2, sticky="nw")
    
    ### size of filling patch ###
    size_of_filling_patch_instructions = Label(master, text = "Please select the size of the filling patch :")
    text_font_size_of_filling_patch = ('3','5','7', '9', '11','13','15','17','19','21','23','25')
    combobox_size_of_filling_patch = ttk.Combobox(values = text_font_size_of_filling_patch, width=5, textvariable = size_of_filling_patch_stringvar)
    combobox_size_of_filling_patch.bind("<<ComboboxSelected>>", select_size_of_filling_patch)
    combobox_size_of_filling_patch.bind("<<KeyPress>>", select_size_of_filling_patch)
    #grid
    size_of_filling_patch_instructions.grid(row=4, column=2, sticky="nw")
    combobox_size_of_filling_patch.grid(row=5, column=2, sticky="nw")
    
    ### size of computation patch ###
    size_of_computation_patch_instructions = Label(master, text = "Please select the size of the computation patch :")
    text_font_size_of_computation_patch = ('3','5','7', '9', '11','13','15','17','19','21','23','25')
    combobox_size_of_computation_patch = ttk.Combobox(values = text_font_size_of_computation_patch, width=5, textvariable = size_of_computation_patch_stringvar)
    combobox_size_of_computation_patch.bind("<<ComboboxSelected>>", select_size_of_computation_patch)
    combobox_size_of_computation_patch.bind("<<KeyPress>>", select_size_of_computation_patch)
    #grid
    size_of_computation_patch_instructions.grid(row=6, column=2, sticky="nw")
    combobox_size_of_computation_patch.grid(row=7, column=2, sticky="nw")
    
    ### which_clustering_method ###
    clustering_method_instructions = Label(master, text = "Please select the clustering method :")
    text_font_size_clustering_method = ('method 1 : on pixels','method 2 : on patches')
    combobox_clustering_method = ttk.Combobox(values = text_font_size_clustering_method, width=20, textvariable = which_clustering_method_stringvar)
    combobox_clustering_method.bind("<<ComboboxSelected>>", select_clustering_method)
    #grid
    clustering_method_instructions.grid(row=8, column=2, sticky="nw")
    combobox_clustering_method.grid(row=9, column=2, sticky="nw")
    
    
    if variable_parameter == "number_of_clusters":
        number_of_clusters_instructions.destroy()
        combobox_number_of_clusters.destroy()
    elif variable_parameter == "patch_size":
        size_of_filling_patch_instructions.destroy()
        combobox_size_of_filling_patch.destroy()
    elif variable_parameter == "computation_patch_size":
        size_of_computation_patch_instructions.destroy()
        combobox_size_of_computation_patch.destroy()


### SELECT FUNCTIONS ###

def select_starting_value(event):
    global starting_value_stringvar, starting_value
    starting_value = int(starting_value_stringvar.get())

def select_ending_value(event):
    global ending_value_stringvar, ending_value
    ending_value = int(ending_value_stringvar.get())

def select_step_value(event):
    global step_value_stringvar, step_value
    step_value = int(step_value_stringvar.get())

def select_number_of_clusters(event):
    global number_of_clusters_stringvar, number_of_clusters
    number_of_clusters = int(number_of_clusters_stringvar.get())

def select_size_of_filling_patch(event):
    global size_of_filling_patch_stringvar, size_of_filling_patch
    size_of_filling_patch = int(size_of_filling_patch_stringvar.get())

def select_size_of_computation_patch(event):
    global size_of_computation_patch_stringvar, size_of_computation_patch
    size_of_computation_patch = int(size_of_computation_patch_stringvar.get())

def select_clustering_method(event):
    global which_clustering_method_stringvar, which_clustering_method
    
    if which_clustering_method_stringvar.get() == "method 1 : on pixels":
        which_clustering_method = 1
    elif which_clustering_method_stringvar.get() == "method 2 : on patches":
        which_clustering_method = 2
    else :
        raise Exception("ERROR : the clustering method '", which_clustering_method_stringvar.get(), "' is unknown.")


def update_description(event):
    global description_stringvar, description
    description = description_stringvar.get()
    
    print("a")

### TOOLS FUNCTIONS ###

def convert_syntax(parameter_name):
    
    if parameter_name == "number of clusters":
        return "number_of_clusters"
    elif parameter_name == "size of filling patch":
        return "patch_size"
    elif parameter_name == "size of computational patch":
        return "computation_patch_size"
    else :
        raise Exception("Unknown parameter name")




#####################
### BUILD THE GUI ###
#####################


### INIT COMMANDS ###
load_data()

### GENERAL COMMANDS ###
master.title( "Multiple tests GUI" )
master.columnconfigure(0, weight=1, minsize=200)#, pad=10)
master.columnconfigure(1, weight=1, minsize=200)#, pad=10)
master.columnconfigure(2, weight=1, minsize=200)#, pad=10)
master.rowconfigure(0, weight=1, minsize=30)#, pad=10)
master.rowconfigure(1, weight=2, minsize=30)#, pad=10)
master.rowconfigure(2, weight=2, minsize=30)#, pad=10)
master.rowconfigure(3, weight=2, minsize=30)#, pad=10)
master.rowconfigure(4, weight=2, minsize=30)#, pad=10)
master.rowconfigure(5, weight=2, minsize=30)#, pad=10)
master.rowconfigure(6, weight=2, minsize=30)#, pad=10)
master.rowconfigure(7, weight=2, minsize=30)#, pad=10)
master.rowconfigure(8, weight=2, minsize=30)#, pad=10)
master.rowconfigure(9, weight=2, minsize=30)#, pad=10)
master.rowconfigure(10, weight=2, minsize=30)#, pad=10)
master.rowconfigure(11, weight=2, minsize=30)#, pad=10)
master.rowconfigure(12, weight=2, minsize=30)#, pad=10)
master.rowconfigure(13, weight=2, minsize=30)#, pad=10)


### LEFT BUTTONS ###

# instructions
instructions_message = Label(master, text = "Please indicate the parameters for your test", wraplength = 300)
instructions_message.grid(row=0, column=0, sticky='n', columnspan=3)

# select a directory for the dataset
dataset_directory_instructions = Label(master, text = "Please choose the directory of the dataset :", wraplength = 300)
dataset_directory_browsing_button = Button(text="Browse", command=browse_dataset_directory, width = 10, height = 2)
dataset_directory_information = Label(master, textvariable = dataset_directory_stringvar)
#grid
dataset_directory_instructions.grid(row=1, column=0, sticky="nw")
dataset_directory_browsing_button.grid(row=2, column=0, sticky="nw")
dataset_directory_information.grid(row=3, column=0, sticky="nw")

# all dataset checkbox
all_dataset_checkbox = Checkbutton(text="Select all images from the dataset", variable=all_dataset_intvar, onvalue=1, offvalue=0, command=enable_all_dataset)
#grid
all_dataset_checkbox.grid(row=4, column=0, sticky="nw")

# select the parameter to be varied
variable_parameter_instructions = Label(master, text = "Please select the parameter that will vary during the test :")
text_font_variable_parameter = ('number of clusters', 'size of filling patch', 'size of computational patch')
combobox_variable_parameter = ttk.Combobox(values = text_font_variable_parameter, width=20, textvariable = variable_parameter_stringvar)
combobox_variable_parameter.bind("<<ComboboxSelected>>", select_variable_parameter)
#grid
variable_parameter_instructions.grid(row=5, column=0, sticky="nw")
combobox_variable_parameter.grid(row=6, column=0, sticky="nw")


### select the values for the variable parameter ###

values_variable_parameter_instructions = Label(master, text = "Please indicate the values of the variable parameter :")
variable_parameter_frame = Frame(master)
#grid
variable_parameter_frame.grid(row=8, column=0, sticky="nw")
values_variable_parameter_instructions.grid(row=7, column=0, sticky="nw")

display_clusters_choices()




### RIGHT BUTTONS ###

# instructions
fixed_parameters_instructions = Label(master, text = "Please select the values of the fixed parameters")
#grid
fixed_parameters_instructions.grid(row=1, column=2, sticky="nw")

# fixed parameters commands
display_fixed_parameters_commands()


# description of the test
description_instructions = Label(master, text = "Please enter a short description for this test :")
description_entry = Entry(master, textvariable=description_stringvar, width=50)
#grid
description_instructions.grid(row=10, column=2, sticky="nw")
description_entry.grid(row=11, column=2, sticky="nw")
description_entry.bind("<<Key>>", update_description)


### BOTTOM BUTTONS ###

start_button = Button(text="START THE TEST", command=start_test, width = 15, height = 2)
start_button.grid(row=13, column=1, sticky="nw")




mainloop()