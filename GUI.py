#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### SYSTEM IMPORTS ###
import sys
import os
import inspect


### ADDING FOLDERS TO THE PATH ###
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
classes_path = parent_folder_path + "/classes"
images_path = parent_folder_path + "/images"
if not parent_folder_path in sys.path :
    sys.path.append(parent_folder_path)
if not classes_path in sys.path :
    sys.path.append(classes_path)

### CLASS IMPORTS ###
import Image as ImageInpainting # to avoid confusion with other python build-in librairies
from Pixel import Pixel
from useful_functions import viewimage
from useful_functions import viewimage_color

### TOOLS IMPORTS ###
import numpy as np
import matplotlib.pyplot as plt
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io as skio
import time

import cv2


""" DEBUGGING """
import pdb
import cProfile
import re


### GUI IMPORTS ###
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

from PIL import Image, ImageTk

"""
conventions pour le masque :
    - blanc = 1 = valeur inconnue
    - noir = 0 = valeur connue
"""


### CONVENTIONS FOR THE GUI ###

canvas_width = 1000
canvas_height = 800
buttons_margin_x1 = 0.80*canvas_width
buttons_margin_y1 = 0.05*canvas_height

buttons_margin_x2 = 0.01*canvas_width
buttons_margin_y2 = 0.65*canvas_height



### BUILD THE ROOT WINDOW ###
master = Tk() # must be done at the beginning so as to initialize specific variables such as IntVar() objects



########################
### GLOBAL VARIABLES ###
########################


# IntVar objects
color_intvar = IntVar()
mask_intvar = IntVar()
patch_size_intvar = IntVar()
patch_size_intvar.set(5)
computation_patch_size_intvar = IntVar()
computation_patch_size_intvar.set(5)

# StringVar objects
frequency_stringvar = StringVar()
frequency_stringvar.set('1')
nb_clusters_stringvar = StringVar()
nb_clusters_stringvar.set('1')
optimization_method_stringvar = StringVar()
optimization_method_stringvar.set('method 1 : clustering on pixels')

# booleans
is_mask_browsed = False
is_color_image = True


# other variables
patch_size = 5
computation_patch_size = 5
frequency = 1
nb_clusters = 1
image = "object not yet created"
target_contour_list = [] # pixels of the contour
img = "no PhotoImage" # it MUST be a global variable, even if it's not used outside browse_button()
imfloat = "no image"
maskfloat = "no mask"
optimization_method = 1




#######################
### EVENT FUNCTIONS ###
#######################

def browse_image():
    """ Browse an image in the computer """
    global window, color_button, optimization_method_message, combobox_optimization_method, nb_clusters_message, combobox_nb_clusters, patch_size_message, patch_size_box, computation_patch_size_message, computation_patch_size_box, inpainting_button, img, loaded_image, imfloat, images_path
    
    filename =  filedialog.askopenfilename(initialdir = parent_folder_path, title = "Select file", filetypes = (("ppm files","*.ppm"), ("jpg files","*.jpg"), ("jpeg files","*.jpeg"), ("all files","*.*")))
    print("path of the image : ",filename)
    
    if filename != None :
        print("valid file")
        
        # display new buttons
        color_button.place(x = buttons_margin_x2, y = buttons_margin_y2)
        optimization_method_message.place(x = buttons_margin_x2, y = buttons_margin_y2 + 25)
        combobox_optimization_method.place(x = buttons_margin_x2, y = buttons_margin_y2 + 50)
        nb_clusters_message.place(x = buttons_margin_x2, y = buttons_margin_y2 + 85)
        combobox_nb_clusters.place(x = buttons_margin_x2, y = buttons_margin_y2 + 110)
        nb_clusters_message2.place(x = buttons_margin_x2 + 150, y = buttons_margin_y2 + 115)
        patch_size_message.place(x = buttons_margin_x2, y = buttons_margin_y2 + 145)
        patch_size_box.place(x = buttons_margin_x2, y = buttons_margin_y2 + 165)
        computation_patch_size_message.place(x = buttons_margin_x2, y = buttons_margin_y2 + 195)
        computation_patch_size_box.place(x = buttons_margin_x2, y = buttons_margin_y2 + 215)
        inpainting_button.place(x = buttons_margin_x2, y = buttons_margin_y2 + 245)
        
        
        # display image and allow drawing
        wrong_format_image = Image.open(filename)
        photoImage_object2 = ImageTk.PhotoImage(image=wrong_format_image)
        master.photoImage_object2 = photoImage_object2 # very important to prevent PhotoImage object being garbage collected
        window.create_image(0,0, anchor=NW, image=photoImage_object2)
        
        loaded_image =skio.imread(filename, as_gray=False)
        loaded_image = np.array(loaded_image)
        imfloat= np.float32(loaded_image)
        
        window.bind( "<B1-Motion>", draw_contour)


def browse_mask():
    """ Browse a mask image in the computer """
    global window, loaded_mask, maskfloat, canvas_width, canvas_height
    
    filename =  filedialog.askopenfilename(initialdir = parent_folder_path, title = "Select file", filetypes = (("ppm files","*.ppm"), ("jpg files","*.jpg"), ("jpeg files","*.jpeg"), ("all files","*.*")))
    print("path of the mask : ",filename)
    
    if filename != None :
        print("valid file")
        
        wrong_format_mask = Image.open(filename)
        photoImage_object = ImageTk.PhotoImage(wrong_format_mask)
        
        # display mask
        master.photoImage_object = photoImage_object
        window.create_image(0.6*canvas_width,0.67*canvas_height, anchor=NW, image=photoImage_object)
        
        loaded_mask =skio.imread(filename, as_gray=False)
        loaded_mask = np.array(loaded_mask)
        maskfloat= np.float32(loaded_mask)
        
        """
        # display it in a new window
        fenetre = Tk()
        canvas = Canvas(fenetre,width=500, height=500)
        fenetre.photoImage_object = photoImage_object
        canvas.create_image(0, 0, anchor=NW, image=photoImage_object)
        canvas.pack()
        """


def enable_mask_browsing():
    """ update the is_mask_browsed boolean and display or hide the mask browsing commands """
    global mask_intvar, is_mask_browsed, browse_mask_message, browse_mask_button
    
    # update the boolean
    if mask_intvar.get() == 0:
        is_mask_browsed = False
    elif mask_intvar.get() == 1:
        is_mask_browsed = True
    else :
        print("Error with the checkbox")
    
    if is_mask_browsed :
        # display add browsing commands
        browse_mask_message.place(x = buttons_margin_x1, y = buttons_margin_y1 + 280, width = 150, height = 72)
        browse_mask_button.place(x = buttons_margin_x1, y = buttons_margin_y1 + 350)
    else :
        print("mask browsing unabled")
        browse_mask_message.place_forget()
        browse_mask_button.place_forget()


def draw_contour( event ):
    """ draw the contour """
    global target_contour_list, imfloat
    python_green = "#476042"
    
    line_max = imfloat.shape[0]
    column_max = imfloat.shape[1]
    
    if (0 < event.y < line_max-1 and 0 < event.x < column_max-1):
        target_contour_list.append([event.x,event.y])
        print("new pixel : ",[event.y,event.x])
    else :
        # x issue
        if (event.x <= 0):
            x_value = 0
        elif (event.x >= column_max-1) :
            x_value = column_max-1
        else :
            x_value = event.x
        
        # y issue
        if (event.y <= 0):
            y_value = 0
        elif (event.y >= line_max-1) :
            y_value = line_max-1
        else :
            y_value = event.y
        
        # add them to the list
        target_contour_list.append([x_value,y_value])
        print("out of the image : (", event.y, ",", event.x, ") becomes (", y_value, ",", x_value, ")")
    
    x1, y1 = ( event.x ), ( event.y )
    x2, y2 = ( event.x ), ( event.y )
    window.create_rectangle( x1, y1, x2, y2, fill = python_green )


def enable_color():
    global is_color_image, color_intvar
    if color_intvar.get() == 0:
        is_color_image = True
    elif color_intvar.get() == 1:
        is_color_image = False
    else :
        print("Error with the checkbox")

def set_patch_size():
    global patch_size_intvar, patch_size
    patch_size = int(patch_size_intvar.get())

def set_computation_patch_size():
    global computation_patch_size_intvar, computation_patch_size
    computation_patch_size = int(computation_patch_size_intvar.get())

def set_frequency():
    global frequency_stringvar, frequency
    if frequency_stringvar.get() == "only at the end" :
        frequency = "only at the end"
    elif  frequency_stringvar.get() != '' :
        frequency = int(frequency_stringvar.get())
        print("selected frequency : ", frequency)

def set_optimization_method():
    global optimization_method_stringvar, optimization_method
    if (optimization_method_stringvar.get() == 'method 1 : clustering on pixels') :
        optimization_method = 1
    elif (optimization_method_stringvar.get() == 'method 2 : clustering on patches') :
        optimization_method = 2
    elif (optimization_method_stringvar.get() == 'method 3 : search mask') :
        optimization_method = 3
    else :
        raise Exception("ERROR : the clustering method '", optimization_method_stringvar.get(), "' is unknown.")


def set_nb_clusters():
    global nb_clusters_stringvar, nb_clusters
    if  nb_clusters_stringvar.get() != '' :
        nb_clusters = int(nb_clusters_stringvar.get())
        print("selected number of clusters : ", nb_clusters)


def start_inpainting():
    """ start inpainting algorithm """
    global imfloat, master, is_color_image, patch_size, computation_patch_size, image, maskfloat, starting_time, ending_time, optimization_method, nb_clusters
    
    # build mask
    if is_mask_browsed :
        built_mask = maskfloat
        
        # process to avoid intermediate values
        for i in range (built_mask.shape[0]):
            for j in range (built_mask.shape[1]):
                if built_mask[i,j]<127:
                    built_mask[i,j] = 0
                else :
                    built_mask[i,j] = 1
    
    else :
        built_mask = build_mask(target_contour_list)
    
    viewimage(built_mask)
    
    # convert 3D matrix (rgb) to 1D matrix (grey scale)
    test = imfloat.tolist()
    redimensioned_imfloat = np.zeros((imfloat.shape[0],imfloat.shape[1]))
    for i in range (imfloat.shape[0]):
        for j in range (imfloat.shape[1]):
            if type(test[i][j]) == float: # ie it is a sigle value, and
                redimensioned_imfloat[i,j] = test[i][j]
            else :
                redimensioned_imfloat[i,j] = test[i][j][0]
    
    print("patch_size = ", patch_size)
    print("computation_patch_size = ", computation_patch_size)
    
    # set parameters
    set_frequency()
    set_nb_clusters()
    set_optimization_method()
    
    
    # notice starting time
    starting_time = time.time()
    
    # build image object
    if (is_color_image):
        image = ImageInpainting.Image(imfloat, built_mask, is_color_image, patch_size, computation_patch_size, optimization_method, nb_clusters, True)
    else :
        image = ImageInpainting.Image(redimensioned_imfloat, built_mask, is_color_image, patch_size, computation_patch_size, optimization_method, nb_clusters, True)
    
    # destroy window
    master.destroy()
    
    # start filling process
    image.start_filling_with_data(display_frequency=frequency)
        
    # notice ending time
    ending_time = time.time()
    
    # display time data
    print("Number of seconds for this inpainting : ", int(ending_time - starting_time))
    

def build_mask(target_contour_list):
    """ build mask from contour """
    
    # warning : target_contour_list may not be a global variable anymore !
    nb_lines = imfloat.shape[0]
    nb_columns = imfloat.shape[1]    
    mask = np.zeros((nb_lines,nb_columns))
    
    array_contour = np.array(target_contour_list)
    cv2.fillPoly(mask, pts =[array_contour], color=1.0)
    
    return mask



#####################
### BUILD THE GUI ###
#####################


### GENERAL COMMANDS ###

master.title( "Graphic User Interace for image inpainting" )
window = Canvas(master, width=canvas_width, height=canvas_height)
window.pack(expand = YES, fill = BOTH)


### RIGHT SIDE ###

# browse button
browse_message = Label(master, text = "Please choose a file with '.jpg', '.jpeg' or '.ppm' extension :", wraplength = 150)
browse_message.place(x = buttons_margin_x1, y = buttons_margin_y1, width = 150, height = 72)

browse_button = Button(text="Browse", command=browse_image, width = 10, height = 2)
browse_button.place(x = buttons_margin_x1 + 30, y = buttons_margin_y1 + 70)

# display frequency button
frequency_message = Label(master, text = "Please choose the frequency of the display (using GIMP):", wraplength = 150)
frequency_message.place(x = buttons_margin_x1, y = buttons_margin_y1 + 110, width = 150, height = 72)

text_font = ('1', '10', '20', "only at the end", "none")
combobox_frequency = ttk.Combobox(values = text_font, width=14, textvariable = frequency_stringvar)
combobox_frequency.place(x = buttons_margin_x1, y = buttons_margin_y1 + 180)

# mask options text
mask_message = Label(master, text = "Please draw the mask on image or :", wraplength = 250)
mask_message.place(x = buttons_margin_x1 - 40, y = buttons_margin_y1 + 220)

# mask option button
mask_checkbox = Checkbutton(text="Load a pre-existing mask ?", variable=mask_intvar , onvalue=1, offvalue=0, command=enable_mask_browsing)
mask_checkbox.place(x = buttons_margin_x1 - 20, y = buttons_margin_y1 + 250)

# mask browsing button
browse_mask_message = Label(master, text = "Please choose a mask file with '.jpg', '.jpeg' or '.ppm' extension :", wraplength = 150)

browse_mask_button = Button(text="Browse for the Mask", command=browse_mask, width = 16, height = 2)


### LEFT SIDE ###

# color check box
color_button = Checkbutton(text="Is this a black and white image?", variable=color_intvar, onvalue=1, offvalue=0, command=enable_color)


# clustering method
optimization_method_message = Label(master, text = "Please choose an optimisation method for this test :")
text_font_optimization_methods = ('method 1 : clustering on pixels', 'method 2 : clustering on patches', 'method 3 : search mask')
combobox_optimization_method = ttk.Combobox(values = text_font_optimization_methods, width=25, textvariable = optimization_method_stringvar)


# number of clusters
nb_clusters_message = Label(master, text = "Please enter the number of clusters for this image :")
text_font_clusters = ('1', '10', '20', '30', '40', '50')
combobox_nb_clusters = ttk.Combobox(values = text_font_clusters, width=14, textvariable = nb_clusters_stringvar)
nb_clusters_message2 = Label(master, text = "(useless for method 3)")

# patch size
patch_size_message = Label(master, text = "Please select the size of the filling patch :")
patch_size_box = Spinbox(master, from_=3, to=21, increment=2, textvariable=patch_size_intvar, width=5, command=set_patch_size)

# computation patch size
computation_patch_size_message = Label(master, text = "Please select the size of the computation patch :")
computation_patch_size_box = Spinbox(master, from_=3, to=21, increment=2, textvariable=computation_patch_size_intvar, width=5, command=set_computation_patch_size)

# inpainting button
inpainting_button = Button(text="Start inpainting", command=start_inpainting, width = 15, height = 2)


mainloop()