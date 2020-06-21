#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import inspect
import os
import time
import cv2
import numpy as np
from skimage import io as skio
from PIL import Image, ImageTk

from tkinter import filedialog, Tk, IntVar, StringVar, NW, Canvas, YES, BOTH, Label, Button, Checkbutton, Spinbox, \
    mainloop
from tkinter import ttk

from classes.image_inpainting import ImageInpainting  # to avoid confusion with other python build-in librairies
from useful_functions import viewimage


# for the browsers
parent_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# convention for the mask: 1 if unknown, 0 if known
# convention for the GUI
canvas_width, canvas_height = 1000, 700
buttons_margin_x1, buttons_margin_y1 = .8 * canvas_width, .005 * canvas_height
buttons_margin_x2, buttons_margin_y2 = .01 * canvas_width, .6 * canvas_height

# build the root window
master = Tk()  # must be done at the beginning so as to initialize specific variables such as IntVar() objects

########################
# Global variables
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
optimization_method_stringvar.set('1: Pixel clustering')

# booleans
is_mask_browsed = False
is_color_image = True

# other variables
patch_size = 5
computation_patch_size = 5
frequency = 1
nb_clusters = 1
image = 'Object not yet created'
target_contour_list = []  # pixels of the contour
img = 'No PhotoImage'  # it MUST be a global variable, even if it's not used outside browse_button()
imfloat = 'No image'
maskfloat = 'No mask'
optimization_method = 1


#######################
# Event functions
#######################

def browse_image():
    """ Browse an image in the computer. """
    global window, color_button, optimization_method_message, combobox_optimization_method, nb_clusters_message, \
        combobox_nb_clusters, patch_size_message, patch_size_box, computation_patch_size_message, \
        computation_patch_size_box, inpainting_button, img, loaded_image, imfloat, images_path

    filename = filedialog.askopenfilename(
        initialdir=parent_folder_path,
        title='Select file',
        filetypes=(
            ('all files', '*.*'),
            ('ppm files', '*.ppm'),
            ('jpg files', '*.jpg'),
            ('jpeg files', '*.jpeg')
        )
    )
    print('Path of the image: ', filename)

    if filename is not None:
        print('Valid file')

        # display new buttons
        color_button.place(x=buttons_margin_x2, y=buttons_margin_y2)
        optimization_method_message.place(x=buttons_margin_x2, y=buttons_margin_y2 + 35)
        combobox_optimization_method.place(x=buttons_margin_x2 + 140, y=buttons_margin_y2 + 35)
        nb_clusters_message.place(x=buttons_margin_x2, y=buttons_margin_y2 + 70)
        combobox_nb_clusters.place(x=buttons_margin_x2 + 120, y=buttons_margin_y2 + 70)
        nb_clusters_message2.place(x=buttons_margin_x2 + 180, y=buttons_margin_y2 + 70)
        patch_size_message.place(x=buttons_margin_x2, y=buttons_margin_y2 + 105)
        patch_size_box.place(x=buttons_margin_x2 + 140, y=buttons_margin_y2 + 105)
        computation_patch_size_message.place(x=buttons_margin_x2, y=buttons_margin_y2 + 140)
        computation_patch_size_box.place(x=buttons_margin_x2 + 180, y=buttons_margin_y2 + 140)
        inpainting_button.place(x=buttons_margin_x2, y=buttons_margin_y2 + 175)

        # display image and allow drawing
        wrong_format_image = Image.open(filename)
        photoImage_object2 = ImageTk.PhotoImage(image=wrong_format_image)
        master.photoImage_object2 = photoImage_object2  # important to prevent PhotoImage from being garbage collected
        window.create_image(0, 0, anchor=NW, image=photoImage_object2)

        loaded_image = skio.imread(filename, as_gray=False)
        loaded_image = np.array(loaded_image)
        imfloat = np.float32(loaded_image)

        window.bind("<B1-Motion>", draw_contour)


def browse_mask():
    """ Browse a mask image in the computer. """
    global window, loaded_mask, maskfloat, canvas_width, canvas_height

    filename = filedialog.askopenfilename(
        initialdir=parent_folder_path,
        title='Select file',
        filetypes=(
            ('all files', '*.*'),
            ('ppm files', '*.ppm'),
            ('jpg files', '*.jpg'),
            ('jpeg files', '*.jpeg')
        )
    )
    print('Path of the mask: ', filename)

    if filename is not None:
        print('Valid file')

        wrong_format_mask = Image.open(filename)
        photo_image_object = ImageTk.PhotoImage(wrong_format_mask)

        # display mask
        master.photoImage_object = photo_image_object
        window.create_image(0.6 * canvas_width, 0.4 * canvas_height, anchor=NW, image=photo_image_object)

        loaded_mask = skio.imread(filename, as_gray=False)
        loaded_mask = np.array(loaded_mask)
        maskfloat = np.float32(loaded_mask)


def enable_mask_browsing():
    """ Update the is_mask_browsed boolean and display or hide the mask browsing commands. """
    global mask_intvar, is_mask_browsed, browse_mask_message, browse_mask_button

    # update the boolean
    if mask_intvar.get() == 0:
        is_mask_browsed = False
    elif mask_intvar.get() == 1:
        is_mask_browsed = True
    else:
        print('Error with the checkbox')

    if is_mask_browsed:
        # display add browsing commands
        # browse_mask_message.place(x=buttons_margin_x1, y=buttons_margin_y1 + 280, width=150, height=72)
        browse_mask_button.place(x=buttons_margin_x1, y=buttons_margin_y1 + 200)
    else:
        print('Mask browsing unabled')
        browse_mask_message.place_forget()
        browse_mask_button.place_forget()


def draw_contour(event):
    """ Draw the contour. """
    global target_contour_list, imfloat
    python_green = "#476042"
    line_max, column_max = imfloat.shape[0], imfloat.shape[1]

    x_value = min(max(0, event.x), column_max-1)
    y_value = min(max(0, event.y), line_max-1)
    target_contour_list.append([x_value, y_value])

    x1, y1 = event.x, event.y
    x2, y2 = event.x, event.y
    window.create_rectangle(x1, y1, x2, y2, fill=python_green)


def enable_color():
    global is_color_image, color_intvar

    if color_intvar.get() == 0:
        is_color_image = True
    elif color_intvar.get() == 1:
        is_color_image = False
    else:
        print('Error with the checkbox')


def set_patch_size():
    global patch_size_intvar, patch_size
    patch_size = int(patch_size_intvar.get())


def set_computation_patch_size():
    global computation_patch_size_intvar, computation_patch_size
    computation_patch_size = int(computation_patch_size_intvar.get())


def set_frequency():
    global frequency_stringvar, frequency

    if frequency_stringvar.get() == 'only at the end':
        frequency = 'only at the end'
    elif frequency_stringvar.get() != '':
        frequency = int(frequency_stringvar.get())
        print("selected frequency : ", frequency)


def set_optimization_method():
    global optimization_method_stringvar, optimization_method
    if optimization_method_stringvar.get() == '1: Pixel clustering':
        optimization_method = 1
    elif optimization_method_stringvar.get() == '2: Patch clustering':
        optimization_method = 2
    elif optimization_method_stringvar.get() == '3: Search mask':
        optimization_method = 3
    else:
        raise Exception("ERROR: The clustering method '", optimization_method_stringvar.get(), "' is unknown.")


def set_nb_clusters():
    global nb_clusters_stringvar, nb_clusters
    if nb_clusters_stringvar.get() != '':
        nb_clusters = int(nb_clusters_stringvar.get())
        print("Selected number of clusters: ", nb_clusters)


def start_inpainting():
    """ Start inpainting algorithm """
    global imfloat, master, is_color_image, patch_size, computation_patch_size, image, maskfloat, starting_time,\
        ending_time, optimization_method, nb_clusters

    # build mask
    if is_mask_browsed:
        built_mask = maskfloat

        # process to avoid intermediate values
        for i in range(built_mask.shape[0]):
            for j in range(built_mask.shape[1]):
                if built_mask[i, j] < 127:
                    built_mask[i, j] = 0
                else:
                    built_mask[i, j] = 1

    else:
        built_mask = build_mask(target_contour_list)

    viewimage(built_mask)

    # convert 3D matrix (rgb) to 1D matrix (grey scale)
    test = imfloat.tolist()
    redimensioned_imfloat = np.zeros((imfloat.shape[0], imfloat.shape[1]))
    for i in range(imfloat.shape[0]):
        for j in range(imfloat.shape[1]):
            if type(test[i][j]) == float:  # ie it is a sigle value, and
                redimensioned_imfloat[i, j] = test[i][j]
            else:
                redimensioned_imfloat[i, j] = test[i][j][0]

    print('patch_size = ', patch_size)
    print('computation_patch_size = ', computation_patch_size)

    # set parameters
    set_frequency()
    set_nb_clusters()
    set_optimization_method()

    # notice starting time
    starting_time = time.time()

    # build image object
    if is_color_image:
        image = ImageInpainting(imfloat, built_mask, is_color_image, patch_size, computation_patch_size,
                                optimization_method, nb_clusters, True)
    else:
        image = ImageInpainting(redimensioned_imfloat, built_mask, is_color_image, patch_size,
                                computation_patch_size, optimization_method, nb_clusters, True)

    # destroy window
    master.destroy()

    # start filling process
    image.start_filling_with_data(display_frequency=frequency)

    # notice ending time
    ending_time = time.time()

    # display time data
    print("Number of seconds for this inpainting: ", int(ending_time - starting_time))


def build_mask(target_contour_list):
    """ Build mask from contour. """

    # warning: target_contour_list may not be a global variable anymore
    nb_lines, nb_columns = imfloat.shape[0], imfloat.shape[1]
    mask = np.zeros((nb_lines, nb_columns))

    array_contour = np.array(target_contour_list)
    cv2.fillPoly(mask, pts=[array_contour], color=1.0)

    return mask


#####################
# Build the GUI
#####################


# General commands
master.title('Graphic User Interface for image inpainting')
window = Canvas(master, width=canvas_width, height=canvas_height)
window.pack(expand=YES, fill=BOTH)

# Right side
# browse button
browse_button = Button(text='Browse an image', command=browse_image, width=15, height=2)
browse_button.place(x=buttons_margin_x1, y=buttons_margin_y1 + 10)

# display frequency button
frequency_message = Label(master, text='Frequency of the display (GIMP):', wraplength=175)
frequency_message.place(x=buttons_margin_x1, y=buttons_margin_y1 + 50, width=175, height=72)

text_font = ('1', '10', '20', 'Only at the end', 'None')
combobox_frequency = ttk.Combobox(values=text_font, width=10, textvariable=frequency_stringvar)
combobox_frequency.place(x=buttons_margin_x1, y=buttons_margin_y1 + 100)

# mask options text
mask_message = Label(master, text='Draw the mask on image or:', wraplength=175)
mask_message.place(x=buttons_margin_x1, y=buttons_margin_y1 + 140)

# mask option button
mask_checkbox = Checkbutton(text='Load a pre-existing mask?', variable=mask_intvar, onvalue=1, offvalue=0,
                            command=enable_mask_browsing)
mask_checkbox.place(x=buttons_margin_x1, y=buttons_margin_y1 + 160)

# mask browsing button
browse_mask_button = Button(text='Browse a mask', command=browse_mask, width=15, height=2)

# Left side
# color check box
color_button = Checkbutton(text='B&W image?', variable=color_intvar, onvalue=1, offvalue=0, command=enable_color)

# clustering method
optimization_method_message = Label(master, text='Optimisation method:')
text_font_optimization_methods = (
    '1: Pixel clustering', '2: Patch clustering', '3: Search mask'
)
combobox_optimization_method = ttk.Combobox(values=text_font_optimization_methods, width=18,
                                            textvariable=optimization_method_stringvar)

# number of clusters
nb_clusters_message = Label(master, text='Number of clusters:')
text_font_clusters = ('1', '10', '20', '30', '40', '50')
combobox_nb_clusters = ttk.Combobox(values=text_font_clusters, width=4, textvariable=nb_clusters_stringvar)
nb_clusters_message2 = Label(master, text='(useless for method 3)')

# patch size
patch_size_message = Label(master, text='Size of the filling patch:')
patch_size_box = Spinbox(master, from_=3, to=21, increment=2, textvariable=patch_size_intvar, width=5, command=set_patch_size)

# computation patch size
computation_patch_size_message = Label(master, text='Size of the computation patch:')
computation_patch_size_box = Spinbox(master, from_=3, to=21, increment=2, textvariable=computation_patch_size_intvar,
                                     width=5, command=set_computation_patch_size)

# inpainting button
inpainting_button = Button(text='Start inpainting', command=start_inpainting, width=15, height=2)

mainloop()
