#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import tempfile
import numpy as np
from skimage import io as skio


def viewimage(im, normalise=True, min_value=0.0, max_value=255.0):
    imt = np.float32(im.copy())
    if platform.system() == 'Darwin':  # Mac
        prephrase = 'open -a Gimp-2.10.app '
        endphrase = ' '
    elif platform.system() == 'Windows':  # Windows
        prephrase = 'start gimp-2.10 '
        endphrase = ' '
    else:  # Linux
        prephrase = 'gimp '
        endphrase = ' &'

    if normalise:
        m = im.min()
        imt = imt - m
        M = im.max()
        if M > 0:
            imt = imt / M

    else:
        imt = (imt - min_value) / (max_value - min_value)
        imt[imt < 0] = 0
        imt[imt > 1] = 1

    nomfichier = tempfile.mktemp('tmp_image.jpg')
    commande = prephrase + nomfichier + endphrase
    skio.imsave(nomfichier, imt)
    os.system(commande)


def viewimage_color(im, normalise=True, min_value=0.0, max_value=255.0):
    imt = np.float32(im.copy())
    if platform.system() == 'Darwin':  # Mac
        prephrase = 'open -a Gimp-2.10.app '
        endphrase = ' '
    else:  # Linux
        prephrase = 'gimp '
        endphrase = ' &'

    if normalise:
        m = im.min()
        imt = imt - m
        M = im.max()
        if M > 0:
            imt = imt / M
    else:
        imt = (imt - min_value) / (max_value - min_value)
        imt[imt < 0] = 0
        imt[imt > 1] = 1

    nomfichier = tempfile.mktemp('tmp_image.pgm')
    commande = prephrase + nomfichier + endphrase
    skio.imsave(nomfichier, imt)
    os.system(commande)
