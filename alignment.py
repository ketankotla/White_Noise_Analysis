#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import numpy as np
import skimage.util as sku
import skimage.transform as skt

from pystackreg import StackReg


tReg = dict(
    translation=StackReg(StackReg.TRANSLATION),
    rigid=StackReg(StackReg.RIGID_BODY),
    rotation=StackReg(StackReg.SCALED_ROTATION),
    affine=StackReg(StackReg.AFFINE),
    bilinear=StackReg(StackReg.BILINEAR))


def registerImage(ref, mov, mode="rigid"):
    """
    Register image to a static reference using constrained transformations
    @param ref: 2D static reference image
    @type ref: numpy.ndarray
    @param mov: 2D image to be mapped onto ref
    @type mov: numpy.ndarray
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string
    @return: 2D transformation matrix mapping mov onto ref
    @rtype: numpy.ndarray
    """
    tmat = tReg[mode].register(ref, mov)
    return tmat


def transformImage(mov, tmat):
    """
    Transform image using a known transformation matrix
    @param mov: 2D image to be transformed
    @type mov: numpy.ndarray
    @param tmat: 2D transformation matrix with which to transform mov
    @type tmat:
    @return: transformed 2D image
    @rtype: numpy.ndarray
    """
    # transform image with nearest neighbor interpolation using skimage
    mov = skt.warp(mov, tmat, order=0, mode="constant", cval=0)
    # convert foating-point image to an unsigned 8-bit image
    mov = sku.img_as_ubyte(mov)
    return mov


def alignStackDimension(ref, mov, axis, channel, mode="rigid"):
    """
    Register and transform an entire hyperstack with a specific reference
    hyperstack of identical shape in all axes except for the alignment axis
    @param ref: 5D static reference hyperstack with identical shape to mov,
        except for an axis of length 1 corresponding to the axis along which
        to do the alignment. Singleton axis must correspond to "axis"
        parameter
    @type ref: numpy.ndarray
    @param mov: 5D hyperstack to be transformed
    @type mov: numpy.ndarray
    @param axis: axis along which to do alignment:
        0: align first time dimension
        1: align second depth dimension
    @type axis: int
    @param channel: reference channel from which to derive transformation
        matrices
    @type channel: int
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string
    @return: aligned hyperstack of the same shape as mov
    @rtype: numpy.ndarray
    """
    for t in range(mov.shape[0]):
        for z in range(mov.shape[1]):
            """
            derive transformation matrix for all images [T, Z, :, :, :] in mov
            by reistering mov[T, Z, reference channel, :, :] image to static
            ref[T (0 if axis = 0), Z (0 if axis == 1), reference channel, :, :]
            """
            axes = ((1, 0) if axis == 1 else (0, 1))
            refImage = ref[t * axes[0], z * axes[1], channel]
            tmat = registerImage(refImage, mov[t, z, channel], mode=mode)
            for c in range(mov.shape[2]):
                mov[t, z, c] = transformImage(mov[t, z, c], tmat)

    return mov


def alignStack(mov, channel, mode):
    """
    Register and transform an entire hyperstack
    @param mov: 5D hyperstack to be transformed
    @type mov: numpy.ndarray
    @param channel: reference channel on which to conduct transformation
    @type channel: int
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string
    @return: aligned hyperstack of the same shape as mov
    @rtype: numpy.ndarray
    """
    if mov.shape[1] > 1:
        ref = np.mean(mov, axis=1).astype("uint8")[:, np.newaxis]
        mov = alignStackDimension(ref, mov, axis=1, channel=channel, mode=mode)
    if mov.shape[0] > 1:
        ref = np.mean(mov, axis=0).astype("uint8")[np.newaxis]
        mov = alignStackDimension(ref, mov, axis=0, channel=channel, mode=mode)
    return mov