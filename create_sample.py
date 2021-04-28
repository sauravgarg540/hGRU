import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import scipy
from scipy import ndimage
from skimage import transform
import time
import cv2
import random
import imageio

def generate_dilation_struct(margin):
    kernel = np.zeros((2 * margin + 1, 2 * margin + 1))
    y, x = np.ogrid[-margin:margin + 1, -margin:margin + 1]
    mask = x ** 2 + y ** 2 <= margin ** 2
    kernel[mask] = 1
    return kernel

aa_scale = 4
small_dilation_structs = generate_dilation_struct(4)

def draw_line_n_mask(im_size, start_coord, orientation, length, thickness, margin, large_dilation_struct, aa_scale, contrast_scale=1.0):

    # sanity check
    if np.round(thickness*aa_scale) - thickness*aa_scale != 0.0:
        raise ValueError('thickness does not break even.')

    # draw a line in a finer resolution
    miniline_blown_shape = (length + int(np.ceil(thickness)) + margin) * 2 * aa_scale + 1
    miniline_blown_center = (length + int(np.ceil(thickness)) + margin) * aa_scale
    miniline_blown_thickness = int(np.round(thickness*aa_scale))
    miniline_blown_head = translate_coord([miniline_blown_center, miniline_blown_center], orientation, length*aa_scale)
    miniline_blown_im = Image.new('F', (miniline_blown_shape, miniline_blown_shape), 'black')
    line_draw = ImageDraw.Draw(miniline_blown_im)
    line_draw.line([(miniline_blown_center, miniline_blown_center),
                    (miniline_blown_head[1],miniline_blown_head[0])],
                   fill='white', width=miniline_blown_thickness)

    # resize with interpolation + apply contrast
    miniline_shape = (length + int(np.ceil(thickness)) + margin) *2 + 1
    miniline_im = transform.resize(np.array(miniline_blown_im),
                                      (miniline_shape, miniline_shape)).astype(float)/255
    if contrast_scale != 1.0:
        miniline_im *= contrast_scale

    # draw a mask
    minimask_blown_im = binary_dilate_custom(miniline_blown_im, large_dilation_struct, value_scale=1.).astype(np.uint8)
    minimask_im = transform.resize(np.array(minimask_blown_im),
                        (miniline_shape, miniline_shape)).astype(float) / 255

    #minimask_im = binary_dilate(miniline_im, margin, type='1', scale=1.).astype(np.uint8)

    # place in original shape
    l_im = np.array(Image.new('F', (im_size[1], im_size[0]), 'black'))
    m_im = l_im.copy()
    l_im_vertical_range_raw = [start_coord[0] - (length + int(np.ceil(thickness)) + margin),
                               start_coord[0] + (length + int(np.ceil(thickness)) + margin)]
    l_im_horizontal_range_raw = [start_coord[1] - (length + int(np.ceil(thickness)) + margin),
                                 start_coord[1] + (length + int(np.ceil(thickness)) + margin)]
    l_im_vertical_range_rectified = [np.maximum(l_im_vertical_range_raw[0], 0),
                                     np.minimum(l_im_vertical_range_raw[1], im_size[0]-1)]
    l_im_horizontal_range_rectified = [np.maximum(l_im_horizontal_range_raw[0], 0),
                                       np.minimum(l_im_horizontal_range_raw[1], im_size[1]-1)]
    miniline_im_vertical_range_rectified = [np.maximum(0,-l_im_vertical_range_raw[0]),
                                            miniline_shape - 1 - np.maximum(0,l_im_vertical_range_raw[1]-(im_size[0]-1))]
    miniline_im_horizontal_range_rectified = [np.maximum(0,-l_im_horizontal_range_raw[0]),
                                              miniline_shape - 1 - np.maximum(0,l_im_horizontal_range_raw[1]-(im_size[1]-1))]
    l_im[l_im_vertical_range_rectified[0]:l_im_vertical_range_rectified[1]+1,
         l_im_horizontal_range_rectified[0]:l_im_horizontal_range_rectified[1]+1] = \
        miniline_im[miniline_im_vertical_range_rectified[0]:miniline_im_vertical_range_rectified[1] + 1,
                    miniline_im_horizontal_range_rectified[0]:miniline_im_horizontal_range_rectified[1] + 1].copy()
    m_im[l_im_vertical_range_rectified[0]:l_im_vertical_range_rectified[1]+1,
         l_im_horizontal_range_rectified[0]:l_im_horizontal_range_rectified[1]+1] = \
        minimask_im[miniline_im_vertical_range_rectified[0]:miniline_im_vertical_range_rectified[1] + 1,
                    miniline_im_horizontal_range_rectified[0]:miniline_im_horizontal_range_rectified[1] + 1].copy()

    return l_im, m_im

def binary_dilate_custom(im, struct, value_scale=1.):
    #out = ndimage.morphology.binary_dilation(np.array(im), structure=struct, iterations=iterations)
    out = np.array(cv2.dilate(np.array(im), kernel=struct.astype(np.uint8), iterations = 1)).astype(float)/value_scale
    #out = np.minimum(signal.fftconvolve(np.array(im), struct, mode='same').astype(np.uint8), np.ones_like(im))
    return out

def translate_coord(coord, orientation, dist, allow_float=False):
    y_displacement = float(dist)*np.sin(orientation)
    x_displacement = float(dist)*np.cos(orientation)
    if allow_float is True:
        new_coord = [coord[0]+y_displacement, coord[1]+x_displacement]
    else:
        new_coord = [int(np.ceil(coord[0] + y_displacement)), int(np.ceil(coord[1] + x_displacement))]
    return new_coord

def draw_circle(window_size, coordinate, radius, aa_scale):
    image = np.zeros((window_size[0]*aa_scale, window_size[1]*aa_scale))
    y, x = np.ogrid[-coordinate[0]*aa_scale:(window_size[0]-coordinate[0])*aa_scale,
                    -coordinate[1]*aa_scale:(window_size[1]-coordinate[1])*aa_scale]
    mask = x ** 2 + y ** 2 <= (radius*aa_scale) ** 2
    image[mask] = 1.0
    return transform.resize(image, (window_size[0], window_size[1]))
start = 100
# img2 = draw_circle([300,300],[150,150],3,4)

# Horizontal dashes

# for run in range(2,12,2):
#     run = 3
#     image = np.zeros((300,300),np.uint8)
#     k = 0
#     for i in range(14):
#         l_im, m_im = draw_line_n_mask((300,300), [start,100+k], 0, 5, 1.5, 4, generate_dilation_struct(4*aa_scale), aa_scale, contrast_scale=1.0)
#         image = np.maximum(image, l_im)
#         k+=9
#     k=0
#     for i in range(14):
#         l_im, m_im = draw_line_n_mask((300,300), [start+run,100+k], 0, 5, 1.5, 4, generate_dilation_struct(4*aa_scale), aa_scale, contrast_scale=1.0)
#         image = np.maximum(image, l_im)
#         k+=9


# Vertical dashes

for run in range(2,11,1):

    image = np.zeros((300,300),np.uint8)
    k = 0
    for i in range(14):
        l_im, m_im = draw_line_n_mask((300,300), [start+k,100+k], np.pi/4, 5, 1.5, 4, generate_dilation_struct(4*aa_scale), aa_scale, contrast_scale=1.0)
        image = np.maximum(image, l_im)
        k+=9
    k=0
    for i in range(14):
        # print(start+run+k,100+k)
        l_im, m_im = draw_line_n_mask((300,300), [start+run+k,100+k], np.pi/4, 5, 1.5, 4, generate_dilation_struct(4*aa_scale), aa_scale, contrast_scale=1.0)
        image = np.maximum(image, l_im)
        k+=9

    sample = draw_circle([300,300],[start,100],3,4)
    image = np.maximum(image, sample)
    # print(start+run+k,100+k)
    sample = draw_circle([300,300],[start+run+k-5,100+k-5],3,4)
    im = np.maximum(image, sample)
    # print(np.max(im))
    imageio.imwrite(f'experiment/{run}_pixels_upper.png', im)
