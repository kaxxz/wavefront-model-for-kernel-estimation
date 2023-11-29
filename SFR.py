# -*- coding: utf-8 -*-

import scipy.signal
import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import csv



def angle_from_slope(slope):
    return torch.rad2deg(torch.arctan(slope))


def slope_from_angle(angle):
    return torch.tan(torch.deg2rad(angle))


def centroid(arr, conv_kernel=3, win_width=5):
    height, width = arr.shape
    win = torch.zeros(arr.shape)
    
    for i in range(height):
        win_c = torch.argmax(torch.abs(convolve(arr[i,:], torch.ones(conv_kernel)/3)))
        win[i, win_c - win_width:win_c + win_width] = 1.0

    x, _ = torch.meshgrid(torch.arange(0,width), torch.arange(0,height))
    x=x.T #和np互为转置
    sum_arr = torch.sum(arr * win, dim=1)
    sum_arr_x = torch.sum(arr * win * x, dim=1)
    
    # sample_diff=win_c
    # with open('centre.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(sample_diff.tolist())

    return sum_arr_x / sum_arr  # divide-by-zero warnings are suppressed


def differentiate(arr0, kernel):
    if len(arr0.shape) == 1 :
        out0 = convolve(arr0, kernel)
        out=out0.squeeze(0).squeeze(0)
        out[0] = 0.0

    else :
        kernel0=torch.flip(kernel, dims=[0])
        kernel =kernel0.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        arr = arr0.unsqueeze(0).unsqueeze(0)

        sample = F.conv2d(arr,kernel,stride=1,padding='same')
        out=torch.squeeze(sample)
        # out[:,0:5] = 0.0      # The first element is not valid since there is no 'symm' option,  
        # out[:,-5:] = 0.0     # replace it with 0.0 (thereby maintaining the itorchut array size)    

    return out

def convolve(arr0, kernel):

    x=arr0.unsqueeze(0).unsqueeze(0).double()
    k0=torch.flip(kernel, dims=[0])
    k=k0.unsqueeze(0).unsqueeze(0).double()
    out=F.conv1d(x,k,padding='same')

    return out

def polyfit(X0, Y0, degree):

    X=X0.unsqueeze(dim=1)
    Y=Y0.unsqueeze(dim=1)
    one=torch.ones(X.size())
    if degree==0:
        return Y-X

    elif degree==1:
        X = torch.cat((X,one), 1)
        pcoefs = torch.linalg.inv(X.T @ X) @ X.T @ Y
        return pcoefs
    
    elif degree==2:
        X = torch.cat((torch.pow(X, 2), X,one), 1)
        pcoefs = torch.linalg.inv(X.T @ X) @ X.T @ Y
        return pcoefs
    elif degree==3:
        X = torch.cat((torch.pow(X, 3),torch.pow(X, 2), X,one), 1)
        pcoefs = torch.linalg.inv(X.T @ X) @ X.T @ Y
    return pcoefs


    

def find_edge(centr, angle=None):

    idx = torch.where(torch.isfinite(centr))[0][1:-1]

    # Find the location and direction of the edge by fitting a line to the 
    # centroids on the form x = y*slope + offset
    if angle is None:
        slope, offset = polyfit(idx, centr[idx], 1)
        # print(centr[idx])
    else:
        slope = slope_from_angle(angle)
        offset = polyfit(idx, centr[idx] - slope * idx, 0)

    # pcoefs contains quadratic polynomial coefficients for the x-coordinate
    # of the curved edge as a function of the y-coordinate: 
    # x = pcoefs[0] * y**2 + pcoefs[1] * y + pcoefs[2]
    pcoefs = polyfit(idx, centr[idx], 2)

    return pcoefs, slope, offset


def midpoint_slope_and_curvature_from_polynomial(a, b, c, y0, y1):
    # Describe itorchut 2nd degree polynomial f(y) = a*y**2 + b*y + c in
    # terms of midpoint, slope (at midpoint), and curvature (at midpoint)
    y_mid = (y1 + y0) / 2
    x_mid = a * y_mid ** 2 + b * y_mid + c
    # Calculated slope as first derivative of x = f(y) at y = y_mid
    slope = 2 * a * y_mid + b
    # Calculate the curvature as k(y) = f''(y) / (1 + f'(y)^2)^(3/2)
    curvature = 2 * a / (1 + slope ** 2) ** (3 / 2)
    return y_mid, x_mid, slope, curvature


def polynomial_from_midpoint_slope_and_curvature(y_mid, x_mid, slope, curvature):
    # Calculate a 2nd degree polynomial x = f(y) = a*y**2 + b*y + c that passes
    # through the midpoint (x_mid, y_mid) with the given slope and curvature 
    a = curvature * (1 + slope ** 2) ** (3 / 2) / 2
    b = slope - 2 * a * y_mid
    c = x_mid - a * y_mid ** 2 - b * y_mid
    return [a, b, c]


def cubic_solver(a, b, c, d):
    # Solve the equation a*x**3 + b*x**2 + c*x + d = 0 for a 
    # real-valued root x by Cardano's method
    # (https://en.wikipedia.org/wiki/Cubic_equation#Cardano's_formula)

    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)

    # A real root exists if 4 * p**3 + 27 * q**2 > 0
    sr = torch.sqrt(q ** 2 / 4 + p ** 3 / 27)
    t = torch.cbrt(-q / 2 + sr) + torch.cbrt(-q / 2 - sr)
    x = t - b / (3 * a)
    return x


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def calc_distance(data_shape, p):
    quadratic_fit=False
    x, y = torch.meshgrid(torch.arange(0,data_shape[1]), torch.arange(data_shape[0]))
    x,y=x.T,y.T
    if not quadratic_fit or p[0] == 0.0:

        slope, offset = p[1], p[2]  # use linear fit to edge
        a,b,c= 1,-slope,-offset
        a_b = torch.sqrt(a ** 2 + b ** 2)

        # |ax+by+c| / |a_b| is the distance from (x,y) to the slanted edge:
        dist = (a * x + b * y + c) / a_b
    else:
        # Define a cubic polynomial equation for the y-coordinate
        # y0 at the point (x0, y0) on the curved edge that is closest to (x, y)
        d = -y + p[1] * p[2] - x * p[1]
        c = 1 + p[1] ** 2 + 2 * p[2] * p[0] - 2 * x * p[0]
        b = 3 * p[1] * p[0]
        a = 2 * p[0] ** 2

        if p[0] == 0.0:
            y0 = -d / c  # solution if edge is straight (quadratic term is zero)
        else:
            y0 = cubic_solver(a, b, c, d)  # edge is curved

        x0 = p[0] * y0 ** 2 + p[1] * y0 + p[2]
        dxx_dyy = torch.array(2 * p[0] * y0 + p[1])  # slope at (x0, y0)
        r2 = dot([1, -dxx_dyy], [1, -dxx_dyy])
        # distance between (x, y) and (x0, y0) along normal to curve at (x0, y0)
        dist = dot([x - x0, y - y0], [1, -dxx_dyy]) / torch.sqrt(r2)
    return dist


def project_and_bin(data, dist, oversampling):
    # print('data',data)
    # print('dist',dist)
    # Create a matrix "bins" where each element represents the bin index of the 
    # corresponding image pixel in "data":
    bins = torch.round(dist * oversampling).type(torch.int) 
    
    bins = bins.flatten()
    
    bins -= torch.min(bins)  # add an offset so that bins start at 0

    esf = torch.zeros(torch.max(bins) + 1)  # Edge spread function

    cnts = torch.zeros(torch.max(bins) + 1).type(torch.int) 
    data_flat = data.flatten()
    
    for b_indx, b_sorted in zip(torch.sort(bins).indices, torch.sort(bins).values):
        esf[b_sorted] += data_flat[b_indx]  # Collect pixel contributions in this bin
        cnts[b_sorted] += 1  # Keep a tab of how many contributions were made to this bin
        # print(b_indx, b_sorted)
    # Calculate mean by dividing by the number of contributing pixels. Avoid
    # division by zero, in case there are bins with no content.
    esf[cnts > 0] /= cnts[cnts > 0]
    # print(bins)
    # print(torch.sort(bins).values)
    # print(esf)
    # print(cnts)
    if torch.any(cnts == 0):
        patch_cntr = 0
        for i in torch.where(cnts == 0)[0]:  # loop through all empty bin locations
            j = [i - 1, i + 1]  # indices of nearest neighbors
            print("我爱你",j)
            print(torch.sort(bins).values)
            if j[0] < 0:  # Is left neighbor index outside esf array?
                j = j[1]
            elif j[1] == len(cnts):  # Is right neighbor index outside esf array?
                j = j[0]
            if torch.all(cnts[j] > 0):  # Now, if neighbor bins are non-empty
                esf[i] = torch.mean(esf[j])  # use the interpolated value
                patch_cntr += 1
    return esf


def peak_width(y, rel_threshold):
    # Find width of peak in y that is above a certain fraction of the maximum value
    val = torch.abs(y)
    val_threshold = rel_threshold * torch.max(val)
    indices = torch.where(val - val_threshold > 0.0)[0]
    return indices[-1] - indices[0]


def filter_window(lsf, oversampling, lsf_centering_kernel_sz=9,
                  win_width_factor=1.5, lsf_threshold=0.10):
    # The window ('hann_win') returned by this function will be used as a filter 
    # on the LSF signal during the MTF calculation to reduce noise

    nn0 = 20 * oversampling  # sample range to be used for the FFT, intial guess
    mid = len(lsf) // 2
    i1 = max(0, mid - nn0)
    i2 = min(2 * mid, mid + nn0)
    nn = (i2 - i1) // 2  # sample range to be used, final 


    lsf_conv = convolve(lsf[i1:i2], torch.ones(lsf_centering_kernel_sz))

    hann_hw = max(torch.round(win_width_factor * peak_width(lsf_conv, lsf_threshold)).type(torch.int), 5 * oversampling)

    bin_c = torch.argmax(torch.abs(lsf_conv))  # center bin, corresponding to LSF max

    crop_l = max(hann_hw - bin_c, 0)
    crop_r = min(2 * nn - (hann_hw + bin_c), 0)
    hann_win = torch.zeros(2 * nn)  # default value outside Hann function
    hann_win[bin_c - hann_hw + crop_l:bin_c + hann_hw + crop_r] = \
        torch.hann_window(2 * hann_hw)[crop_l:2 * hann_hw + crop_r]
    return hann_win, [i1, i2]


def calc_mtf(lsf, hann_win, idx, oversampling, diff_ft):

    i1, i2 = idx
    mtf = torch.abs(torch.fft.fft(lsf[i1:i2] * hann_win))
    nn = (i2 - i1) // 2
    mtf = mtf[:nn]
    mtf_nl =torch.zeros(nn)
    mtf_nl = mtf/mtf[0] 
    f = torch.arange(0, oversampling / 2, oversampling / nn / 2) 
    mtf_nl *= (1 / torch.sinc(4 * f / (diff_ft * oversampling))).clip(0.0, 10.0)
    return torch.column_stack((f, mtf_nl))



def calc_sfr(image, oversampling=4, offset=None, angle=None):

    diff_kernel = torch.tensor([-0.5, 0.0, 0.5])
    diff_offset = torch.tensor(0.0)
    diff_ft = torch.tensor(2.0)
    pad = transforms.Pad([1,1,1,1], fill=0, padding_mode='edge')
    image_d= pad(image)
    sample_diff = differentiate(image_d, diff_kernel)
    #print(sample_diff)##这里好像没问题
    M = image.size(0)
    H = transforms.CenterCrop(M)
    sample_diff= H(sample_diff)
    # with open('sample_diff.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(sample_diff.tolist())

    centr = centroid(sample_diff) + diff_offset
    
    # Calculate centroids also for the 90° right rotated image
    # rotate by transposing and mirroring
    image_rot90=torch.flip(image.T, dims=[1])	
    sample_diff = differentiate(image_rot90, diff_kernel)
    centr_rot = centroid(sample_diff) + diff_offset

    # Use rotated image if it results in fewer rows without edge transitions
    if torch.sum(torch.isnan(centr_rot)) < torch.sum(torch.isnan(centr)):
        print('不会吧！')
        image, centr = image_rot90, centr_rot
        rotated = True
    else:
        rotated = False

    pcoefs, slope, offset = find_edge(centr, angle)
    pcoefs = torch.tensor([0.0, slope, offset])#
    # print(pcoefs)
    dist = calc_distance(torch.tensor(image.shape), pcoefs)

    esf = project_and_bin(image, dist, oversampling)  # edge spread function
  
    lsf = differentiate(esf, diff_kernel)  # line spread function
    
    hann_win, idx = filter_window(lsf, oversampling)  # define window to be applied on LSF

    mtf = calc_mtf(lsf, hann_win, idx, oversampling, diff_ft)

    angle = angle_from_slope(slope)
    # print('mtf:',mtf)
    # plt.figure()
    # plt.plot(mtf[:, 0], mtf[:, 1])
    # plt.show()
    # mtfa=1e4*torch.trapz(mtf[:2, 1], dx = 0.01)-99 #计算sfr下的面积
    mtfa=torch.trapz(mtf[:2, 1], dx = 1) #计算sfr下的面积
    return mtfa

def relative_luminance(rgb_image, rgb_w=(1/3, 1/3, 1/3)):
    # Return relative luminance of image, based on sRGB MxNx3 (or MxNx4) itorchut
    # Default weights rgb_w are the ones for the sRGB colorspace
    if rgb_image.ndim == 2:
        return rgb_image  # do nothing, this is an MxN image without color data
    else:
        return rgb_w[0] * rgb_image[:, :, 0] + rgb_w[1] * rgb_image[:, :, 1] + rgb_w[2] * rgb_image[:, :, 2]


#low_level hi_level0.15，1.2【4:1 对比度在 ISO 12233:2014 标准中指定。】

def make_ideal_slanted_edge(image_shape=(50, 50), angle=5.0, low_level=0.25, hi_level=1,
                            pixel_fill_factor=1.0):

    height, width = image_shape
    xx, yy = torch.meshgrid(torch.tensor(range(0, width)), torch.tensor(range(0, height)))
    x_midpoint = torch.tensor(width / 2.0 - 0.5)
    y_midpoint = torch.tensor(height / 2.0 - 0.5)
    angle = torch.tensor(angle)
    pixel_fill_factor=torch.tensor(pixel_fill_factor)
    dist_edge = torch.cos(-angle * torch.pi / 180) * (xx - x_midpoint) + -torch.sin(-angle * torch.pi / 180) * (yy - y_midpoint)

    dist_edge /= torch.sqrt(pixel_fill_factor)

    return low_level + (hi_level - low_level) * (0.5 + dist_edge.clip(-0.5, 0.5))


def conv(a, b):

    pad_width = len(b)
    a_padded = torch.pad(a, pad_width, mode='edge')
    return convolve(a_padded, b)[pad_width:-pad_width] / torch.sum(b)


class InterpolateESF:
    def __init__(self, xp, yp):
        self.xp = xp
        self.yp = yp

    def f(self, x):
        # linear interpolation
        return torch.interp(x, self.xp, self.yp, left=0.0, right=1.0)


def make_slanted_curved_edge(image_shape=(50, 50), angle=5.0, curvature=0.001,
                             low_level=0.25, hi_level=1, black_lvl=0.05,
                             illum_gradient_angle=75.0,
                             illum_gradient_magnitude=+0.05, esf=InterpolateESF([-0.5, 0.5], [0.0, 1.0]).f):


    angle = torch.clip(-angle, a_min=-90.0, a_max=90.0)

    inv_c = 1.0
    step_fctr = 0.0
    angle_offset = 0.0

    if torch.abs(angle) > 45.0:
        angle_offset = -90.0
        image_shape = image_shape. permute(1,0)  # width -> height, and height -> width
    if angle > 45.0:
        step_fctr = -1.0
        inv_c = -1.0

    def midpoint(image_shape):
        return image_shape[0] / 2.0 - 0.5, image_shape[1] / 2.0 - 0.5

    y_midpoint, x_midpoint = midpoint(image_shape)

    slope = slope_from_angle(angle + angle_offset)
    p = polynomial_from_midpoint_slope_and_curvature(y_midpoint, x_midpoint, slope, curvature * inv_c)

    dist_edge = calc_distance(image_shape, p, quadratic_fit=True)


    step_dir = -1 if torch.cos(torch.deg2rad(angle + step_fctr * angle_offset)) < 0 else 1

    im = low_level + (hi_level - low_level) * esf(step_dir * dist_edge)

    # If previously rotated, reverse rotation of image back to the original orientation
    if torch.abs(angle) > 45.0:
        im = im.T[:, ::-1]  # rotate 90° right by transposing and mirroring
        image_shape = image_shape[::-1]  # width -> height, and height -> width
        y_midpoint, x_midpoint = midpoint(image_shape)

    # Apply illumination gradient
    if illum_gradient_magnitude != 0.0:
        slope_gradient = slope_from_angle(illum_gradient_angle - 90.0)
        p = polynomial_from_midpoint_slope_and_curvature(y_midpoint, x_midpoint,
                                                             slope_gradient, 0.0)
        illum_gradient_dist = calc_distance(image_shape, p, quadratic_fit=False)
        illum_gradient = 1 + illum_gradient_dist / (image_shape[0] / 2) * illum_gradient_magnitude
        im = torch.clip((im - black_lvl) * illum_gradient, a_min=0.0, a_max=None) + black_lvl

    return im, dist_edge


def calc_custom_esf(x_length=5.0, x_step=0.01, x_edge=0.0, pixel_fill_factor=1.00,
                    pixel_pitch=1.0, sigma=0.2, show_plots=0):


    def gauss(x, h=0.0, a=1.0, x0=0.0, sigma=1.0):
        return h + a * torch.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    w = pixel_pitch / 2 * torch.sqrt(pixel_fill_factor)  # aperture width

    # 1-d position vector
    x = torch.arange(-x_length / 2, x_length / 2, x_step)

    # ideal edge (step function)
    edge = torch.heaviside(x - x_edge, 0.5)

    # optics line spread function (LSF)
    lsf = gauss(x, x0=x_edge, sigma=sigma)

    # pixel aperture (box filter)
    pixel = torch.heaviside(x - (x_edge - w), 0.5) * torch.heaviside((x_edge + w) - x, 0.5)

    # Convolve edge with lsf and pixel
    edge_lsf = conv(edge, lsf)
    edge_lsf_pixel = conv(edge_lsf, pixel)

    return x, edge_lsf_pixel


def rgb2gray(im_rgb_crop_dark, im_rgb_crop_light, im_rgb):
    r0 = torch.mean(im_rgb_crop_dark[0::2, 0::2])
    r1 = torch.mean(im_rgb_crop_light[0::2, 0::2])
    g0 = torch.mean(im_rgb_crop_dark[0::2, 1::2])
    g1 = torch.mean(im_rgb_crop_light[0::2, 1::2])
    b0 = torch.mean(im_rgb_crop_dark[1::2, 1::2])
    b1 = torch.mean(im_rgb_crop_light[1::2, 1::2])

    k_gr = (g1 - g0) / (r1 - r0)
    k_gb = (g1 - g0) / (b1 - b0)
    k_rb = (r1 - r0) / (b1 - b0)
    pedestal = torch.mean([(g0 - k_gr * r0) / (1 - k_gr),
                        (g1 - k_gr * r1) / (1 - k_gr),
                        (g0 - k_gb * b0) / (1 - k_gb),
                        (g1 - k_gb * b1) / (1 - k_gb),
                        (r0 - k_rb * b0) / (1 - k_rb),
                        (r1 - k_rb * b1) / (1 - k_rb)])

    x = torch.mean([(r1 - pedestal) / (r0 - pedestal),  # x = light / dark luminance ratio
                 (g1 - pedestal) / (g0 - pedestal),
                 (b1 - pedestal) / (b0 - pedestal)])

    gain_r = torch.mean([r0 - pedestal, (r1 - pedestal) / x])
    gain_g = torch.mean([g0 - pedestal, (g1 - pedestal) / x])
    gain_b = torch.mean([b0 - pedestal, (b1 - pedestal) / x])

    gain_image = torch.zeros_like(im_rgb)
    gain_image[0::2, 0::2] = 1 / gain_r
    gain_image[0::2, 1::2] = 1 / gain_g
    gain_image[1::2, 0::2] = 1 / gain_g
    gain_image[1::2, 1::2] = 1 / gain_b

    return (im_rgb - pedestal) * gain_image


