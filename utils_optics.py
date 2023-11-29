# --coding:utf-8--
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision
import torch.nn as nn
from torch.fft import fftshift, fft2
import pandas as pd
from math import pi 
import cv2
from SFR import *
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

NA = 0.2343251
wavelength = 0.5876
f = 12.08984e3
pixelsize = 3.45
BS = 1
savepath = ".\data\output\Lenses54854_001\simulated_1129"
SFRbasepath = ".\data\input\SFR_cal\simulated_1129"
SFRsavepath = ".\data\output\SFR_cal\simulated_1129"

filename_edge_info ='edge_info_1119.csv'
filepath_edge_info = os.path.join(SFRbasepath, filename_edge_info )
data_edge_info= pd.read_csv(filepath_edge_info,header=None,index_col=None)



data_edge_name=data_edge_info.iloc[1:,0].values
delta_y = data_edge_info.iloc[1:,1].values
delta_x = data_edge_info.iloc[1:,2].values
delta_x  = np.array(delta_x ,dtype=np.float32)
delta_y  = np.array(delta_y ,dtype=np.float32)

Field_height=data_edge_info.iloc[1:,3].values
measured_fov = np.array(Field_height,dtype=np.float32)
measured_fov = torch.from_numpy(measured_fov).unsqueeze(1)



def add_noise(sample_edge):
    return torch.poisson(sample_edge / output_FS * n_well_FS) / n_well_FS

# Generate ideal edge
N = 50  # sample ROI 50*50
n_well_FS = 60000  # simulated no. of electrons at full scale for the noise calculation
output_FS = 1.0  # image sensor output at full scale
ideal_edge = make_ideal_slanted_edge((N, N), angle=85.0)
sample0 = add_noise(ideal_edge)
# torchvision.utils.save_image(sample0, savepath+'\sample.tif')
sample =sample0.unsqueeze(0).unsqueeze(0)
pad = nn.ReplicationPad2d(padding=(8, 8, 8, 8))
sample= pad(sample)

# input wavefront
# output: kernel_phs--actual psf which sensor can resolve
#         AP0--visualization PSF
def FourierTransform(WF): 
    M = WF.size(0)
    W =nn.ZeroPad2d(2*M)(WF)
    phase = torch.exp(-1j * 2 * torch.pi * W)
    phase=torch.where(phase==1,0,phase)
    AP = abs(fftshift(fft2(phase))) ** 2

    H = torchvision.transforms.CenterCrop(M)
    AP =H(AP)
    AP0 = AP / torch.max(AP)
    AP=AP0.unsqueeze(0).unsqueeze(0)
    kernel_phs0=F.interpolate(AP, scale_factor=1/15, mode='bilinear', antialias=True)#1/15
    kernel_phs=torch.squeeze(kernel_phs0)
    # plt.imshow(kernel_phs)
    # plt.show()
    # kernel_phs = kernel_phs.numpy()
    # np.savetxt("kernel_phs.csv",kernel_phs)
    # kernel_norm = kernel_phs/torch.max(torch.max(kernel_phs))
    mtf = abs(fftshift(fft2(kernel_phs)))
    mtf = mtf/torch.max(torch.max(mtf))
    # angle = torch.angle(complex)    
    # angle = angle / torch.max(torch.max(angle))
    return  kernel_phs, mtf, AP0



def PlotPSF(PSF, degree=0, j=0):
    rsl = wavelength/(np.tan(np.arcsin(NA))*2)                                                                                             
    pad_f = 1 / 5   # DFT_padding
    rsl = rsl * pad_f   # physical unit 
    
    plt.imshow(PSF, extent=[-25*rsl, 25*rsl,-25*rsl, 25*rsl], cmap="inferno", interpolation="bicubic")#bicubic
    # plt.ylabel('V[μm]')
    # plt.xlabel('U[μm]')
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    # image_name = os.path.join(savepath, str(j)+' initial_PSF of FoV='+str(degree)+'degree.png')
    # plt.savefig(image_name,bbox_inches='tight', pad_inches = -0.1)
    plt.show()

    return rsl

def PlotMTF(MTF0,rsl):

    MTF = torch.squeeze (MTF0, dim=0)
    fre_max=1/rsl/1e-3
    fre=torch.arange(0,1/2,1/51)*fre_max
    SFR=MTF[25,25:]
    plt.plot(fre,SFR)
    plt.xlim(0,2000)
    plt.ylim(0,1)
    plt.title("MTF")
    plt.xlabel("Spatial Frequency in cycles per mm")
    plt.ylabel("Modulus of the OTF")

    plt.show()




def zernike_wavefront(zer_co, A,rou,BS):
    WF = torch.zeros(BS,256,256)
    WF = WF
    for i in range(21):
        # WF = WF + A[...,i] *zer_co[i,:].unsqueeze(2)
        WF = WF + A[..., i] *zer_co[i]
    WF = torch.where(rou >= 1, 0, WF)
    return WF



def fitting_prepare(BS):
    m = 256
    x = torch.linspace(-1, 1, m)
    [Y, X] = torch.meshgrid(x, x)
    rou = abs(torch.sqrt(X ** 2 + Y ** 2))
    theta = torch.arctan2(Y, X)
    mask = rou.reshape(256 * 256, 1) < 1

    A = torch.zeros((256,256,21))
    A[..., 0] = 1  
    A[..., 1] = 4 ** .5 * rou * torch.sin(theta)
    A[..., 2] = 3 ** .5 * (2 * rou ** 2 - 1) 
    A[..., 3] = 6 ** .5 * rou ** 2 * torch.cos(2 * theta) 
    A[..., 4] = 8 ** .5 * (3 * rou ** 3 - 2 * rou) * torch.sin(theta)
    A[..., 5] = 8 ** .5 * rou ** 3 * torch.sin(3 * theta)
    A[..., 6] = 5 ** .5 * (6 * rou ** 4 - 6 * rou ** 2 + 1)
    A[..., 7] = 10 ** .5 * (4 * rou ** 4 - 3 * rou ** 2) * torch.cos(2 * theta)
    A[..., 8] = 10 ** .5 * rou ** 4 * torch.cos(4 * theta) 
    A[..., 9] = 12 ** .5 * (10 * rou ** 5 - 12 * rou ** 3 + 3 * rou) * torch.sin(theta)
    A[..., 10] = 12 ** .5 * (5 * rou ** 5 - 4 * rou ** 3) * torch.sin(3 * theta)
    A[..., 11] = 12 ** .5 * rou ** 5 * torch.sin(5 * theta)
    A[..., 12] = 7 ** .5 * (20 * rou ** 6 - 30 * rou ** 4 + 12 * rou ** 2 - 1)
    A[..., 13] = 14 ** .5 * (15 * rou ** 6 - 20 * rou ** 4 + 6 * rou ** 2) * torch.cos(2 * theta)
    A[..., 14] = 14 ** .5 * (6 * rou ** 6 - 5 * rou ** 4) * torch.cos(4 * theta)
    A[..., 15] = 14 ** .5 * rou ** 6 * torch.cos(6 * theta)
    A[..., 16] = 16 ** .5 * (35 * rou ** 7 - 60 * rou ** 5 + 30 * rou ** 3 - 4 * rou) * torch.sin(theta)
    A[..., 17] = 16 ** .5 * (21 * rou ** 7 - 30 * rou ** 5 + 10 * rou ** 3) * torch.sin(3 * theta)
    A[..., 18] = 16 ** .5 * (7 * rou ** 7 - 6 * rou ** 5) * torch.sin(5 * theta)
    A[..., 19] = 16 ** .5 * rou ** 7 * torch.sin(7 * theta)
    A[..., 20] = 9 ** .5 * (70 * rou ** 8 - 140 * rou ** 6 + 90 * rou ** 4 - 20 * rou ** 2 + 1)  # Z60

    B = A.reshape(256**2,1,21)
    c = []
    for i in range(21):
        c.append(torch.masked_select(B[...,i],mask))

    matrix = torch.stack(c,dim=1)
    matrix = matrix.repeat(BS,1,1)
    mask = mask.repeat(BS,1,1)
    A = A.repeat(BS,1,1,1)
    rou = rou.repeat(BS,1,1)
    return A,rou




def wavefront_fitting(WF,matrix,mask,BS):
    data = torch.masked_select(WF.reshape(BS,256**2,1),mask)
    length = len(data)
    data = data.reshape(BS,length//BS,1)
    matrixT = matrix.transpose(1,2)
    B = torch.linalg.inv(torch.bmm(matrixT,matrix)  + 0.0001 * torch.eye(21))
    temp = torch.bmm(B,matrixT)
    zer_co = torch.bmm(temp,data)
    fmatrix = torch.bmm(matrix, zer_co)
    error = torch.sum(abs(fmatrix - data),dim=1) / length
    return zer_co,error



def up_limit_of_zer(zerpath,fov):
    zer = pd.read_excel(zerpath, sheet_name='Sheet1', header=None, index_col=None)
    info = pd.read_excel(zerpath, sheet_name='Sheet2', header=None, index_col=None)
    zer = zer.drop(columns=[1, 4, 7, 9, 12, 14, 15, 17, 19, 22, 24, 26, 29, 31, 33, 35]).values
    fov_info = np.expand_dims(info.iloc[:, 3].values, 1)
    fov_info = np.repeat(fov_info,21,axis=1)
    mask = (fov_info <= fov)
    select_zer = torch.masked_select(torch.from_numpy(zer), torch.from_numpy(mask))
    select_zer = select_zer.reshape(-1, 21)
    max_zer,max_index = torch.max(select_zer,dim=0)
    return max_zer



def fieldmatrix_prepare(field):
# Robert W. Gray et al, An analytic expression for the field dependence of 
# Zernike polynomials in rotationally symmetric optical systems.
    n=len(field)
    one=torch.ones(n,1)
    H=torch.tan(field*pi/180) #degree to hight
    H0=torch.cat((one,torch.pow(H, 2) ,torch.pow(H, 4)),dim=1)
    H1=torch.cat((H,torch.pow(H, 3) ,torch.pow(H, 5)),dim=1)
    H=torch.stack((H0,H1,H0,H0,H1,H1,H0,
                   H0,H0,H1,H1,H1,H0,H0,
                   H0,H0,H1,H1,H1,H1,H0
                   ),dim=0)
    return H


def fov_preparation(y,x):
    
    if x>=0 and y>=0:
        angle=np.arctan(y/x)
    elif x<0 and y<0:
        angle=np.arctan(y/x)+np.pi
    elif x>0 and y<0:
        angle=np.arctan(y/x)
    elif x<0 and y>0:
        angle=np.arctan(y/x)-np.pi
    angle=torch.tensor(angle,dtype=torch.float32)
    return angle-np.pi/2





def get_rot_mat(theta,point_x,point_y):
    # print(theta,point_x,point_y)
    # M = torch.tensor([[torch.cos(theta), -torch.sin(theta),
    #                (1-torch.cos(theta))*point_x+point_y*torch.sin(theta)],
    #               [torch.sin(theta), torch.cos(theta),
    #                (1-torch.cos(theta))*point_y-point_x*torch.sin(theta)]])
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])
    return M

# def binarization(img_t):
#     z = torch.zeros(img_t.shape)
#     o = torch.ones(img_t.shape)
#     # print(img_t.mean())
#     img_t_b = torch.where(img_t < img_t.mean().item(), z, o)
#     return img_t_b

# def cal(img, k=0, l=0):
#     rows = img.shape[0]
#     cols = img.shape[1]

#     ret = 0
#     for y in range(rows):
#         for x in range(cols):
#             f_x_y = img[y, x]
#             ret += (x ** k) * (y ** l) * f_x_y
#     return ret


def rot_img(x, theta):

    # binary_img = binarization(x.squeeze())
    # plt.imshow( binary_img)
    # plt.show()
    # m00 = cal(binary_img, 0, 0)
    # m10 = cal(binary_img, 1, 0)
    # m01 = cal(binary_img, 0, 1)
    # print(m00,m10,m01)
    # c_x = m10 / m00
    # c_y = m01 / m00

    rot_mat = get_rot_mat(theta,0,0)[None, ...].repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(),align_corners=True)
    x = F.grid_sample(x, grid, align_corners=True)

    return x




#fit Field dependence of Zernike polynomials(least square approximation)
#initial ZernikeCoefficients field dependence generation
def ZernikeCoefficientsfit(sampled_field,sampled_Zer):
    
    H=fieldmatrix_prepare(sampled_field)
    HT = H.mT
    C_pre = torch.linalg.inv(torch.bmm(HT,H)) 
    temp = torch.bmm(C_pre,HT)
    C = torch.bmm(temp,sampled_Zer) 
    f0 = torch.bmm(H,C)
    err = torch.sum(abs(f0 - sampled_Zer),dim=1) / torch.sum(sampled_Zer,dim=1)

    return C

# visualization of every Zernike polynomials
zero=torch.tensor([0,0,0,0,0,0,0,
                  1,0,0,0,0,0,0,
                  0,0,0,0,0,0,0]
                  )


def ImagingSimulation(C):
#plot Field dependence of Zernike polynomials
    # full_field0 = torch.linspace(1, 24, steps=24)
    # full_field = torch.unsqueeze (full_field0, dim=1)
    # H=fieldmatrix_prepare(full_field)
    measured_H=fieldmatrix_prepare(measured_fov)
    full_Zer = torch.bmm(measured_H,C) # measured_H
    zernikecoefficients=torch.squeeze(full_Zer,dim=2)
    oversampling = 4
    A,rou = fitting_prepare(BS)

    for j in range(53):
        zer_cor=zernikecoefficients[:,j] 
        wf0=zernike_wavefront(zer_cor, A,rou,BS)
        wf=torch.squeeze(wf0,dim=0)
        psf,MTF,visualpsf = FourierTransform(wf)
        psf=psf/torch.sum(psf)

        y = delta_y[j]
        x = delta_x[j]
        rotate_angle=fov_preparation(y,x)

        # print('旋转角度',rotate_angle)
        # plt.imshow(psf)
        # PlotPSF(visualpsf)
    
        kernel0 = psf 
        kernel =kernel0.unsqueeze(0).unsqueeze(0)
        kernel = rot_img(kernel,rotate_angle) # Rotate image by 90 degrees.
        # torchvision.utils.save_image(kernel.squeeze(0).squeeze(0), savepath+'\psf%d.tif'%(j))
        # plt.imshow(kernel.squeeze(0).squeeze(0))
        # plt.show()
        blur_edge0 = F.conv2d(sample,kernel,stride=1,padding=0)
        blur_edge =blur_edge0.squeeze(0).squeeze(0)
        # torchvision.utils.save_image(blur_edge, savepath+'\blur_edge%d.tif'%(j))
        simulate_mtfa3= calc_sfr(blur_edge, oversampling).reshape(1)
        if j==0:
            simulate_mtfa=simulate_mtfa3
        simulate_mtfa=torch.cat((simulate_mtfa, simulate_mtfa3),dim=-1)
        # simulate_mtfa_print=simulate_mtfa.detach().numpy()
        # np.savetxt(SFRsavepath+"\simulate_mtfa1.csv",simulate_mtfa_print)

    return simulate_mtfa


def Loss(C, mtfa):
    simulate_mtfa = ImagingSimulation(C)
    loss=torch.sum(torch.abs(mtfa-simulate_mtfa))/simulate_mtfa.size(0)
    return loss