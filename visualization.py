# --coding:utf-8--
# 棋盘格切割程序
# 输入：Zernike场系数C，模糊edge
# 输出：全视场PSF，清晰edge
# 功能：去卷积

from utils_optics import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
import pandas as pd
from SFR import *
from skimage import color, data, restoration

savepath = ".\data\output\Lenses54854_001\simulated_1129"
SFRbasepath = ".\data\input\SFR_cal\simulated_1129"
SFRsavepath = ".\data\output\SFR_cal\simulated_1129"

filename_edge_info = 'testedge.csv'
filename_disturbedC_info = 'disturbed_C1129_60.csv'
filename_initial_C_info = 'initial_C.csv'
filename_simulated_C_info = 'simulated_C.csv'


filepath_edge_info = os.path.join(savepath, filename_edge_info )
data_edge_info = pd.read_csv(filepath_edge_info,header=None,index_col=None)


disturbedC_info = os.path.join(savepath, filename_disturbedC_info )
data = pd.read_csv(disturbedC_info, delimiter=' ',header=None).values.astype('float32')
disturbed_C = torch.tensor(data).unsqueeze(2)

initial_C_info = os.path.join(savepath, filename_initial_C_info )
data_initial = pd.read_csv(initial_C_info, delimiter=' ',header=None).values.astype('float32')
initial_C = torch.tensor(data_initial).unsqueeze(2)

simulated_C_info = os.path.join(savepath, filename_simulated_C_info )
data_simulated = pd.read_csv(simulated_C_info, delimiter=' ',header=None).values.astype('float32')
simulated_C = torch.tensor(data_simulated).unsqueeze(2)


#Generate ideal edge
N = 50  # sample ROI size
n_well_FS = 10000  # simulated no. of electrons at full scale for the noise calculation
output_FS = 1.0  # image sensor output at full scale
def add_noise(sample_edge):
    return torch.poisson(sample_edge / output_FS * n_well_FS) / n_well_FS
ideal_edge = make_ideal_slanted_edge((N, N), angle=85.0)
sample0 = add_noise(ideal_edge)
# torchvision.utils.save_image(sample0, savepath+'\sample0.png')
sample =sample0.unsqueeze(0).unsqueeze(0)
pad = nn.ReplicationPad2d(padding=(8, 8, 8, 8))
sample = pad(sample)



def visualization(C,j,measured_H,C_name):

    full_Zer = torch.bmm(measured_H,C)
    zernikecoefficients = torch.squeeze(full_Zer,dim=2)
    zer_cor = zernikecoefficients[:,j] 
    # print(C_name,zer_cor)
    A,rou = fitting_prepare(BS)
    wf0 = zernike_wavefront(zer_cor, A,rou,BS)
    wf = torch.squeeze(wf0,dim=0)
    psf,MTF,visualpsf = FourierTransform(wf)
    # PlotPSF(visualpsf,NA,wavelength,savepath,j)
    return psf
    #    torchvision.utils.save_image(psf, savepath+'\piston_psf%d.png'%(j))
    #     PSF=psf/torch.sum(psf)
    #     th=8
    #     PlotPSF(psf,NA,wavelength,savepath,j,th)
    # plt.show(block=False)
    # plt.pause(3) # 3 seconds, I use 1 usually
    # plt.close("all")
    #     oversampling = 4
    #     kernel0 = PSF 
    #     kernel =kernel0.unsqueeze(0).unsqueeze(0)

    #     outcome0 = F.conv2d(sample,kernel,stride=1,padding=0)
    #     outcome =outcome0.squeeze(0).squeeze(0)

    #     # torchvision.utils.save_image(outcome, savepath+'\outcome%d.png'%(i))

    #     simulate_mtfa3= calc_sfr(outcome, oversampling).reshape(1)
    #     if j==0:
    #         simulate_mtfa=simulate_mtfa3
    #     simulate_mtfa=torch.cat((simulate_mtfa, simulate_mtfa3),dim=-1)
    #     # simulate_mtfa_print=simulate_mtfa.numpy()
    #     # np.savetxt(SFRsavepath+"\simulate_mtfa1.csv",simulate_mtfa_print)
    #     # simulate_mtfa_mean= torch.mean(simulate_mtfa)
        

    #     # def hook_y(grad):
    #     #     print('打印非叶子节点: ',grad) 

    #     # full_Zer.register_hook(hook_y) 


    # return simulate_mtfa



if __name__ == '__main__':

    data_edge_name = data_edge_info.iloc[1:,0].values
    measured_fov0 = data_edge_info.iloc[1:,3].values
    measured_fov = np.array(measured_fov0,dtype=np.float32)
    measured_fov = torch.from_numpy(measured_fov).unsqueeze(1)
    measured_H = fieldmatrix_prepare(measured_fov)
    
    delta_y = data_edge_info.iloc[1:,1].values
    delta_x = data_edge_info.iloc[1:,2].values
    delta_x  = np.array(delta_x ,dtype=np.float32)
    delta_y  = np.array(delta_y ,dtype=np.float32)

    j = 0
    for edge_name in data_edge_name:
        read_path = os.path.join(SFRbasepath, edge_name)
        captured_blur_edge0 = plt.imread(read_path).astype(np.float32)  
        captured_blur_edge = captured_blur_edge0[:,:,0]/255
        # plt.imshow(captured_blur_edge,cmap='gray')
        # plt.title("captured blur edge")
        # plt.show()
      
        y = delta_y[j]
        x = delta_x[j]
        rotate_angle = fov_preparation(y,x)
        print(x,y,rotate_angle)

        simulated_psf = visualization(simulated_C,j,measured_H,'simulated_zer')
        initial_psf = visualization(initial_C,j,measured_H,'initial_zer')
        disturbed_psf = visualization(disturbed_C,j,measured_H,'disturbed_zer')

        simulated_kernel0 = simulated_psf/torch.sum(simulated_psf)
        simulated_kernel = simulated_kernel0.unsqueeze(0).unsqueeze(0)

        initial_kernel0 = initial_psf/torch.sum(initial_psf)
        initial_kernel = initial_kernel0.unsqueeze(0).unsqueeze(0)

        disturbed_kernel0 = disturbed_psf/torch.sum(disturbed_psf)
        disturbed_kernel = disturbed_kernel0.unsqueeze(0).unsqueeze(0)

        # conv_transpose2d 没用
        # outcome0 = F.conv_transpose2d(ima,kernel,stride=1,padding=8)
        # outcome =outcome0.squeeze(0).squeeze(0)        
        # pad = nn.ReplicationPad2d(padding=(8, 8, 8, 8))
        # im= pad(im)

        simulated_kernel_r = rot_img(simulated_kernel,rotate_angle)
        initial_kernel_r = rot_img(initial_kernel,rotate_angle)
        disturbed_kernel_r = rot_img(disturbed_kernel,rotate_angle)

        simulated_blur_edge0 = F.conv2d(sample,simulated_kernel_r,stride=1,padding=0)
        simulated_blur_edge = simulated_blur_edge0.squeeze(0).squeeze(0)
        plt.imshow(simulated_blur_edge)
        plt.title("simulated blur edge")
        plt.show()
        torchvision.utils.save_image(simulated_blur_edge, edge_name)


        j=j+1
        disturbed_psf_r = disturbed_kernel_r.squeeze().numpy()
        initial_psf_r = initial_kernel_r.squeeze().numpy()
        simulated_psf_r = simulated_kernel_r.squeeze().numpy()
        

        captured_blur_edge1 = np.pad(captured_blur_edge, ((8,8),(8,8)),'edge')
        simulated_blur_edge = simulated_blur_edge.numpy()
        simulated_blur_edge1 = np.pad(simulated_blur_edge, ((8,8),(8,8)),'edge')
 
        # deconv

        deconvolved_simulated = restoration.richardson_lucy(captured_blur_edge1, simulated_psf_r/np.sum(simulated_psf_r), num_iter=50)
        deconvolved_disturbed = restoration.richardson_lucy(captured_blur_edge1, disturbed_psf_r/np.sum(disturbed_psf_r), num_iter=50)
        
        # deconvolved_disturbed  = restoration.wiener(captured_blur_edge, psf/np.sum(psf), 0.2, reg=None, is_real=True, clip=True)
        deconvolved_simulated = deconvolved_simulated[8:58,8:58]
        deconvolved_disturbed = deconvolved_disturbed[8:58,8:58]

        # H = torchvision.transforms.CenterCrop(100)
        # outcome =H(deconvolved)
        # torchvision.utils.save_image(outcome, savepath+'\\1120_initial_wienerdeconv_'+edge_name+'.png')
 
        # Edge visualization
        fig0, ax0 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4),
                            sharex=True, sharey=True)
        plt.gray()

        ax0[0].imshow(simulated_blur_edge, vmin=simulated_blur_edge.min(), vmax=simulated_blur_edge.max())
        ax0[0].axis('off')
        ax0[0].set_title('simulated_blur_edge',fontsize=8)

        ax0[1].imshow(deconvolved_simulated, vmin=deconvolved_simulated.min(), vmax=deconvolved_simulated.max())
        ax0[1].axis('off')
        ax0[1].set_title('deconvolved_simulated',fontsize=8)

        ax0[2].imshow(captured_blur_edge, vmin=captured_blur_edge.min(), vmax=captured_blur_edge.max())
        ax0[2].axis('off')
        ax0[2].set_title('captured_blur_edge',fontsize=8)


        ax0[3].imshow(deconvolved_disturbed,vmin=deconvolved_disturbed.min(), vmax=deconvolved_disturbed.max())
        ax0[3].axis('off')
        ax0[3].set_title('deconvolved_disturbed',fontsize=8)
        fig0.tight_layout()
        plt.savefig(savepath+'\\deconv\\edgedeconv_compare11291633'+'%d'%(j)+edge_name,dpi=300)
        plt.show()


        # PSF visualization
        fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(6, 2),
                            sharex=True, sharey=True)

        ax1[0].imshow(initial_psf_r,cmap="inferno")
        ax1[0].axis('off')
        ax1[0].set_title('initial_psf_r',fontsize=8)

        ax1[1].imshow(disturbed_psf_r,cmap="inferno")
        ax1[1].axis('off')
        ax1[1].set_title('disturbed_psf_r',fontsize=8)


        
        ax1[2].imshow(simulated_psf_r,cmap="inferno")
        ax1[2].axis('off')
        ax1[2].set_title('simulated_psf_r',fontsize=8)



        fig1.tight_layout()
        plt.savefig(savepath+'\\deconv\\psf_compare11291633'+'%d'%(j)+edge_name,dpi=300)
        plt.show()


        # fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(6, 2),
        #                     sharex=True, sharey=True)

        # ax2[0].imshow(initial_psf,cmap="inferno")
        # ax2[0].axis('off')
        # ax2[0].set_title('initial_psf',fontsize=8)

        # ax2[1].imshow(disturbed_psf,cmap="inferno")
        # ax2[1].axis('off')
        # ax2[1].set_title('disturbed_psf',fontsize=8)


        
        # ax2[2].imshow(simulated_psf,cmap="inferno")
        # ax2[2].axis('off')
        # ax2[2].set_title('simulated_psf',fontsize=8)

        # fig2.tight_layout()
        # plt.savefig(savepath+'\\deconv\\nonr_vispsf_compare'+'%d'%(j)+edge_name,dpi=300)
        # plt.show()

