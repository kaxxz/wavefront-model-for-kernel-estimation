# --coding:utf-8--

from utils_optics import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


imgbasepath = '.\data\input\Lenses54854_001'
savepath = '.\data\output\Lenses54854_001\simulated_1129'
filename_zer_cor ='zernike_samplefield.xlsx'
SFRbasepath = ".\data\input\SFR_cal\simulated_1129"
SFRsavepath = ".\data\output\SFR_cal\simulated_1129"
filename_edge_info ='edge_info_1119.csv'
filename_initialC_info ='initial_C.csv'
filepath_zercor = os.path.join(imgbasepath, filename_zer_cor)
filepath_edge_info = os.path.join(SFRbasepath, filename_edge_info )
data_zer_cor = pd.read_excel(filepath_zercor,sheet_name='54854',header=None,index_col=None)#_simulated
data_edge_info= pd.read_csv(filepath_edge_info,header=None,index_col=None)

# unit:um
NA = 0.2343251
wavelength = 0.5876
f = 12.08984e3
pixelsize = 3.45
Maximum_Radial_Field  = 24

zer_co = data_zer_cor.iloc[1:22,:].values
FoV = data_zer_cor.iloc[0,:].values
zer_co = np.array(zer_co,dtype=np.float32)
zer_co = torch.from_numpy(zer_co)
zer_co = torch.unsqueeze(zer_co, dim=-1)
FoV = np.array(FoV,dtype=np.float32)
FoV = torch.from_numpy(FoV)
FoV = torch.unsqueeze (FoV, dim=1)

# intial Damping factor 
lam = torch.tensor([  0.001,0.001,0.001,0.001,0.001,0.001,0.001,
                    0.001,0.001,0.001,0.001,0.001,0.001,0.001,
                    0.001,0.001,0.001,0.001,0.001,0.001,0.001
                    ])
lam = torch.diag(lam)
# Levenberg-Marquardt optimization function
# lam较小的时候，牛顿高斯法，较大的时候近似于梯度下降法。
# lam不应该是一个固定值而应该是一个根据不同系数特征的对角矩阵。这就要弄清楚21项里面哪几个最为关键，比如散焦defocus、彗差coma、像散
def lm_optimize(loss, mtfa, C, max_iter=100):
    for i in range(max_iter):

        
        r = loss(C, mtfa)
        print("Iteration: %d/%d"%(i,max_iter))
        print("loss: %.8f"%r)
        
        r.backward(retain_graph=True)
        J = C.grad
        J = torch.squeeze(J).T
        A = J.T @ J + lam * torch.eye(J.shape[1])
        g = J.T * r
        step = torch.linalg.solve(A, g).unsqueeze(2)
        C_new = torch.zeros(size=C.size())

        C_new.data = C.data - 3*(0.99**i)*step.data
        # update Damping factor
        # TODO

        
        # print('Step',-step.data)
        if torch.linalg.norm(C_new.data - C.data) < 1e-8:  # Convergence check
            break
        C.data= C_new.data

        C.grad.data.zero_()
        if i == 30 or i == 60 or i == 90:
            disturbed_C_print =C.squeeze().detach().numpy()
            np.savetxt(savepath +"\disturbed_C1129_%d.csv"%(i),disturbed_C_print ) 
    return C


# Initial field-dependence coefficent
# C=ZernikeCoefficientsfit(FoV, zer_co)
# C_print =C.squeeze().detach().numpy()
# np.savetxt(savepath +"\simulated_C0.csv",C_print )
# C.requires_grad_(True)

# C is zernike field-dependence coefficent -- a 21*3 matrix
C_info = os.path.join(savepath, filename_initialC_info )
data = pd.read_csv(C_info, delimiter=' ',header=None).values.astype('float32')
C = torch.tensor(data).unsqueeze(2)
C.requires_grad_(True)

data_edge_name = data_edge_info.iloc[1:,0].values
# print(data_edge_name)
i=0
oversampling = 4
for edge_name in data_edge_name:
    read_path = os.path.join(SFRbasepath, edge_name)
    im = plt.imread(read_path)
    im = relative_luminance(im)
    im = torch.from_numpy(im).to(torch.float32)
    mtfa3 = calc_sfr(im, oversampling).reshape(1)
    if i==0:
        mtfa = mtfa3
    else:
        mtfa = torch.cat((mtfa, mtfa3),dim=-1)
    i=1

# print(mtfa)
# measured_mtf=mtfa.numpy()
# np.savetxt(SFRsavepath+"\measured_mtfa1.csv",measured_mtf)


# # Run Levenberg-Marquardt optimization
disturbed_C = lm_optimize(Loss, mtfa, C)


disturbed_C_print = disturbed_C.squeeze().detach().numpy()
np.savetxt(savepath +"\disturbed_C_1129_1353.csv",disturbed_C_print )
