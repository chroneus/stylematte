import torch
import numpy as np
import torch.nn.functional as F
import torch

def gauss_kernel(size=5, device=torch.device('cpu'), channels=1):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LaplacianLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=1, device=torch.device('cuda')):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
            
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice



def calculate_sad_mse_mad(predict_old,alpha,trimap):
    predict = np.copy(predict_old)
    pixel = float((trimap == 128).sum())
    predict[trimap == 255] = 1.
    predict[trimap == 0  ] = 0.
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    if pixel==0:
        pixel = trimap.shape[0]*trimap.shape[1]-float((trimap==255).sum())-float((trimap==0).sum())
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff
    
def calculate_sad_mse_mad_whole_img(predict, alpha):
    pixel = predict.shape[0]*predict.shape[1]
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff	

def calculate_sad_fgbg(predict, alpha, trimap):
    sad_diff = np.abs(predict-alpha)
    weight_fg = np.zeros(predict.shape)
    weight_bg = np.zeros(predict.shape)
    weight_trimap = np.zeros(predict.shape)
    weight_fg[trimap==255] = 1.
    weight_bg[trimap==0  ] = 1.
    weight_trimap[trimap==128  ] = 1.
    sad_fg = np.sum(sad_diff*weight_fg)/1000
    sad_bg = np.sum(sad_diff*weight_bg)/1000
    sad_trimap = np.sum(sad_diff*weight_trimap)/1000
    return sad_fg, sad_bg

def compute_gradient_whole_image(pd, gt):
    from scipy.ndimage import gaussian_filter
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x**2 + pd_y**2)
    gt_mag = np.sqrt(gt_x**2 + gt_y**2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map) / 10
    return loss

def compute_connectivity_loss_whole_image(pd, gt, step=0.1):
    from scipy.ndimage import morphology
    from skimage.measure import label, regionprops
    h, w = pd.shape
    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]
        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords
        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1
        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]
        dist_maps = morphology.distance_transform_edt(omega==0)
        dist_maps = dist_maps / dist_maps.max()
    l_map[l_map == -1] = 1
    d_pd = pd - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
    return loss