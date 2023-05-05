# modified from Github repo: https://github.com/JizhiziLi/P3M
# added inference code for other networks


import torch
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import tqdm
import logging
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from models import *
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

config = OmegaConf.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'config/test_ours.yaml'))
device = 'cuda'

##############################
# Training loses for P3M-10k
##############################


def get_crossentropy_loss(gt, pre):
    gt_copy = gt.clone()
    gt_copy[gt_copy == 0] = 0
    gt_copy[gt_copy == 255] = 2
    gt_copy[gt_copy > 2] = 1
    gt_copy = gt_copy.long()
    gt_copy = gt_copy[:, 0, :, :]
    criterion = nn.CrossEntropyLoss()
    entropy_loss = criterion(pre, gt_copy)

    return entropy_loss


def get_alpha_loss(predict, alpha, trimap):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    diff = predict - alpha_f
    diff = diff * weighted
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
    alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1.)

    return alpha_loss_weighted


def get_alpha_loss_whole_img(predict, alpha):
    weighted = torch.ones(alpha.shape).cuda()
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    diff = predict - alpha_f
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
    alpha_loss = alpha_loss.sum()/(weighted.sum())

    return alpha_loss

# Laplacian loss is refer to
# https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError('kernel size must be uneven')
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    def gaussian(x): return np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :]).cuda()

    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    ''' convolve img with a gaussian kernel that has been built with build_gauss_kernel '''
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')

    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []
    for _ in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyr.append(current)

    return pyr


def get_laplacian_loss(predict, alpha, trimap):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    alpha_f = alpha_f.clone()*weighted
    predict = predict.clone()*weighted
    gauss_kernel = build_gauss_kernel(
        size=5, sigma=1.0, n_channels=1, cuda=True)
    pyr_alpha = laplacian_pyramid(alpha_f, gauss_kernel, 5)
    pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
    laplacian_loss_weighted = sum(F.l1_loss(a, b)
                                  for a, b in zip(pyr_alpha, pyr_predict))

    return laplacian_loss_weighted


def get_laplacian_loss_whole_img(predict, alpha):
    alpha_f = alpha / 255.
    alpha_f = alpha_f.cuda()
    gauss_kernel = build_gauss_kernel(
        size=5, sigma=1.0, n_channels=1, cuda=True)
    pyr_alpha = laplacian_pyramid(alpha_f, gauss_kernel, 5)
    pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
    laplacian_loss = sum(F.l1_loss(a, b)
                         for a, b in zip(pyr_alpha, pyr_predict))

    return laplacian_loss


def get_composition_loss_whole_img(img, alpha, fg, bg, predict):
    weighted = torch.ones(alpha.shape).cuda()
    predict_3 = torch.cat((predict, predict, predict), 1)
    comp = predict_3 * fg + (1. - predict_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12)
    comp_loss = comp_loss.sum()/(weighted.sum())

    return comp_loss

##############################
# Test loss for matting
##############################


def calculate_sad_mse_mad(predict_old, alpha, trimap):
    predict = np.copy(predict_old)
    pixel = float((trimap == 128).sum())
    predict[trimap == 255] = 1.
    predict[trimap == 0] = 0.
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    if pixel == 0:
        pixel = trimap.shape[0]*trimap.shape[1] - \
            float((trimap == 255).sum())-float((trimap == 0).sum())
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
    weight_fg[trimap == 255] = 1.
    weight_bg[trimap == 0] = 1.
    weight_trimap[trimap == 128] = 1.
    sad_fg = np.sum(sad_diff*weight_fg)/1000
    sad_bg = np.sum(sad_diff*weight_bg)/1000
    sad_trimap = np.sum(sad_diff*weight_trimap)/1000

    return sad_fg, sad_bg


def compute_gradient_whole_image(pd, gt):
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
        dist_maps = morphology.distance_transform_edt(omega == 0)
        dist_maps = dist_maps / dist_maps.max()
    l_map[l_map == -1] = 1
    d_pd = pd - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000

    return loss


def gen_trimap_from_segmap_e2e(segmap):
    trimap = np.argmax(segmap, axis=1)[0]
    trimap = trimap.astype(np.int64)
    trimap[trimap == 1] = 128
    trimap[trimap == 2] = 255

    return trimap.astype(np.uint8)


def get_masked_local_from_global(global_sigmoid, local_sigmoid):
    values, index = torch.max(global_sigmoid, 1)
    index = index[:, None, :, :].float()
    # index <===> [0, 1, 2]
    # bg_mask <===> [1, 0, 0]
    bg_mask = index.clone()
    bg_mask[bg_mask == 2] = 1
    bg_mask = 1 - bg_mask
    # trimap_mask <===> [0, 1, 0]
    trimap_mask = index.clone()
    trimap_mask[trimap_mask == 2] = 0
    # fg_mask <===> [0, 0, 1]
    fg_mask = index.clone()
    fg_mask[fg_mask == 1] = 0
    fg_mask[fg_mask == 2] = 1
    fusion_sigmoid = local_sigmoid*trimap_mask + fg_mask

    return fusion_sigmoid


def get_masked_local_from_global_test(global_result, local_result):
    weighted_global = np.ones(global_result.shape)
    weighted_global[global_result == 255] = 0
    weighted_global[global_result == 0] = 0
    fusion_result = global_result * \
        (1.-weighted_global)/255+local_result*weighted_global
    return fusion_result


def inference_once(model, scale_img, scale_trimap=None):
    tensor_img = torch.from_numpy(scale_img[:, :, :]).permute(2, 0, 1).cuda()
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0).float()
    # pred_global, pred_local, pred_fusion = model(input_t)[:3]
    pred_fusion = model(input_t)[:3]
    pred_global = pred_fusion
    pred_local = pred_fusion

    pred_global = pred_global.data.cpu().numpy()
    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local.data.cpu().numpy()[0, 0, :, :]
    pred_fusion = pred_fusion.data.cpu().numpy()[0, 0, :, :]

    return pred_global, pred_local, pred_fusion


def inference_img(model, img):
    h, w, _ = img.shape
    if h % 8 != 0 or w % 8 != 0:
        img = cv2.copyMakeBorder(img, 8-h % 8, 0, 8-w %
                                 8, 0, cv2.BORDER_REFLECT)

    tensor_img = torch.from_numpy(img).permute(2, 0, 1).cuda()
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0).float()
    with torch.no_grad():
        out = model(input_t)
    result = out[0][:, -h:, -w:].cpu().numpy()

    return result[0]


def test_am2k(model):
    ############################
    # Some initial setting for paths
    ############################
    ORIGINAL_PATH = config['datasets']['am2k']['validation_original']
    MASK_PATH = config['datasets']['am2k']['validation_mask']
    TRIMAP_PATH = config['datasets']['am2k']['validation_trimap']
    img_paths = glob.glob(ORIGINAL_PATH+'/*.jpg')

    ############################
    # Start testing
    ############################
    sad_diffs = 0.
    mse_diffs = 0.
    mad_diffs = 0.
    grad_diffs = 0.
    conn_diffs = 0.
    sad_trimap_diffs = 0.
    mse_trimap_diffs = 0.
    mad_trimap_diffs = 0.
    sad_fg_diffs = 0.
    sad_bg_diffs = 0.

    total_number = len(img_paths)
    log('===============================')
    log(
        f'====> Start Testing\n\t--Dataset: AM2k\n\t-\n\t--Number: {total_number}')

    for img_path in tqdm.tqdm(img_paths):
        img_name = (img_path.split('/')[-1])[:-4]
        alpha_path = MASK_PATH+img_name+'.png'
        trimap_path = TRIMAP_PATH+img_name+'.png'
        pil_img = Image.open(img_path)
        img = np.array(pil_img)
        trimap = np.array(Image.open(trimap_path))
        alpha = np.array(Image.open(alpha_path))/255.
        img = img[:, :, :3] if img.ndim > 2 else img
        trimap = trimap[:, :, 0] if trimap.ndim > 2 else trimap
        alpha = alpha[:, :, 0] if alpha.ndim > 2 else alpha

        with torch.no_grad():
            torch.cuda.empty_cache()
            predict = inference_img(model, img)

            sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(
                predict, alpha, trimap)
            sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(
                predict, alpha)
            sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(
                predict, alpha, trimap)
            conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
            grad_diff = compute_gradient_whole_image(predict, alpha)

            log(f'[{img_paths.index(img_path)}/{total_number}]\n'
                f'Image:{img_name}\n'
                f'sad:{sad_diff}\n'
                f'mse:{mse_diff}\n'
                f'mad:{mad_diff}\n'
                f'sad_trimap:{sad_trimap_diff}\n'
                f'mse_trimap:{mse_trimap_diff}\n'
                f'mad_trimap:{mad_trimap_diff}\n'
                f'sad_fg:{sad_fg_diff}\n'
                f'sad_bg:{sad_bg_diff}\n'
                f'conn:{conn_diff}\n'
                f'grad:{grad_diff}\n'
                f'-----------')

            sad_diffs += sad_diff
            mse_diffs += mse_diff
            mad_diffs += mad_diff
            mse_trimap_diffs += mse_trimap_diff
            sad_trimap_diffs += sad_trimap_diff
            mad_trimap_diffs += mad_trimap_diff
            sad_fg_diffs += sad_fg_diff
            sad_bg_diffs += sad_bg_diff
            conn_diffs += conn_diff
            grad_diffs += grad_diff
            Image.fromarray(np.uint8(predict*255)).save(f'test/{img_name}.png')

    log('===============================')
    log(f'Testing numbers: {total_number}')
    log('SAD: {}'.format(sad_diffs / total_number))
    log('MSE: {}'.format(mse_diffs / total_number))
    log('MAD: {}'.format(mad_diffs / total_number))
    log('GRAD: {}'.format(grad_diffs / total_number))
    log('CONN: {}'.format(conn_diffs / total_number))
    log('SAD TRIMAP: {}'.format(sad_trimap_diffs / total_number))
    log('MSE TRIMAP: {}'.format(mse_trimap_diffs / total_number))
    log('MAD TRIMAP: {}'.format(mad_trimap_diffs / total_number))
    log('SAD FG: {}'.format(sad_fg_diffs / total_number))
    log('SAD BG: {}'.format(sad_bg_diffs / total_number))

    return sad_diffs/total_number, mse_diffs/total_number, grad_diffs/total_number


def test_p3m10k(model, dataset_choice, max_image=-1):
    ############################
    # Some initial setting for paths
    ############################
    if dataset_choice == 'P3M_500_P':
        val_option = 'VAL500P'
    else:
        val_option = 'VAL500NP'
    ORIGINAL_PATH = config['datasets']['p3m10k']['path']+'/validation/' + \
        config['datasets']['p3m10k_test'][val_option]['ORIGINAL_PATH']
    MASK_PATH = config['datasets']['p3m10k']['path']+'/validation/' + \
        config['datasets']['p3m10k_test'][val_option]['MASK_PATH']
    TRIMAP_PATH = config['datasets']['p3m10k']['path']+'/validation/' + \
        config['datasets']['p3m10k_test'][val_option]['TRIMAP_PATH']
    ############################
    # Start testing
    ############################
    sad_diffs = 0.
    mse_diffs = 0.
    mad_diffs = 0.
    sad_trimap_diffs = 0.
    mse_trimap_diffs = 0.
    mad_trimap_diffs = 0.
    sad_fg_diffs = 0.
    sad_bg_diffs = 0.
    conn_diffs = 0.
    grad_diffs = 0.
    model.eval()
    img_paths = glob.glob(ORIGINAL_PATH+'/*.jpg')
    if (max_image > 1):
        img_paths = img_paths[:max_image]
    total_number = len(img_paths)
    log('===============================')
    log(
        f'====> Start Testing\n\t----Test: {dataset_choice}\n\t--Number: {total_number}')

    for img_path in tqdm.tqdm(img_paths):
        img_name = (img_path.split('/')[-1])[:-4]
        alpha_path = MASK_PATH+img_name+'.png'
        trimap_path = TRIMAP_PATH+img_name+'.png'
        pil_img = Image.open(img_path)
        img = np.array(pil_img)

        trimap = np.array(Image.open(trimap_path))
        alpha = np.array(Image.open(alpha_path))/255.
        img = img[:, :, :3] if img.ndim > 2 else img
        trimap = trimap[:, :, 0] if trimap.ndim > 2 else trimap
        alpha = alpha[:, :, 0] if alpha.ndim > 2 else alpha
        with torch.no_grad():
            torch.cuda.empty_cache()
            predict = inference_img(model, img)  # HYBRID show less accuracy
            sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(
                predict, alpha, trimap)
            sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(
                predict, alpha)

            sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(
                predict, alpha, trimap)
            conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
            grad_diff = compute_gradient_whole_image(predict, alpha)
            log(f'[{img_paths.index(img_path)}/{total_number}]\nImage:{img_name}\nsad:{sad_diff}\nmse:{mse_diff}\nmad:{mad_diff}\nconn:{conn_diff}\ngrad:{grad_diff}\n-----------')
            sad_diffs += sad_diff
            mse_diffs += mse_diff
            mad_diffs += mad_diff
            mse_trimap_diffs += mse_trimap_diff
            sad_trimap_diffs += sad_trimap_diff
            mad_trimap_diffs += mad_trimap_diff
            sad_fg_diffs += sad_fg_diff
            sad_bg_diffs += sad_bg_diff
            conn_diffs += conn_diff
            grad_diffs += grad_diff

            Image.fromarray(np.uint8(predict*255)).save(f'test/{img_name}.png')

    log('===============================')
    log(f'Testing numbers: {total_number}')
    log('SAD: {}'.format(sad_diffs / total_number))
    log('MSE: {}'.format(mse_diffs / total_number))
    log('MAD: {}'.format(mad_diffs / total_number))
    log('SAD TRIMAP: {}'.format(sad_trimap_diffs / total_number))
    log('MSE TRIMAP: {}'.format(mse_trimap_diffs / total_number))
    log('MAD TRIMAP: {}'.format(mad_trimap_diffs / total_number))
    log('SAD FG: {}'.format(sad_fg_diffs / total_number))
    log('SAD BG: {}'.format(sad_bg_diffs / total_number))
    log('CONN: {}'.format(conn_diffs / total_number))
    log('GRAD: {}'.format(grad_diffs / total_number))

    return sad_diffs/total_number, mse_diffs/total_number, grad_diffs/total_number


def log(str):
    print(str)
    logging.info(str)


if __name__ == '__main__':
    print('*********************************')
    print(config)
    model = StyleMatte()
    model = model.to(device)
    checkpoint = f'{config.checkpoint_dir}/{config.checkpoint}'
    state_dict = torch.load(checkpoint, map_location=f'{device}')
    print('loaded', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    logging.basicConfig(filename=f"report/{config.checkpoint.replace('/','--')}.report",
                        encoding='utf-8', filemode='w', level=logging.INFO)

    if config.dataset_to_use == 'AM2K':
        test_am2k(model)
    else:
        for dataset_choice in ['P3M_500_P', 'P3M_500_NP']:
            test_p3m10k(model, dataset_choice)
