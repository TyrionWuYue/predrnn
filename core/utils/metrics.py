__author__ = 'yunbo'

import numpy as np
import os
import matplotlib.pyplot as plt

def batch_psnr(gen_frames, gt_frames):
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)


def visualize(inputs, targets, outputs, idx, cache_dir):
    _, axarray = plt.subplots(3, targets.shape[1], figsize=(targets.shape[1] * 5, 10))

    for t in range(targets.shape[1]):
        axarray[0][t].imshow(inputs[0, t, 0].detach().cpu().numpy(), cmap='gray')
        axarray[1][t].imshow(targets[0, t, 0].detach().cpu().numpy(), cmap='gray')
        axarray[2][t].imshow(outputs[0, t, 0].detach().cpu().numpy(), cmap='gray')

    plt.savefig(os.path.join(cache_dir, '{:03d}.png'.format(idx)))
    plt.close()