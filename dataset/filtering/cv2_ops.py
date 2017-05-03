import cv2
import numpy as np
from scipy import stats


def brightness_contrast(img, alpha=1.0, beta=0):
    img_contrast = img * alpha
    img_bright = img_contrast + beta
    # img_bright = img_bright.astype(int)
    img_bright = stats.threshold(img_bright, threshmax=255, newval=255)
    img_bright = stats.threshold(img_bright, threshmin=0, newval=0)
    img_bright = img_bright.astype(np.uint8)
    return img_bright


def channel_enhance(img, channel, level=1):
    if channel == 'B':
        blue_ch = img[:, :, 0]
        # blue_ch = (blue_ch - 128) * (level) +128
        blue_ch = blue_ch * level
        blue_ch = stats.threshold(blue_ch, threshmax=255, newval=255)
        img[:, :, 0] = blue_ch
    elif channel == 'G':
        green_ch = img[:, :, 1]
        # green_ch = (green_ch - 128) * (level) +128
        green_ch *= level
        green_ch = stats.threshold(green_ch, threshmax=255, newval=255)
        img[:, :, 0] = green_ch
    elif channel == 'R':
        red_ch = img[:, :, 2]
        # red_ch = (red_ch - 128) * (level) +128
        red_ch *= level
        red_ch = stats.threshold(red_ch, threshmax=255, newval=255)
        img[:, :, 0] = red_ch
    img = img.astype(np.uint8)
    return img


def hue_saturation(img_rgb, alpha=1, beta=1):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    hue = img_hsv[:, :, 0]
    saturation = img_hsv[:, :, 1]
    hue = stats.threshold(hue * alpha, threshmax=179, newval=179)
    saturation = stats.threshold(saturation * beta, threshmax=255, newval=255)
    img_hsv[:, :, 0] = hue
    img_hsv[:, :, 1] = saturation
    img_transformed = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_transformed


# Use interpolate instead of look uptable which was so 10 years ago
def histogram_matching(img, matching_img, number_of_bins=255):
    img_res = img.copy()
    for d in range(img.shape[2]):
        img_hist, bins = np.histogram(img[:, :, d].flatten(), number_of_bins,
                                      normed=True)
        matching_img_hist, bins = np.histogram(matching_img[:, :, d].flatten(),
                                               number_of_bins, normed=True)
        # print bins[:-1]

        cdf_img = img_hist.cumsum()
        cdf_img = (255 * cdf_img / cdf_img[-1]).astype(np.uint8)  # normalize

        cdf_match = matching_img_hist.cumsum()
        cdf_match = (255 * cdf_match / cdf_match[-1]).astype(
            np.uint8)  # normalize

        im2 = np.interp(img[:, :, d].flatten(), bins[:-1], cdf_img)
        im3 = np.interp(im2, cdf_match, bins[:-1])

        img_res[:, :, d] = im3.reshape((img.shape[0]), img.shape[1])

    return img_res
