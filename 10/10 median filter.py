from common import *


def gaussian_filter(_img, k):
    out = np.zeros_like(_img)#.astype(np.float)
    img = _img.copy()#.astype(np.float)
    H, W, C = _img.shape

    # 填充图像
    p = k // 2
    padded = np.zeros([H + 2 * p, W + 2 * p, C])#.astype(np.float)
    padded[p:p + H, p:p + W, :] = img

    # 卷积
    for i in range(H): # 直接对于输出图片的位置进行遍历
        for j in range(W):
            out[i, j, :] = np.median(padded[i:i + k, j:j + k, :], axis=(0,1))  # 不熟悉广播机制就别瞎用... 中值说的是中位数
    out = out.clip(0, 255)
    return out.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread(os.path.join(dir, "imori_noise.jpg"))
    cvshow(img)
    out = gaussian_filter(img, k=3)
    cvshow(out)
