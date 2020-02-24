from common import *


def gaussian_filter(_img, k):
    out = np.zeros_like(_img)#.astype(np.float)
    img = _img.copy()#.astype(np.float)
    H, W, C = _img.shape

    # 填充图像
    p = k // 2
    padded = np.zeros([H + 2 * p, W + 2 * p, C])#.astype(np.float)
    padded[p:p + H, p:p + W, :] = img

    # 生成卷积核
    kernel = []
    sig = 1.3
    for x in range(-(k // 2), k // 2 + 1): # 对概率分布的位置进行遍历
        for y in range(-(k // 2), k // 2 + 1):
            kernel.append(1 / (2 * np.pi * sig ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2)))
    kernel = np.array(kernel).reshape(k, k)
    kernel = kernel / kernel.sum()
    kernel = np.tile(np.expand_dims(kernel, 2), (1, 1, 3)) # 没有复制通道就会出现错误

    # 卷积
    for i in range(H): # 直接对于输出图片的位置进行遍历
        for j in range(W):
            out[i, j, :] = (kernel * padded[i:i + k, j:j + k, :]).sum(axis=(0, 1)) # 不熟悉广播机制就别瞎用...
    out = out.clip(0, 255)
    return out.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread(os.path.join(dir, "imori_noise.jpg"))
    cvshow(img)
    out = gaussian_filter(img, k=3)
    cvshow(out)
