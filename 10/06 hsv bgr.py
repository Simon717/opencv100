from common import *


def BGR2HSV(_img):
    img = _img.copy() / 255. # 代码就算是翻译 也切记不可以无脑翻译...必然会出现问题
    # 本次出现bug就是因为没有在一开始就把图像数据变成浮点数据
    b, g, r =  img.transpose(2, 0, 1) # 千万注意顺序...
    _max = img.max(2)
    _min = img.min(2)
    min_idx = img.argmin(2)
    # print(min_idx)

    hsv = np.zeros_like(img).astype(np.float)
    hsv[..., 0][_max == _min] = 0
    idx = np.where(min_idx == 0)
    hsv[..., 0][idx] = (60 * (g - r) / (_max - _min) + 60)[idx]
    idx = np.where(min_idx == 2)
    hsv[..., 0][idx] = (60 * (b - g) / (_max - _min) + 180)[idx]
    idx = np.where(min_idx == 1)
    hsv[..., 0][idx] = (60 * (r - b) / (_max - _min) + 300)[idx]

    hsv[..., 1] = _max - _min
    hsv[..., 2] = _max
    return hsv

def HSV2BGR(img, hsv):
    _max, _min = img.max(2), img.min(2)
    h, s, v = hsv.transpose(2, 0, 1)
    bgr = np.zeros_like(img).astype(np.float)
    c  = s
    h_ = h / 60
    x = c * (1 - np.abs(h_ % 2 - 1))
    out = np.expand_dims((v - c), 2).astype(np.float)
    z = np.zeros_like(h)
    add = [[c, x, z], [x, c, z],
           [z, c, x], [z, x, c],
           [x, z, c], [c, z, x]]
    for i in range(6):
        # 这段代码只是为了练习numpy的花式索引（一次性访问多个分散的数据点） 实际上可读性不是很好
        # 一次性准备好所有数据的x,y,z坐标数据
        inx, iny = np.where((h_ >= i) & (h_ < i + 1))
        inx = np.tile(np.expand_dims(inx, 1), (1, 3)).flatten()
        iny = np.tile(np.expand_dims(iny, 1), (1, 3)).flatten()
        inz = list(range(3)) * (len(inx) // 3)
        bgr[inx, iny, inz] = np.array(add[i]).transpose(1, 2, 0)[inx, iny, inz]
    bgr += out
    bgr[_max == _min] = 0
    bgr = bgr.clip(0, 1)
    bgr = (bgr * 255).astype(np.uint8)
    return bgr

if __name__ == '__main__':
    img = cv2.imread(os.path.join(dir, "imori.jpg"))

    hsv = BGR2HSV(img)
    hsv[..., 0] = (hsv[..., 0] + 180) % 360
    bgr = HSV2BGR(img, hsv)
    cvshow(bgr)