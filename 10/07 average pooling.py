from common import *

img = cv2.imread(os.path.join(dir, "imori.jpg"))

def AveragePooling(img, k=8):
    H, W, C = img.shape
    H_, W_ = H // k, W // k
    pool = np.zeros([H_, W_, C])

    for i in range(H // k):
        for j in range(W // k):
            pool[i, j, :] = img[i * k:(i + 1) * k, j * k:(j + 1) * k, :].mean(axis=(0, 1))
    return pool.astype(np.uint8)

res = AveragePooling(img)
cvshow(res)
