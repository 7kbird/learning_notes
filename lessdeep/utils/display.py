import numpy as np
import matplotlib.pyplot as plt


def imshow(images, figsize=(12,6), rows=1, interpolation=False, titles=None):
    plt.clf()
    f = plt.figure(figsize=figsize)
    if len(images) % 2 == 0:
        cols = len(images) // rows
    else:
        cols = len(images) // rows + 1

    for i, img in enumerate(images):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        if len(img.shape) == 3 and img.shape[-1] not in [1, 3]:
            # some image is in shape of (3, w, h)
            img = img.transpose((1, 2, 0))
        plt.imshow(img.astype(np.uint8), interpolation=None if interpolation else 'none')
    plt.show()