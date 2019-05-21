import numpy as np
import gzip
import os

import png


DIRS = {
    't10k': 'test',
    'train': 'train'
}

SAMPLES_PER_CLASS = 50


def transform(prefix):


    subdir = 'data/{}'.format(DIRS[prefix])

    labels_path = 'data/{}-labels-idx1-ubyte'.format(prefix)
    images_path = 'data/{}-images-idx3-ubyte'.format(prefix)

    with open(labels_path, 'rb') as labels_f:
        labels = np.frombuffer(labels_f.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as images_f:
        images = np.frombuffer(images_f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    label_counts = [0] * np.unique(labels).size

    for label in np.unique(labels):
        os.makedirs('{}/{}'.format(subdir, label), exist_ok=True)

    writer = png.Writer(width=28, height=28, greyscale=True)
    for i, label in enumerate(labels):
        if label_counts[label] >= SAMPLES_PER_CLASS:
            continue
        img_filename = '{}.png'.format(label_counts[label])
        with open('{}/{}/{}'.format(subdir, label, img_filename), 'wb') as img_f:
            writer.write(img_f, images[i])
        label_counts[label] += 1


def main():
    transform('train')
    transform('t10k')


if __name__ == '__main__':
    main()
