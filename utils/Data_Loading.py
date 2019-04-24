import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pickle as pkl
import random

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('JPEGImages', 'labels').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        pad_img = np.pad(img, pad, 'constant', constant_values=128)

        padded_h, padded_w, _ = pad_img.shape

        # Resize
        pad_img = cv2.resize(pad_img, self.img_shape)
        # Channels-first
        input_img = pad_img[:, :, ::-1].transpose((2, 0, 1)).copy()
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float().div(255.0)

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]

        filled_labels = torch.from_numpy(filled_labels)

        sample = {'input_img': input_img, 'orig_img': pad_img, 'label': filled_labels, 'path': img_path}

        return sample

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def write(x, img):

    if x.sum() != 0:
        cls = int(x[0])
        label = "{0}".format(classes[cls])
        c1 = (int(x[1]), int(x[2]))
        c2 = (int(x[3]), int(x[4]))
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return img

def convert_label(image_anno, img_width, img_height):

    """
    Function: convert image annotation : center x, center y, w, h (normalized) to x1, y1, x2, y2 for corresponding img
    """
    x_center = image_anno[:, 1]
    y_center = image_anno[:, 2]
    width = image_anno[:, 3]
    height = image_anno[:, 4]

    output = torch.zeros_like(image_anno)
    output[:,0] = image_anno[:,0]
    output[:, 1], output[:, 3] = x_center - width / 2, x_center + width / 2
    output[:, 2], output[:, 4] = y_center - height / 2, y_center + height / 2

    output[:, [1, 3]] *= img_width
    output[:, [2, 4]] *= img_height

    return output.type(torch.FloatTensor)

if __name__ == '__main__':

    classes = load_classes("../data/voc.names")
    colors = pkl.load(open("../data/pallete", "rb"))
    inp_dim = 416
    train_path = '/home/xingyu/Desktop/Project/Pascal_VOC/train.txt'
    test_path = '/home/xingyu/Desktop/Project/Pascal_VOC/2007_test.txt'
    batch_size = 1
    dataloaders = {'train': torch.utils.data.DataLoader(ListDataset(train_path, img_size=inp_dim), batch_size=batch_size, shuffle=True),
                   'test': torch.utils.data.DataLoader(ListDataset(test_path, img_size=inp_dim), batch_size=batch_size, shuffle=True)}

    for i_batch, sample_batched in enumerate(dataloaders["test"]):

        input_images_batch, Orig_images_batch, label_batch, path_batch = sample_batched['input_img'], sample_batched['orig_img'], sample_batched['label'], sample_batched['path']

        print(i_batch, input_images_batch.shape, Orig_images_batch.shape, label_batch.shape, path_batch)

        if i_batch == 0:
            break

    orig_im = Orig_images_batch[0].numpy()
    image_anno = label_batch[0]
    output = convert_label(image_anno, img_width=inp_dim, img_height=inp_dim)
    list(map(lambda x: write(x, orig_im), output))
    cv2.imshow('image',orig_im)
    cv2.waitKey(0)












