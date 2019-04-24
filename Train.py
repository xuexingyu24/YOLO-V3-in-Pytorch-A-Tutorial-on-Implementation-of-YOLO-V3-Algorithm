import os
import torch
from Darknet_VOC import Darknet
from utils.Data_Loading import ListDataset
import time
import copy
import numpy as np
import math
import torch.nn as nn
from utils.util import *
import argparse
import warnings
warnings.filterwarnings("ignore")

def build_targets(target, anchors, grid_size, num_anchors = 3, num_classes = 20):
    """
    Function: build the corresponding feature map containing ground truth label for loss calcualtion
    Accept: target: ground truth label in format of torch [batch_size, number of object (50), attribute (5)]
    Return: mask, tx, ty, tw, th, tconf, tcls: torch.size [batch x num_anchors x grid x grid x (num_class)].
    """
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.zeros(nB, nA, nG, nG)
    tcls = torch.zeros(nB, nA, nG, nG, nC)

    for b in range(nB):  # for each image
        for t in range(target.shape[1]):  # for each object
            if target[b, t].sum() == 0:  # if the row is empty
                continue
            # Convert to object label data to feature map
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)  # 1 x 4
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Masks
            mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

    return mask, tx, ty, tw, th, tconf, tcls

def Loss(input, target, anchors, inp_dim, num_anchors = 3, num_classes = 20):

    """
    Accept: network output && target (ground truth label)
    Return: loss
    """
    nA = num_anchors  # number of anchors
    nB = input.size(0)  # number of batches
    nG = input.size(2)  # number of grid size
    nC = num_classes
    stride = inp_dim / nG

    # Tensors for cuda support
    FloatTensor = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
    ByteTensor = torch.cuda.ByteTensor if input.is_cuda else torch.ByteTensor

    prediction = input.view(nB, nA, 5 + nC, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # reshape the output data

    # Get outputs
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred

    # Calculate offsets for each grid
    grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
    grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
    scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    mask, tx, ty, tw, th, tconf, tcls = build_targets(
        target=target.cpu().data,
        anchors=scaled_anchors.cpu().data,
        grid_size=nG,
        num_anchors=nA,
        num_classes=num_classes)

    # Handle target variables
    tx, ty = tx.type(FloatTensor), ty.type(FloatTensor)
    tw, th = tw.type(FloatTensor), th.type(FloatTensor)
    tconf, tcls = tconf.type(FloatTensor), tcls.type(FloatTensor)
    mask = mask.type(ByteTensor)

    mse_loss = nn.MSELoss(reduction='sum')  # Coordinate loss
    bce_loss = nn.BCELoss(reduction='sum')  # Confidence loss
    loss_x = mse_loss(x[mask], tx[mask])
    loss_y = mse_loss(y[mask], ty[mask])
    loss_w = mse_loss(w[mask], tw[mask])
    loss_h = mse_loss(h[mask], th[mask])
    loss_conf = bce_loss(pred_conf, tconf)
    loss_cls = bce_loss(pred_cls[mask], tcls[mask])
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    return (loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls)

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

def eval(output, labels, img_width, img_height):

    """
    Funcation: calculate the precision, recall, and F1 score
    Accept: output from model: torch.size [nProposal x (img_index, x1, y1, x2, y2, obj_score, hig_class_score, class_index]
            labels: from dataloader: torch.size [batch x 50 x 5(label)]
    return: the batch precision, recall, F1 score
    """
    nProposals = int((output[:, 5] > 0.5).sum().item())
    nGT = 0
    nCorrect = 0
    for b in range(labels.shape[0]):  # for each image
        prediction = output[output[:,0] == b]  # filter out the predictions of corresponding image
        for t in range(labels.shape[1]):  # for each object
            if labels[b, t].sum() == 0:  # if the row is empty
                continue
            nGT += 1
            gt_label = convert_label(labels[b, t].unsqueeze(0), img_width, img_height)
            gt_box = gt_label[:, 1:5]
            for i in range(prediction.shape[0]):
                pred_box = prediction[i, 1:5].unsqueeze(0)
                iou = bbox_iou(pred_box, gt_box)
                pred_label = prediction[i, -1]
                target_label = gt_label[0, 0]
                if iou > 0.5 and pred_label == target_label:
                    nCorrect += 1
    recall = float(nCorrect / nGT) if nGT else 1
    precision = float(nCorrect / nProposals) if nProposals else 0
    F1_score = 2 * recall * precision / (recall + precision + 1e-16)

    return F1_score, precision, recall

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
parser.add_argument("--data_folder", type=str, default="/home/xingyu/Desktop/Project/Pascal_VOC", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--weights_path", type=str, default="weights/Dartnet_VOC_weights_ini", help="path to weights file")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
opt = parser.parse_args()

# get data loader
train_path = os.path.join(opt.data_folder, 'train.txt')
val_path = os.path.join(opt.data_folder, '2007_test.txt')
inp_dim = opt.img_size
dataloaders = {'train': torch.utils.data.DataLoader(ListDataset(train_path, img_size=inp_dim), batch_size=opt.batch_size, shuffle=True),
               'val': torch.utils.data.DataLoader(ListDataset(val_path, img_size=inp_dim), batch_size=opt.batch_size, shuffle=True)}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA = torch.cuda.is_available()

# load the model and weights for initialization
model = Darknet()
model.load_state_dict(torch.load(opt.weights_path, map_location=lambda storage, loc: storage))

#  This section is to freeze all the network except the output three layers
for name, param in model.named_parameters():
    param.requires_grad = False
    if int(name.split('.')[1]) in (79, 80, 81, 91, 92, 93, 103, 104, 105):
        param.requires_grad = True

model = model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_F1_score = 0.0

num_epochs = opt.epochs
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train() # set model to training mode
        else:
            model.eval() # set model to evaluate mode

        running_loss, running_xy_loss, running_wh_loss, running_conf_loss, running_cls_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        running_recall, running_precision, running_F1_score = 0.0, 0.0, 0.0

        # iterate over data
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            inputs, labels = sample_batched['input_img'], sample_batched['label']
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):

                Final_pre, *output = model(inputs, CUDA)
                Final_pre = write_results(Final_pre, confidence=0.5, num_classes=20, nms_conf=0.4)

                anchors = (
                [(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)], [(116, 90), (156, 198), (373, 326)])

                loss_item = {"total_loss": 0, "x": 0, "y": 0, "w": 0, "h": 0, "conf": 0, "cls": 0}

                for i in range(len(output)):
                    losses = Loss(output[i], labels.float(), anchors[i], inp_dim=inp_dim, num_anchors = 3, num_classes = 20)
                    for i, name in enumerate(loss_item):
                        loss_item[name] += losses[i]

                if isinstance(Final_pre, int) == False:
                    F1_score, precision, recall = eval(Final_pre.cpu(), labels, img_width=inp_dim, img_height=inp_dim)
                else:
                    F1_score, precision, recall = 0, 0, 0

                loss = loss_item['total_loss']
                xy_loss = loss_item['x']+loss_item['y']
                wh_loss = loss_item['w']+loss_item['h']
                conf_loss = loss_item['conf']
                cls_loss = loss_item['cls']

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_xy_loss += xy_loss.item()
                running_wh_loss += wh_loss.item()
                running_conf_loss += conf_loss.item()
                running_cls_loss += cls_loss.item()
                running_recall += recall
                running_precision += precision
                running_F1_score += F1_score

        epoch_loss = running_loss / ((i_batch+1)*batch_size)
        epoch_xy_loss = running_xy_loss / ((i_batch+1)*batch_size)
        epoch_wh_loss = running_wh_loss / ((i_batch + 1) * batch_size)
        epoch_conf_loss = running_conf_loss / ((i_batch + 1) * batch_size)
        epoch_cls_loss = running_cls_loss / ((i_batch + 1) * batch_size)

        epoch_recall = running_recall / (i_batch+1)
        epoch_precision = running_precision / (i_batch+1)
        epoch_F1_score = running_F1_score / (i_batch+1)

        print(
            '{} Loss: {:.4f} Recall: {:.4f} Precision: {:.4f} F1 Score: {:.4f}'.format(phase, epoch_loss, epoch_recall,
                                                                                       epoch_precision, epoch_F1_score))
        print(
            '{} xy: {:.4f} wh: {:.4f} conf: {:.4f} class: {:.4f}'.format(phase, epoch_xy_loss, epoch_wh_loss,
                                                                                                        epoch_conf_loss,
                                                                                                        epoch_cls_loss))

        # deep copy the model
        if phase == 'val' and epoch_F1_score > best_F1_score:
            best_F1_score = epoch_F1_score
            best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best F1 score: {:4f}'.format(best_F1_score))

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'Dartnet_VOC_Weights')
