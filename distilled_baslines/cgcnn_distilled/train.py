import argparse
import os
import shutil
import sys
import time
import pytz
from tqdm import tqdm
import warnings
from random import sample
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from data import PKLData,CIFData
from data import collate_pool, get_train_val_test_loader
from model import CrystalGraphConvNet
from crysgnn.model import *


parser = argparse.ArgumentParser(description='Crystal Graph Neural Networks')
parser.add_argument('--data-path', type=str, default="../../data_small/",help='Root data path')
parser.add_argument('--task', choices=['regression', 'classification'],default='regression', help='complete a regression or ''classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate (default: ''0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,metavar='N', help='milestones for scheduler (default: ''[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

parser.add_argument('--train-ratio', default=0.8, type=float, metavar='N',help='percentage of train data ')
parser.add_argument('--val-ratio', default=0.1, type=float, metavar='N',help='percentage of validation data ')
parser.add_argument('--test-ratio', default=0.1, type=float, metavar='N',help='percentage of test data ')
parser.add_argument('--train-size', default=None, type=int, metavar='N',help='size of train data ')
parser.add_argument('--val-size', default=None, type=int, metavar='N',help='size of validation data ')
parser.add_argument('--test-size', default=None, type=int, metavar='N',help='size of test data ')

parser.add_argument('--optim', default='Adam', type=str, metavar='Adam',help='choose an optimizer, SGD or Adam, (default: Adam)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',help='number of hidden layers after pooling')
parser.add_argument('--alpha', default=0.5, type=float, help='weightage of property loss')
parser.add_argument('--is-log', default=1, type=int, help='indicator of logged file')
parser.add_argument('--patience', default=30, type=int, help='patience parameter for early stopping')
parser.add_argument('--split', default='split_0', type=str, help='patience parameter for early stopping')
parser.add_argument('--pretrain_model', default='', type=str, help='Pretrained Model Name')
args=parser.parse_args()

args.cuda = torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

def plot(epoch_list, loss_prop_list,path):
    frontsize=20
    line_width=5
    fig_path=path
    df=pd.DataFrame({'x': epoch_list,'y1': loss_prop_list})
    plt.plot( 'x', 'y1', data=df, color='green', linewidth=line_width,label="Train Loss")
    plt.xlabel("Number of epochs ", fontsize=frontsize,labelpad=10,fontweight='bold')
    plt.ylabel("Losses", fontsize=frontsize,fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def main():
    global args, best_mae_error
    eastern = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now().astimezone(eastern).time()
    current_date = datetime.datetime.now().astimezone(eastern).date()
    data_path = args.data_path
    print(data_path)

    # load data
    dataset = PKLData(data_path)
    path, dirs, files = next(os.walk(data_path))
    datasize = len(files)

    print('Cuda :',args.cuda)

    print("Data size", datasize)
    print("train size :" + str(args.train_size))
    print("val size :" + str(args.val_size))
    print("test size :" + str(args.test_size))
    print("train ratio :" + str(args.train_ratio))
    print("val ratio :" + str(args.val_ratio))
    print("test ratio :" + str(args.test_ratio))

    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset=dataset,total_size=datasize,collate_fn=collate_fn,batch_size=args.batch_size,
                                             train_ratio=args.train_ratio,num_workers=args.workers,val_ratio=args.val_ratio,
                                             test_ratio=args.test_ratio,pin_memory=args.cuda,train_size=args.train_size,
                                             val_size=args.val_size,test_size=args.test_size,return_test=True)

    # obtain target value normalizer
    if datasize < 2000:
        sample_data_list = [dataset[i] for i in range(datasize)]
    else:
        sample_data_list = [dataset[i] for i in sample(range(datasize), 2000)]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)


    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,atom_fea_len=args.atom_fea_len,n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,n_h=args.n_h,classification=True if args.task =='classification' else False)

    checkpoint_file_path = "../../model/pretrain_model.pth"
    print(checkpoint_file_path)
    if args.cuda:
        t_model = torch.load(checkpoint_file_path)
    else:
        t_model = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))

    # Pre-trained CrysGNN model define and load from saved checkpoints

    # checkpoint_file_path = "../../model/pretrain_model.pth"
    # t_model = CrysGNN(orig_atom_fea_len, nbr_fea_len, atom_fea_len=args.atom_fea_len, n_conv=5)
    # checkpoint = torch.load(checkpoint_file_path)
    # t_model.load_state_dict(checkpoint['state_dict'])


    pytorch_total_params = sum(p.numel() for p in t_model.parameters())
    print("Teacher Model parameters :", pytorch_total_params)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Student Model parameters :", pytorch_total_params)

    if args.cuda:
        model.cuda()

    if args.is_log:
        path = 'results/' + str(current_date) + '/' + str(current_time)
        if not os.path.exists(path):
            os.makedirs(path)
        print("Out path :", path)
        out = open(path + "/out.txt", "w")
        print("***** Hyper-Parameters Details ********")
        out.writelines("***** Hyper-Parameters Details ********")
        out.writelines("\n")
        print("data_path :" + str(data_path))
        out.writelines("data_path :" + str(data_path))
        out.writelines("\n")
        print("Data size :" + str(datasize))
        out.writelines("Data size :" + str(datasize))
        out.writelines("\n")

        print("train size :" + str(args.train_size))
        out.writelines("train size :" + str(args.train_size))
        out.writelines("\n")
        print("val size :" + str(args.val_size))
        out.writelines("val size :" + str(args.val_size))
        out.writelines("\n")
        print("test size :" + str(args.test_size))
        out.writelines("test size  :" + str(args.test_size))
        out.writelines("\n")
        print("train ratio :" + str(args.train_ratio))
        out.writelines("train ratio :" + str(args.train_ratio))
        out.writelines("\n")
        print("val ratio :" + str(args.val_ratio))
        out.writelines("val ratio :" + str(args.val_ratio))
        out.writelines("\n")
        print("test ratio :" + str(args.test_ratio))
        out.writelines("test ratio  :" + str(args.test_ratio))
        out.writelines("\n")

        print("L_Rate :" + str(args.lr))
        out.writelines("L_Rate :" + str(args.lr))
        out.writelines("\n")
        print("Optimizer :" + str(args.optim))
        out.writelines("Optimizer :" + str(args.optim))
        out.writelines("\n")
        print("atom_fea_len :" + str(args.atom_fea_len))
        out.writelines("atom_fea_len :" + str(args.atom_fea_len))
        out.writelines("\n")
        print("h-fea-len :" + str(args.h_fea_len))
        out.writelines("h-fea-len :" + str(args.h_fea_len))
        out.writelines("\n")
        print("Epochs :" + str(args.epochs))
        out.writelines("Epochs :" + str(args.epochs))
        out.writelines("\n")
        print("n_conv :" + str(args.n_conv))
        out.writelines("n_conv :" + str(args.n_conv))
        out.writelines("\n")
        print("alpha :" + str(args.alpha))
        out.writelines("alpha :" + str(args.alpha))
        out.writelines("\n")
        print("Batch Size :" + str(args.batch_size))
        out.writelines("Batch Size:" + str(args.batch_size))
        out.writelines("\n")
        out.writelines("Teacher Model Name:" + str(checkpoint_file_path))
        out.writelines("\n")
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Model parameters :", pytorch_total_params)
        out.writelines("Model parameters :" + str(pytorch_total_params))
        out.writelines("\n")
        out.writelines("\n")

    # define loss func and optimizer
    criterion = nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')


    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,gamma=0.1)
    best_model=model
    start_time=time.time()
    train_loss_list=[]
    train_emb_loss_list = []
    train_prop_loss_list = []
    for epoch in range(args.start_epoch, args.epochs):
        t = time.time()
        # train for one epoch
        train_loss, train_loss_p, train_loss_emb, train_mae_error = train(train_loader, model, t_model, criterion, optimizer, epoch, normalizer,args.alpha)

        # evaluate on validation and test set
        val_mae_error = validate(val_loader, model, criterion, normalizer)
        test_mae_error = validate(test_loader, model, criterion, normalizer)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if test_mae_error < best_mae_error:
            best_mae_error = min(test_mae_error, best_mae_error)
            best_epoch = epoch
            best_model = model

        print('Epoch: {:d}'.format((epoch + 1)),
              ' Train Loss: {:.3f}'.format(train_loss),
              ' Train property loss : {:.3f}'.format(train_loss_p),
              ' Train embedding loss : {:.3f}'.format(train_loss_emb),
              ' Train MAE: {:.3f}'.format(train_mae_error),
              ' Val MAE: {:.3f}'.format(val_mae_error),
              ' time: {:.4f} min'.format((time.time() - t) / 60))

        if args.is_log:
            out.writelines('Epoch: {:d}'.format((epoch + 1)) +
                           '\t Train Loss: {:.3f}'.format(train_loss) +
                           '\t Train property loss : {:.3f}'.format(train_loss_p) +
                           '\t Train embedding loss : {:.3f}'.format(train_loss_emb) +
                           '\t Train MAE: {:.3f}'.format(train_mae_error) +
                           '\t Val MAE: {:.3f}'.format(val_mae_error) +
                           '\t time: {:.4f} min'.format((time.time() - t) / 60))
            out.writelines("\n")
        train_loss_list.append(train_loss)
        train_emb_loss_list.append(train_loss_emb)
        train_prop_loss_list.append(train_loss_p)

    print("Test error :" + str(best_mae_error))
    out.writelines("Test error :" + str(best_mae_error))

def train(train_loader, model,t_model, criterion, optimizer, epoch, normalizer,alpha):
    losses = AverageMeter()
    losses_p = AverageMeter()
    losses_emb = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()
    loss=0

    for i, (input, target, _) in enumerate(train_loader):

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        target_normed = normalizer.norm(target)

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output, atom_fea = model(*input_var)
        _, _, _, _, bt_atom_fea = t_model(*input_var)

        loss_p = criterion(output, target_var)
        loss_emb=0
        for i in range(len(bt_atom_fea)):
            loss_emb += criterion(atom_fea[i], bt_atom_fea[i])
        loss_emb=loss_emb/len(bt_atom_fea)

        loss = (1 - alpha) * loss_p + alpha * loss_emb

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        losses_p.update(loss_p.data.cpu(), target.size(0))
        losses_emb.update(loss_emb.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg,losses_p.avg,losses_emb.avg,mae_errors.avg


def validate(val_loader, model, criterion, normalizer):
    losses = AverageMeter()
    mae_errors = AverageMeter()


    # switch to evaluate mode
    model.eval()

    for i, (input, target, batch_cif_ids) in enumerate(val_loader):

        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

        target_normed = normalizer.norm(target)

        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output,_ = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

    return mae_errors.avg



class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
