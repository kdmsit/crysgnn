import os
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_metric_learning.losses import NTXentLoss

from data import *
from crysgnn import *


parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--data-path', type=str, default='../data_small/',help='Root data path')
parser.add_argument('--task', choices=['regression', 'classification'],default='regression', help='complete a regression or ''classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,metavar='LR', help='initial learning rate (default: ''0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,metavar='N', help='milestones for scheduler (default: ''[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--test-ratio', default=0.8, type=float, metavar='N',help='percentage of test data to be loaded (default 0.1)')
parser.add_argument('--optim', default='Adam', type=str, metavar='Adam',help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=5, type=int, metavar='N',help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',help='number of hidden layers after pooling')
args=parser.parse_args()

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.



def main():
    import pytz
    global args, best_mae_error
    data_path = args.data_path

    # load data
    dataset = StructureData(data_path)
    # datasize = 755318
    datasize = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])-1
    print('Total Datasize :',datasize)

    idx_train=list(range(datasize))

    collate_fn = collate_pool
    train_loader = get_train_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        pin_memory=args.cuda,
        train_size=idx_train)

    # build model
    structures,_, = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = CrysGNN(orig_atom_fea_len, nbr_fea_len,atom_fea_len=args.atom_fea_len,n_conv=args.n_conv)


    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    criterion = NTXentLoss(temperature=0.07)
    print('Optimizer : '+str(args.optim))
    print()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')
    best_loss=999
    best_model = model
    args.start_epoch=0
    loss_train=[]
    adj_loss_train = []
    feat_loss_train = []
    ngh_loss_train = []
    contrastive_loss_train = []
    sg_loss_train = []

    for epoch in range(args.start_epoch, args.epochs):
        t_epoch = time.time()
        loss, adj_loss, feat_loss, contrastive_losses,sg_loss = train(train_loader, model, criterion, optimizer, epoch)

        if epoch >1:
            loss_train.append(loss)
            adj_loss_train.append(adj_loss)
            feat_loss_train.append(feat_loss)
            contrastive_loss_train.append(contrastive_losses)
            sg_loss_train.append(sg_loss)
        print()
        print(' Epoch Summary : Epoch ' + str(epoch),
              ' Loss: {:.2f}'.format(loss),
              ' Adj Reconst Loss: {:.2f}'.format(adj_loss),
              ' Feat Reconst Loss: {:.2f}'.format(feat_loss),
              ' Contrastive Loss: {:.2f}'.format(contrastive_losses),
              ' Space Group Loss: {:.2f}'.format(sg_loss),
              ' time: {:.4f} min'.format((time.time() - t_epoch)/60))

        if loss<best_loss:
            best_loss=loss
            best_model=model
            print(" Best Loss :"+str(best_loss)+", Saving the model !!")
            torch.save({'state_dict': best_model.state_dict()}, '../model/crysgnn_state_checkpoint_'+str(epoch)+'.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch):
    losses =[]
    adj_losses = []
    feat_losses = []
    sg_loss=[]
    contrastive_losses = []
    # switch to train mode
    model.train()
    for i, (input, _) in enumerate(train_loader):
        t = time.time()
        atom_fea=input[0]
        nbr_fea = input[1]
        nbr_fea_idx = input[2]
        adj = torch.LongTensor(input[3])
        sg = torch.LongTensor(np.asarray(input[4]))
        sg_no = torch.LongTensor(np.asarray(input[5]))
        crys_index=input[6]

        if args.cuda:
            input_var = (Variable(atom_fea.cuda(non_blocking=True)),
                         Variable(nbr_fea.cuda(non_blocking=True)),
                         nbr_fea_idx.cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in crys_index])
            atom_fea=atom_fea.cuda()
            adj=adj.cuda()
            sg=sg.cuda()
            sg_no=sg_no.cuda()
        else:
            input_var = (Variable(atom_fea),
                         Variable(nbr_fea),
                         nbr_fea_idx,
                         crys_index)

        # compute output
        edge_prob_list, atom_feature_list, sg_pred_list, crys_fea_list,_ = model(*input_var,args.cuda)


        # region Node Level Loss
        # Connection reconstruction
        pos_weight=torch.Tensor([0.1,1,1,1,1,1])
        if args.cuda:
            pos_weight=pos_weight.cuda()
        loss_adj_reconst = F.nll_loss(edge_prob_list, adj,weight=pos_weight)

        # Feature reconstruction
        loss_atom_feat_reconst = F.binary_cross_entropy_with_logits(atom_feature_list, atom_fea)
        # endregion

        # region Graph Level Loss
        # Contrastive Loss
        info_nce_loss = criterion(crys_fea_list, sg)

        # Space Group Reconstruction
        loss_sg_reconst = F.nll_loss(sg_pred_list, sg_no)
        # endregion

        loss = 0.25 * loss_adj_reconst + 0.25 * loss_atom_feat_reconst + 0.25 * info_nce_loss + 0.25 * loss_sg_reconst


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        adj_losses.append(loss_adj_reconst.item())
        feat_losses.append(loss_atom_feat_reconst.item())
        contrastive_losses.append(info_nce_loss.item())
        sg_loss.append(loss_sg_reconst.item())
        if (i+1)%1000==0:
            print(' Epoch ' + str(epoch),
                  ' Batch ' + str(i),
                  ' Loss: {:.2f}'.format(loss),
                  ' Adj Reconst Loss: {:.2f}'.format(loss_adj_reconst),
                  ' Feat Reconst Loss: {:.2f}'.format(loss_atom_feat_reconst),
                  ' Contrastive Loss: {:.2f}'.format(info_nce_loss),
                  ' Space Group Loss: {:.2f}'.format(loss_sg_reconst),
                  ' time: {:.4f} min'.format((time.time() - t) / 60))
    return np.mean(losses),np.mean(adj_losses),np.mean(feat_losses),np.mean(contrastive_losses),np.mean(sg_loss)



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


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
