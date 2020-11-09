import argparse
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as dataloader
import torchvision.datasets as datasets

from sklearn.metrics import roc_auc_score, roc_curve, auc

from models.allconv import AllConvNet
from models.wrn_prime import WideResNet
from RotDataset import RotDataset
from utils import *
import matplotlib.pyplot as plt


def arg_parser():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--method', type=str, default='rot', help='rot, msp')
    parser.add_argument('--ood_dataset', type=str, default='cifar100', help='cifar100 | svhn')
    parser.add_argument('--num_workers', type=int, default=2)

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--rot-loss-weight', type=float, default=0.5, help='Multiplicative factor on the rot losses')

    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

    args = parser.parse_args()

    return args


def test(args, model, test_loader, num_classes):
    model.eval()

    with torch.no_grad():
        scores_list = None

        for x_tf_0, x_tf_90, x_tf_180, x_tf_270, batch_y in tqdm(test_loader):
            batch_size = x_tf_0.shape[0]

            batch_x = torch.cat([x_tf_0, x_tf_90, x_tf_180, x_tf_270], 0).cuda()
            batch_y = batch_y.cuda()

            batch_rot_y = torch.cat((
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size)
            ), 0).long().cuda()

            logits, pen = model(batch_x)

            classification_logits = logits[:batch_size]
            rot_logits = model.rot_head(pen)

            # use self-supervised rotation loss
            if args.method == 'rot':
                U = torch.ones([batch_size, num_classes]).cuda() / num_classes
                p = F.softmax(classification_logits, dim=1)

                kl_loss = F.kl_div(U.log(), p, reduction='none').sum(1)
                rot_loss = F.cross_entropy(rot_logits, batch_rot_y, reduction='none').reshape(4, batch_size).mean(0)

                score = -kl_loss + rot_loss
                # print(kl_loss.mean().item(), rot_loss.mean().item()/4.)
                # baseline, maximum softmax probability
            elif args.method == 'msp':
                score = -F.softmax(classification_logits, dim=1).max(1)[0]

            if scores_list is None:
                scores_list = score
            else:
                scores_list = torch.cat([scores_list, score], dim=0)

        return scores_list


def ROC(id_scores_list, ood_scores_list, method, dataset):
    id_num, ood_num = id_scores_list.size(0), ood_scores_list.size(0)

    total_list = torch.cat([-id_scores_list, -ood_scores_list], dim=0).detach().cpu().numpy()
    label_list = np.concatenate((np.ones(id_num), np.zeros(ood_num)))

    fpr, tpr, thresholds = roc_curve(label_list, total_list, pos_label=1)

    auroc = auc(fpr, tpr)

    print('AUROC: {:.2f}'.format(auroc * 100))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC {} {}'.format(method, dataset))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('ROC_{}_{}.png'.format(method, dataset))


def calculate_AUROC(id_scores_list, ood_scores_list):
    id_num, ood_num = id_scores_list.size(0), ood_scores_list.size(0)

    total_list = torch.cat([id_scores_list, ood_scores_list], dim=0).detach().cpu().numpy()
    label_list = np.concatenate((np.zeros(id_num), np.ones(ood_num)))
    # the greater score, the greater label
    return roc_auc_score(label_list, total_list)


def main():
    # arg parser
    args = arg_parser()
    print(args.method, args.ood_dataset)
    # set seed
    set_seed(args.seed)

    # dataset
    id_testdata = datasets.CIFAR10('./data/', train=False, download=True)
    id_testdata = RotDataset(id_testdata, train_mode=False)

    if args.ood_dataset == 'cifar100':
        ood_testdata = datasets.CIFAR100('./data/', train=False, download=True)
    elif args.ood_dataset == 'svhn':
        ood_testdata = datasets.SVHN('./data/', split='test', download=True)
    else:
        raise ValueError(args.ood_dataset)
    ood_testdata = RotDataset(ood_testdata, train_mode=False)

    # data loader
    id_test_loader = dataloader(id_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    ood_test_loader = dataloader(ood_testdata, batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_memory=True)

    # load model
    num_classes = 10
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    model.rot_head = nn.Linear(128, 4)
    model = model.cuda()
    # model.load_state_dict(torch.load('./models/backup/trained_model_{}.pth'.format(args.method)))
    model.load_state_dict(torch.load('./my_trained_model_{}.pth'.format(args.method)))
    # TODO
    # 1. calculate ood score by two methods(MSP, Rot)
    # 2. calculate AUROC by using ood scores
    id_scores_list = test(args, model, id_test_loader, 10)
    ood_scores_list = test(args, model, ood_test_loader, 10)

    print('id score: {:.3f} | ood score: {:.3f}'.format(id_scores_list.mean().item(), ood_scores_list.mean().item()))
    # print('AUROC: {:.2f}%'.format(calculate_AUROC(id_scores_list, ood_scores_list) * 100))
    ROC(id_scores_list, ood_scores_list, args.method, args.ood_dataset)


if __name__ == "__main__":
    main()
