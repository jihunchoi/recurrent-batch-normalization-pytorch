"""Train the model using MNIST dataset."""
import argparse
import os
from datetime import datetime
from functools import partial

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pycrayon import CrayonClient

from bnlstm import LSTM, LSTMCell, BNLSTMCell


def transform_flatten(tensor):
    return tensor.view(-1, 1).contiguous()


def transform_permute(tensor, perm):
    return tensor.index_select(0, perm)


def main():
    data_path = args.data
    model_name = args.model
    save_dir = args.save
    hidden_size = args.hidden_size
    pmnist = args.pmnist
    batch_size = args.batch_size
    max_iter = args.max_iter
    use_gpu = args.gpu

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pmnist:
        perm = torch.randperm(784)
    else:
        perm = torch.range(0, 783).long()
    train_dataset = datasets.MNIST(
        root=data_path, train=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transform_flatten,
                                      partial(transform_permute, perm=perm)]),
        download=True)
    valid_dataset = datasets.MNIST(
        root=data_path, train=False,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transform_flatten,
                                      partial(transform_permute, perm=perm)]),
        download=True)

    tb_client = CrayonClient()
    tb_xp_name = '{}-{}'.format(datetime.now().strftime("%y%m%d-%H%M%S"),
                                save_dir)
    tb_xp_train = tb_client.create_experiment('{}/train'.format(tb_xp_name))
    tb_xp_valid = tb_client.create_experiment('{}/valid'.format(tb_xp_name))

    if model_name == 'bnlstm':
        model = LSTM(cell_class=BNLSTMCell, input_size=1,
                     hidden_size=hidden_size, batch_first=True)
    elif model_name == 'lstm':
        model = LSTM(cell_class=LSTMCell, input_size=1,
                     hidden_size=hidden_size, batch_first=True)
    else:
        raise ValueError
    fc = nn.Linear(in_features=hidden_size, out_features=10)
    params = list(model.parameters()) + list(fc.parameters())
    optimizer = optim.RMSprop(params=params, lr=1e-3, momentum=0.9)

    def compute_loss_accuracy(data, label):
        hx = None
        if not pmnist:
            h0 = Variable(data.data.new(data.size(0), hidden_size)
                          .normal_(0, 0.1))
            c0 = Variable(data.data.new(data.size(0), hidden_size)
                          .normal_(0, 0.1))
            hx = (h0, c0)
        _, (h_n, _) = model(input_=data, hx=hx)
        logits = fc(h_n[0])
        loss = functional.cross_entropy(input=logits, target=label)
        accuracy = (logits.max(1)[1] == label).float().mean()
        return loss, accuracy

    if use_gpu:
        model.cuda()
        fc.cuda()

    iter_cnt = 0
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=True, pin_memory=True)
    while iter_cnt < max_iter:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True, pin_memory=True)
        for train_batch in train_loader:
            train_data, train_label = train_batch
            train_data = Variable(train_data)
            train_label = Variable(train_label)
            if use_gpu:
                train_data = train_data.cuda()
                train_label = train_label.cuda()
            model.train(True)
            model.zero_grad()
            train_loss, train_accuracy = compute_loss_accuracy(
                data=train_data, label=train_label)
            train_loss.backward()
            clip_grad_norm(parameters=params, max_norm=1)
            optimizer.step()
            tb_xp_train.add_scalar_dict(
                data={'loss': train_loss.data[0],
                      'accuracy': train_accuracy.data[0]},
                step=iter_cnt)

            if iter_cnt % 50 == 49:
                for valid_batch in valid_loader:
                    valid_data, valid_label = valid_batch
                    # Dirty, but don't get other solutions
                    break
                valid_data = Variable(valid_data, volatile=True)
                valid_label = Variable(valid_label, volatile=True)
                if use_gpu:
                    valid_data = valid_data.cuda()
                    valid_label = valid_label.cuda()
                model.train(False)
                valid_loss, valid_accuracy = compute_loss_accuracy(
                    data=valid_data, label=valid_label)
                tb_xp_valid.add_scalar_dict(
                    data={'loss': valid_loss.data[0],
                          'accuracy': valid_accuracy.data[0]},
                    step=iter_cnt)
                save_path = '{}/{}'.format(save_dir, iter_cnt)
                torch.save(model, save_path)
            iter_cnt += 1
            if iter_cnt == max_iter:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the model using MNIST dataset.')
    parser.add_argument('--data', required=True,
                        help='The path to save MNIST dataset, or '
                             'the path the dataset is located')
    parser.add_argument('--model', required=True, choices=['lstm', 'bnlstm'],
                        help='The name of a model to use')
    parser.add_argument('--save', required=True,
                        help='The path to save model files')
    parser.add_argument('--hidden-size', required=True, type=int,
                        help='The number of hidden units')
    parser.add_argument('--pmnist', default=False, action='store_true',
                        help='If set, it uses permutated-MNIST dataset')
    parser.add_argument('--batch-size', required=True, type=int,
                        help='The size of each batch')
    parser.add_argument('--max-iter', required=True, type=int,
                        help='The maximum iteration count')
    parser.add_argument('--gpu', default=False, action='store_true',
                        help='The value specifying whether to use GPU')
    args = parser.parse_args()
    main()
