import copy
import os
from os.path import join as pjoin

import torch
from tensorboardX import SummaryWriter

from utils.timeutil import now_str
from metric.metric import acc_perclass


def train(model, device, train_loader, optimizer, loss_fn, epoch, writter):   # 训练模型
    model.train()
    step = len(train_loader) // 3

    for batch_idx, s in enumerate(train_loader):
        x, y = s
        x, y = x.to(device), y.to(device)
        out = model(x)

        if isinstance(out, list):
            y_pred = out[0]
        else:
            y_pred = out
        model.zero_grad()             # 梯度清零
        loss = loss_fn(y_pred, y.squeeze())  # 得到loss
        loss.backward()
        optimizer.step()

        writter.add_scalar('loss/train_loss', loss.item(), (epoch-1) * len(train_loader) + batch_idx+1)
        if (batch_idx + 1) % step == 0:    # 打印loss
            print_str = 'Train Epoch: {} [{}/{} ({:>6.2f}% )]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(y),
                                                                                    len(train_loader.dataset),
                                                                                    100. * (batch_idx+1) /
                                                                                    len(train_loader), loss.item())
            print(print_str)


def test(model, device, test_loader, loss_fn, n_classes):    # 测试模型, 得到测试集评估结果
    model.eval()
    test_loss = 0.0
    acc = 0

    preds, gt = [], []

    for batch_idx, s in enumerate(test_loader):
        x, y = s
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)

        if isinstance(out, list):
            y_ = out[0]
        else:
            y_ = out
        test_loss += loss_fn(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]   # .max(): 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
        preds.append(pred)
        gt.append(y)
    test_loss /= len(test_loader)
    preds = torch.cat(preds)
    gt = torch.cat(gt)
    accs = acc_perclass(preds, gt, n_classes)
    print_str = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, acc,
                                                                                     len(test_loader.dataset),
                                                                                     100. * acc /
                                                                                     len(test_loader.dataset))
    print(print_str)
    for c in range(n_classes):
        print('Test set Accuracy for class {}: {:.2f}%'.format(c+1, accs[c]))
    return acc / len(test_loader.dataset) * 100, accs


def run(train_loader, test_loader, model, optimizer, loss_fn, n_classes, num_epochs=4, gpu_idx=0,
        output_embeddings=False, output_model=False):

    logdir = pjoin('../run', now_str())
    os.mkdir(logdir)
    writter = SummaryWriter(logdir)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_idx > 0:
        torch.cuda.set_device(gpu_idx)
    model.to(DEVICE)

    NUM_EPOCHS = num_epochs

    best_acc = 0.0
    best_model = None

    for epoch in range(1, NUM_EPOCHS + 1):  # 3epoch
        train(model, DEVICE, train_loader, optimizer, loss_fn, epoch, writter)
        acc, acc_class = test(model, DEVICE, test_loader, loss_fn, n_classes)
        if best_acc < acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            torch.save(best_model, pjoin('../run', logdir, 'best_model.pt'))

        print("acc is: {:.4f}%, best acc is {:.4f}%\n".format(acc, best_acc))
        writter.add_scalar('acc/val_acc', acc, epoch)
        for c in range(n_classes):
            writter.add_scalar('acc/acc_cls_%d' % (c+1), acc_class[c], epoch)

    if output_embeddings:
        return best_acc, best_model.label_embeddings.data.cpu()

    if output_model:
        return best_acc, best_model
    return best_acc
