import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


def train_CC(
    net,
    optimizer,
    instance_loss,
    cluster_loss,
    train_dataloader,
    args):

    net.train()
    for param in net.parameters():
        param.requires_grad = True

    loss_list = []
    for step, ((x_w, x_s, x), _) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        x_w = x_w.to('cuda')
        x_s = x_s.to('cuda')

        z1, z2, c1, c2 = net(x_w, x_s)
        loss1 = instance_loss(z1, z2)
        loss2 = cluster_loss(c1, c2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        loss_list.append(loss)

        if step % int(len(train_dataloader) / 5) == 0:
            args.write_log(args.log_path, "step[{}/{}] loss: {} instance loss: {} cluster loss: {}\n".format(
                step, len(train_dataloader), torch.mean(torch.Tensor(loss_list)), loss1, loss2))
    
    args.write_log(args.log_path, 'step loss: {}\n'.format(torch.mean(torch.Tensor(loss_list))))
    print('step loss: {}'.format(torch.mean(torch.Tensor(loss_list))))
    return net