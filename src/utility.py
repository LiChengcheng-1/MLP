import numpy as np
def adjust_lr(epoch, optimizer, lr):
    # optimize lr
    step = [10, 20, 30, 40]
    base_lr = lr
    lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    return lr