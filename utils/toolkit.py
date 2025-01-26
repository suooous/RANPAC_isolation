import numpy as np
import torch

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor2numpy(x):
    """将tensor转换为numpy数组
    
    Args:
        x: 输入tensor
        
    Returns:
        numpy.ndarray: 转换后的numpy数组
    """
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot

def accuracy(y_pred, y_true, nb_old, class_increments):
    """计算分类准确率
    
    Args:
        y_pred: 预测标签
        y_true: 真实标签
        nb_old: 已知类别数量
        class_increments: 类别增量列表
        
    Returns:
        tuple: (总体准确率, 分组准确率)
    """
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    
    # 计算总体准确率
    acc_total = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # 计算分组准确率
    for classes in class_increments:
        idxes = np.where(
            np.logical_and(y_true >= classes[0], y_true <= classes[1])
        )[0]
        label = "{}-{}".format(
            str(classes[0]).rjust(2, "0"), str(classes[1]).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    return acc_total, all_acc
