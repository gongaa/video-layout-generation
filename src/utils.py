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


# def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
#     """
#     Calcute the confusion matrix by given label and pred
#     """
#     output = pred.cpu().numpy().transpose(0, 2, 3, 1)
#     seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
#     seg_gt = np.asarray(
#     label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

#     ignore_index = seg_gt != ignore
#     seg_gt = seg_gt[ignore_index]
#     seg_pred = seg_pred[ignore_index]

#     index = (seg_gt * num_class + seg_pred).astype('int32')
#     label_count = np.bincount(index)
#     confusion_matrix = np.zeros((num_class, num_class))

#     for i_label in range(num_class):
#         for i_pred in range(num_class):
#             cur_index = i_label * num_class + i_pred
#             if cur_index < len(label_count):
#                 confusion_matrix[i_label,
#                                  i_pred] = label_count[cur_index]
#     return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr