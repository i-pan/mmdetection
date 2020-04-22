import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    from . import sigmoid_focal_loss_cuda
    CUDA_EXT = True
except ImportError:
    CUDA_EXT = False
    print('Unable to import `sigmoid_focal_loss_cuda`')
    print('>>Using torch-based `sigmoid_focal_loss` ...')


class SigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        loss = sigmoid_focal_loss_cuda.forward(input, target, num_classes,
                                               gamma, alpha)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss,
                                                   num_classes, gamma, alpha)
        return d_input, None, None, None, None


if CUDA_EXT:
    sigmoid_focal_loss = SigmoidFocalLossFunction.apply
else:
    import torch.nn.functional as F
    def sigmoid_focal_loss(pred,
                           target,
                           gamma=2.0,
                           alpha=0.25):
        pred_sigmoid = pred.sigmoid()
        # one-hot encode, then get rid of background class with [:,1:]
        target = F.one_hot(target.long(), pred.shape[-1]+1)[:,1:].type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        return loss


# TODO: remove this module
class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        assert logits.is_cuda
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(gamma={}, alpha={})'.format(
            self.gamma, self.alpha)
        return tmpstr
