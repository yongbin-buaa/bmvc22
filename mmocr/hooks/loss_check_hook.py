from mmcv.runner import HOOKS, Hook
from torch.nn.utils import clip_grad


@HOOKS.register_module()
class LossCheckHook(Hook):
    def __init__(self):
        pass

    def after_train_iter(self, runner):
        if runner.outputs['loss'].isnan().sum().item() > 0:
            print(runner.epoch, runner.iter)
            exit(0)
