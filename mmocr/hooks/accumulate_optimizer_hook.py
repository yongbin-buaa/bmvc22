from mmcv.runner import HOOKS, Hook
from torch.nn.utils import clip_grad
import torch


@HOOKS.register_module()
class AccumulateOptimizerHook(Hook):

    def __init__(self, grad_clip=None, accumulate_factor=1, detect_anomaly=False):
        self.grad_clip = grad_clip
        assert isinstance(accumulate_factor, int)
        self.accumulate_factor = accumulate_factor
        self.detect_anomaly = detect_anomaly

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        if self.accumulate_factor == 1:
            runner.optimizer.zero_grad()
            if self.detect_anomaly:
                with torch.autograd.detect_anomaly():
                    runner.outputs['loss'].backward()
            else:
                runner.outputs['loss'].backward()
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
            runner.optimizer.step()
        else:
            runner.outputs['loss'] = runner.outputs['loss'] / self.accumulate_factor
            if self.detect_anomaly:
                with torch.autograd.detect_anomaly():
                    runner.outputs['loss'].backward()
            else:
                runner.outputs['loss'].backward()

            if (runner.iter + 1) % self.accumulate_factor == 0:
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters())
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                                runner.outputs['num_samples'])
                runner.optimizer.step()
                runner.optimizer.zero_grad()
