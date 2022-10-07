from mmcv.runner import HOOKS, Hook
from tensorboardX import SummaryWriter
import os


@HOOKS.register_module()
class TensorboardLossHook(Hook):

    def __init__(self, tensorboard_log_dir):
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.tensorboard_log_dir = tensorboard_log_dir

    def after_train_iter(self, runner):
        loss = runner.outputs['loss'].detach().item()
        self.writer.add_scalar("loss", loss, runner.iter)

    def after_run(self, runner):
        self.writer.export_scalars_to_json(
            os.path.join(self.tensorboard_log_dir, "all_scalars.json")
        )
        self.writer.close()
