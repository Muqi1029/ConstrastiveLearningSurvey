from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import logging
import os
from torch.cuda.amp import GradScaler, autocast
from utils import save_config_file, accuracy, save_checkpoint
from tqdm import tqdm
from loss.loss import info_nce_loss


class Trainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 optimizer, scheduler, args) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.writer = SummaryWriter()
        logging.basicConfig(
            filename=os.path.join(
                self.writer.log_dir, 'training.log'),
            level=logging.DEBUG)

    def train(self, train_loader=None):
        if train_loader is None:
            train_loader = self.dataloader

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    loss, logits, labels = info_nce_loss(
                        features,
                        device=self.args.device,
                        batch_size=self.args.batch_size,
                        temperature=self.args.temperature)

                self.optimizer.zero_grad()
                # 使用梯度缩放器反向传播
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar(
                        'acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar(
                        'acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[
                                           0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        },
            is_best=False,
            filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
