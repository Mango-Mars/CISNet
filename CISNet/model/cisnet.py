import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses.dice import DiceLoss
import timm
from model.util.sync_batchnorm import SynchronizedBatchNorm2d
from model.aspp import build_aspp
from model.decoder import build_decoder
from change_head import build_detector
import pytorch_lightning as pl
import torchmetrics

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        weights = torch.ones(num, requires_grad=True)
        self.weights = torch.nn.Parameter(weights)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.weights[i] ** 2) * loss + torch.log(1 + self.weights[i] ** 2)
        return loss_sum

class CISNet(pl.LightningModule):
    def __init__(self, hparams, metric=torchmetrics.IoU(num_classes=4, reduction="none", absent_score=1.0), n_classes=4):
        super(CISNet, self).__init__()
        self.save_hyperparameters(hparams)
        self.n_channels_of_input = 1
        self.non_linearity = "Leaky_ReLU"
        self.metric = metric
        self.n_classes = n_classes
        self.uncertain_loss = AutomaticWeightedLoss(2)

        if self.hparams.sync_bn is True:
            batchNorm = SynchronizedBatchNorm2d
        else:
            batchNorm = nn.BatchNorm2d


        self.backbone = timm.create_model('convnextv2_tiny', num_classes=4, in_chans=1,
                                          features_only=True, out_indices=(-4, -3, -2, -1,), pretrained=True,
                                          pretrained_cfg_overlay=dict(
                                              file='./pytorch_model.bin'))

        self.decoder = build_decoder(self.n_classes)
        self.aspp = build_aspp(batchNorm)
        self.change_detector = build_detector(self.hparams.in_channels, self.hparams.inner_channels, self.hparams.out_channels,
                                            self.hparams.scale, self.hparams.num_convs, self.hparams.drop_rate)


    def forward(self, input, is_train=True):
        if is_train:
            x, low_level_feat_128, low_level_feat_64, low_level_feat_32, low_level_feat_16 = self.backbone(input[:, :1, :, :])
            x1, _, _, _, _ = self.backbone(input[:, 1:, :, :])
            x_feature = self.aspp(x)
            x1_feature = self.aspp(x1)

            x = self.decoder(x_feature, [low_level_feat_128, low_level_feat_64, low_level_feat_32, low_level_feat_16])
            logit = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

            change_xvx1_logit = self.change_detector(torch.cat([x_feature, x1_feature], dim=1))
            change_x1vx_logit = self.change_detector(torch.cat([x1_feature, x_feature], dim=1))

            return logit, change_x1vx_logit, change_xvx1_logit

        else:
            x, low_level_feat_128, low_level_feat_64, low_level_feat_32, low_level_feat_16 = self.backbone(input)
            x_feature = self.aspp(x)

            x = self.decoder(x_feature, [low_level_feat_128, low_level_feat_64, low_level_feat_32, low_level_feat_16])
            logit = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return logit


    def class_wise_iou(self, y_hat, mask):
        ious = []
        classwise_ious = self.metric(y_hat.argmax(dim=1), mask)
        ious.append(torch.mean(classwise_ious))
        for i in classwise_ious:
            ious.append(i)
        return ious


    def give_prediction_for_batch(self, batch, is_train=False):
        x, y, x_name = batch
        y_hat = self.forward(x, is_train)
        return y_hat

    def adapt_mask(self, y):
        mask_type = torch.float32 if self.n_classes == 1 else torch.long
        y = y.squeeze(1)
        y = y.type(mask_type)
        return y

    def calc_loss(self, y_hat, mask, is_train=False):
        criterion_ce = nn.CrossEntropyLoss()
        criterion_dice = DiceLoss('multiclass')
        if is_train:
            seg_mask = mask[:, 0, :, :]
            mask_1 = mask[:, 1, :, :]

            y_0 = torch.where(seg_mask == 2, torch.ones_like(seg_mask), torch.zeros_like(seg_mask))
            y_1 = torch.where(mask_1 == 2, torch.ones_like(mask_1), torch.zeros_like(mask_1))

            xor = y_0 + y_1
            change_mask = torch.where(xor == 1, torch.ones_like(xor), torch.zeros_like(xor))

            seg_mask = self.adapt_mask(seg_mask)
            change_mask = self.adapt_mask(change_mask)
            change_loss = 0.5 * criterion_ce(y_hat[1], change_mask) + 0.5 * criterion_ce(y_hat[2], change_mask)
            seg_loss = criterion_ce(y_hat[0], seg_mask)

            dice_loss = criterion_dice(y_hat[0], seg_mask)
            seg_loss = 0.5 * seg_loss + 0.5 * dice_loss
            loss = self.uncertain_loss(0.5*seg_loss, 0.5*change_loss)

            metric = self.class_wise_iou(y_hat[0], seg_mask)
            return loss, metric

        else:
            mask = self.adapt_mask(mask)
            loss = criterion_ce(y_hat, mask) + criterion_dice(y_hat, mask)
            metric = self.class_wise_iou(y_hat, mask)
            return loss, metric


    def make_batch_dictionary(self, loss, metric, name_of_loss):
        """ Give batch_dictionary corresponding to the number of metrics for zone segmentation """
        """ name_of_loss: 'loss' or 'val_loss' or 'test_loss' """

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            name_of_loss: loss,
            # info to be used at epoch end
            "IoU": metric[0],
            "IoU NA Area": metric[1],
            "IoU Stone": metric[2],
            "IoU Glacier": metric[3],
            "IoU Ocean and Ice Melange": metric[4]
        }
        return batch_dictionary


    def log_metric(self, outputs, train_or_val_or_test):
        # calculating average metric
        avg_iou = torch.stack([x["IoU"] for x in outputs]).mean()
        avg_iou_na = torch.stack([x["IoU NA Area"] for x in outputs]).mean()
        avg_iou_stone = torch.stack([x["IoU Stone"] for x in outputs]).mean()
        avg_iou_glacier = torch.stack([x["IoU Glacier"] for x in outputs]).mean()
        avg_iou_ocean = torch.stack([x["IoU Ocean and Ice Melange"] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU", avg_iou, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_NA_Area", avg_iou_na, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_Stone", avg_iou_stone, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_Glacier", avg_iou_glacier, self.current_epoch)
        self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_Ocean_and_Ice_Melange", avg_iou_ocean, self.current_epoch)
        if train_or_val_or_test == "Val":
            self.log('avg_metric_validation', avg_iou)
            self.log('avg_iou_na_validation', avg_iou_na)
            self.log('avg_iou_stone_validation', avg_iou_stone)
            self.log('avg_iou_glacier_validation', avg_iou_glacier)
            self.log('avg_iou_ocean_validation', avg_iou_ocean)


    def training_step(self, batch, batch_idx):
        x, mask, x_name = batch
        y_hat = self.give_prediction_for_batch(batch, is_train=True)
        train_loss, metric = self.calc_loss(y_hat, mask, is_train=True)
        self.log('train_loss', train_loss)
        return self.make_batch_dictionary(train_loss, metric, "loss")


    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Train")


    def validation_step(self, batch, batch_idx):
        x, mask, x_name = batch
        y_hat = self.give_prediction_for_batch(batch, is_train=False)
        val_loss, metric = self.calc_loss(y_hat, mask, is_train=False)
        self.log('val_loss', val_loss)
        return self.make_batch_dictionary(val_loss, metric, "val_loss")



    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Val")
        self.log('avg_loss_validation', avg_loss)


    def test_step(self, batch, batch_idx):
        x, mask, x_name = batch
        y_hat = self.give_prediction_for_batch(batch, is_train=False)
        test_loss, metric = self.calc_loss(y_hat, mask, is_train=False)
        self.log('test_loss', test_loss)
        return self.make_batch_dictionary(test_loss, metric, "test_loss")


    def test_step_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.log_metric(outputs, "Test")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr,
                                                      cycle_momentum=False,
                                                      step_size_up=30000)
        scheduler_dict = {
            'scheduler': scheduler,

            'interval': 'step'
        }
        return [optimizer], [scheduler_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CISNet")
        parser.add_argument('--base_lr', default=4e-5)
        parser.add_argument('--max_lr', default=2e-4)
        parser.add_argument('--batch_size', default=16)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--sync_bn', default=True, type=bool)

        # Hyperparameters for augmentation
        parser.add_argument('--bright', default=0.1, type=float)
        parser.add_argument('--wrap', default=0.1, type=float)
        parser.add_argument('--noise', default=0.5, type=float)
        parser.add_argument('--rotate', default=0.5, type=float)
        parser.add_argument('--flip', default=0.3, type=float)


        # Change detector arguments
        parser.add_argument('--in_channels', default=512, type=int)
        parser.add_argument('--inner_channels', default=16, type=int)
        parser.add_argument('--out_channels', default=2, type=int)
        parser.add_argument('--scale', default=32., type=float)
        parser.add_argument('--num_convs', default=4, type=int)
        parser.add_argument('--drop_rate', default=0.2, type=float)


        return parent_parser
