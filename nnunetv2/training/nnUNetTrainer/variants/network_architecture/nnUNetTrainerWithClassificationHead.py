from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class SegmentationNetworkWithClassificationHead(nn.Module):
    def __init__(self, seg_network: nn.Module, features_per_stage: List[int],
                 num_hidden_features: int, num_classes: int):
        super().__init__()
        self.seg_network = seg_network
        assert hasattr(self.seg_network, 'encoder')
        assert hasattr(self.seg_network, 'decoder')
        self.encoder = self.seg_network.encoder
        self.decoder = self.seg_network.decoder

        self.conv_block = nn.Sequential(
            nn.Conv3d(features_per_stage[-1], features_per_stage[-1], 1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv3d(features_per_stage[-1], features_per_stage[-1], 1, padding='same'),
            nn.LeakyReLU()
        )
        self.max_pool = nn.AdaptiveMaxPool3d(output_size=1)
        self.class_head = nn.Sequential(
            nn.Linear(features_per_stage[-1], num_hidden_features),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_features, num_hidden_features),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_features, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        skips = self.seg_network.encoder(x)
        seg_output = self.seg_network.decoder(skips)
        x = self.conv_block(skips[-1])
        x = self.max_pool(x)
        B = x.shape[0]
        x = self.class_head(x.view(B, -1))
        return seg_output, x


class nnUNetTrainerWithClassificationHead(nnUNetTrainer):
    def configure_optimizers(self):
        self.initial_lr = 3e-4
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        segmentation_network = nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels,
                                                        num_output_channels, enable_deep_supervision)

        return SegmentationNetworkWithClassificationHead(segmentation_network,
                                                         arch_init_kwargs["features_per_stage"],
                                                         128, 3)
