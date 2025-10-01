from torch import nn
import torch
from collections import OrderedDict
from typing import Dict, List
# Local imports
from pl_modules.base import BaseModel
from losses.infmasking_loss import InfMaskingLoss
from models.inffusion import InfFusion

class InfMasking(BaseModel):
    def __init__(self,
                 encoder: InfFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict
                 ):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters 
            loss_kwargs: Hyper-parameters for the InfMasking loss
        """
        super(InfMasking, self).__init__(optim_kwargs)

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

        # Build the loss
        self.loss = InfMaskingLoss(**loss_kwargs)

        # initialize the parameters
        self.epoch = -100
        self.total_epochs = -100

    def set_current_epoch(self, epoch):
        self.epoch = epoch
    
    def set_total_epochs(self, total_epochs):
        self.total_epochs = total_epochs

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))


    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]):
        # compute features for all modalities
        all_masks = self.gen_all_possible_masks(len(x1))

        z1,mask_out1 = self.encoder(x1, mask_modalities=all_masks)
        z2,mask_out2 = self.encoder(x2, mask_modalities=all_masks)
        z1 = [self.head(z) for z in z1] 
        z2 = [self.head(z) for z in z2] 
        for i in range(len(mask_out1)):
            mask_out1[i] = [self.head(ttt) for ttt in mask_out1[i]]
            mask_out2[i] = [self.head(ttt) for ttt in mask_out2[i]]
        return {'aug1_embed': z1, 
                'aug2_embed': z2,
                'mask_out1': mask_out1, 
                'mask_out2': mask_out2,
                "prototype": -1,
                "epoch": self.epoch,
                "max_epochs": self.total_epochs}
    

    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Examples:
        *   For n_mod==2:
            masks == [[True, False], [False, True], [True, True]]
        *   For n_mod == 3:
            masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks
    
    
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract multimodal features from the encoder.
           Args:
                loader: Dataset loader to serve `(X, y)` tuples.
                kwargs: given to `encoder.forward()`
           Returns: 
                Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor): # needs to cast it as list of one modality
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode(): # no gradient computation
                # compute output
                output, _ = self.encoder(X_, **kwargs)
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)
