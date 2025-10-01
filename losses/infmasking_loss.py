import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad
from losses.entropy import entropy_gradeint

class InfMaskingLoss(nn.Module):
    def __init__(self, temperature=0.1, weights=None, cross=False, only_mask_last=False,
                 mask_lambda=0.25,penalty=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        self.mask_lambda = mask_lambda
        self.cross = cross
        self.INF = 1e8
        self.penalty = penalty
        self.only_mask_last = only_mask_last

    def infonce(self, z1, z2): # InfoNCE Loss
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def get_mask_loss(self, q, mask_out, ratio):
        """
        :param q: [bsize, feature_dim]
        :param mask_out: List of tensors with shape (bsize, feature_dim), length T
        :param ratio: float
        """
        B, E = q.shape
        T = len(mask_out)
        k_pos = torch.cat(mask_out, dim=0) # dim [T*B, E]

        # compute mean and covariance
        k_tryies = k_pos.shape[0] / B
        k_tryies = int(k_tryies) 
        k_new_shape = k_pos.view(k_tryies, B, -1) # dim [T, B, E]

        # compute mean
        k_means = torch.mean(k_new_shape, dim=0, keepdim=True) # dim [1, B, E]
        value_minus_mean = k_new_shape - k_means # minus mean, dim [T, B, E]
        # compute covariance
        value_minus_mean_part_1 = value_minus_mean.permute(1, 0, 2) # dim [B, T, E]
        value_minus_mean_part_2 = value_minus_mean.permute(1, 2, 0) # dim [B, E, T]
        k_sigma = torch.bmm(value_minus_mean_part_2, value_minus_mean_part_1) / k_tryies # dim [B, E, E]

        # apply softplus to covariance 
        k_sigma = k_sigma * ratio / self.temperature

        # positive logits: Bx1
        k_means = k_means.squeeze(dim=0) # dim [B, E]
        l_pos = torch.einsum('nc,nc->n', [q, k_means + 0.5 * torch.bmm(k_sigma, q.unsqueeze(dim=-1)).squeeze(dim=-1)]).unsqueeze(-1) # dim [B, 1]

        k_neg = torch.stack(mask_out, dim=0) # dim [T, B, E]

        l_neg = torch.einsum('be,tne->tbn', [q, k_neg]) # dim [T, B, B]
        # delete the diagonal
        l_neg = l_neg - self.INF * torch.eye(B, device=q.device).unsqueeze(0).expand(T, -1, -1) # dim [T, B, B]
        l_neg =  list(torch.unbind(l_neg, dim=0))
        self_sim = torch.einsum('nc,bc->nb', [q, q])-self.INF * torch.eye(B, device=q.device)
        # add self-similarity to the negative logits
        l_neg.append(self_sim) 
        # .append(torch.einsum('nc,bc->nb', [q, q])-self.INF * torch.eye(B, device=q.device)) # dim [T+1, B, B]
        l_neg= torch.cat(l_neg, dim=1) # dim [B, (T+1)*B]

        # logits: Bx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        
        # compute the masklabel
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        mask_loss = 0.5 * torch.bmm(torch.bmm(q.reshape(B, 1, E), k_sigma), q.reshape(B, E, 1)).mean() / self.temperature

        criterion = nn.CrossEntropyLoss().cuda()
        cls_loss = criterion(logits, labels)

        return mask_loss + cls_loss


    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z is stored in "aug1_embed" and "aug2_embed". 
                - "mask_out1", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "mask_out2", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "epoch", current epoch number.
                - "max_epochs", total number of epochs.
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        mask_out1, mask_out2 = outputs["mask_out1"][prototype], outputs["mask_out2"][prototype] # List
        epoch, max_epochs = outputs["epoch"], outputs["max_epochs"]

        assert len(z1) == len(z2)
        assert len(mask_out1) == len(mask_out2)

        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2) 
        z1, z2 = Z[:n_emb], Z[n_emb:]

        mask_out1 = [func.normalize(mask_z, p=2, dim=-1) for mask_z in mask_out1]
        mask_out2 = [func.normalize(mask_z, p=2, dim=-1) for mask_z in mask_out2]
        MASK_OUT = all_gather_batch_with_grad(mask_out1 + mask_out2)
        num_mask = len(mask_out1)
        mask_out1, mask_out2 = MASK_OUT[:num_mask], MASK_OUT[num_mask:]

        loss = []
        acc = []
        loss_uniform=0

        # compute mask loss
        ratio = self.mask_lambda * ((epoch + 1) * 1.0 / max_epochs)
        if self.cross:
            loss1 = self.get_mask_loss(z1[prototype], mask_out2, ratio)
            loss2 = self.get_mask_loss(z2[prototype], mask_out1, ratio)
        else:
            loss1 = self.get_mask_loss(z1[prototype], mask_out1, ratio)
            loss2 = self.get_mask_loss(z2[prototype], mask_out2, ratio)

        loss.append(loss1)
        loss.append(loss2)

        if self.penalty is not None:
            if self.only_mask_last:
                for i in range(num_mask):
                    loss_uniform += entropy_gradeint(mask_out1[i])
                    loss_uniform += entropy_gradeint(mask_out2[i])
                loss_uniform /= (2 * num_mask)
            else:
                for i in range(n_emb):
                    loss_uniform += entropy_gradeint(z1[i])
                    loss_uniform += entropy_gradeint(z2[i])
                loss_uniform /= (2 * n_emb)

        # compute infonce loss(unimodal and multimodal)
        for i in range(n_emb):
            loss3, acc1 = self.infonce(z1[i], z2[prototype]) # z‘’
            loss4, acc2 = self.infonce(z2[i], z1[prototype]) # z‘
            loss.append((loss3 + loss4) / 2.)
            acc.append((acc1 + acc2) / 2.)
        ssl_acc = {"ssl_acc_%i"%i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i"%i: l for i, l in enumerate(loss)}

        if self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            loss = torch.mean(torch.stack(loss)) 
        if self.penalty is not None:
            loss -= self.penalty*loss_uniform

        acc = torch.mean(torch.stack(acc))
       
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)