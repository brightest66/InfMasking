import torch
# import torch.distributed as dist


import torch
#最大化特征熵来避免模型坍塌
def entropy_gradeint(keys):
   dlog_q=com_score(keys)
   grad_en=torch.mean(torch.sum(-dlog_q.detach()*keys,-1))
   return grad_en

@torch.no_grad()
def com_score(keys,eta=0.01):
   batch_size= keys.size()[0]
   pairwise_similar=torch.mm(keys,torch.t(keys))#计算相似度
   tau=heuristic_kernel_width(keys,keys,pairwise_similar)#核宽度，当特征高度相似，导致pairwise_similar趋于0，tau过小
   tau = torch.clamp(tau, min=0.1)  # 防止tau过小,当tau的值为0.0787,就会爆炸
   scaled_similar=pairwise_similar/tau
   # max_val = torch.max(scaled_similar)
   # scaled_similar = scaled_similar - max_val  # 避免指数爆炸
   
   Gram=torch.exp(scaled_similar)#核矩阵，会出现指数爆炸，当tau过下的时候
   x_row=torch.unsqueeze(keys,-2)
   diff =x_row/tau
   grad_x =torch.sum(Gram.unsqueeze(-1)*diff,-2)
   Gram_ivs=torch.inverse(Gram+eta*torch.eye(batch_size).cuda())

   dlog_q= -torch.einsum('ik,kj->ij',[Gram_ivs,grad_x])
   return  dlog_q
@torch.no_grad()
def heuristic_kernel_width( x_samples, x_basis,pairwise_similar):
        n_samples = x_samples.size()[-2]
        n_basis = x_basis.size()[-2]
      #   pairwise_dist = torch.arccos(pairwise_similar)
        pairwise_dist = 1-pairwise_similar
        k = n_samples * n_basis // 2
        top_k_values = torch.topk(torch.reshape(pairwise_dist, [-1, n_samples * n_basis]), k=k)[0]
        kernel_width = torch.reshape(top_k_values[:, -1], x_samples.size()[:-2])
        return kernel_width.detach()

  
