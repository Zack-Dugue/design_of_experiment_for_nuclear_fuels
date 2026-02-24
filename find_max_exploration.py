import torch
import torch.nn as nn
import torch.functional as F

class SequenceEnsemble(nn.Module):
    def __init__(self, ensemble_list : nn.ModuleList):
        super(SequenceEnsemble, self).__init__()
        self.ensemble_list = ensemble_list
        self.n = len(ensemble_list)

    def forward(self,x,t):
        bsz , L  = t.size()
        outputs = torch.zeros([self.n,bsz, L])
        for i, model in enumerate(self.ensemble_list):
            y = model.decode(x,t)
            outputs[i,:,:] = y
        return outputs.std(0).mean()

