from lib import *


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        init.constant_(self.weight, self.scale)
    
    def forward(self, x):
        # x.size() = (batch_size, channels, height, width)
        # norm.size() = (batch_size, 1, height, width)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        #weight.size() = (512,) -> (1,512,1,1)
        weight = self.weight.unsqueeze(dim=0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weight*x
    

