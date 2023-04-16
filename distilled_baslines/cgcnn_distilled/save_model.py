import torch
from model import *

cuda = torch.cuda.is_available()
model_name = "../../model/pretrain_model.pth"
print(model_name)
if cuda:
    model = torch.load(model_name)
else:
    model = torch.load(model_name, map_location=torch.device('cpu'))

filename='../../model/crysgnn_state_checkpoint_v2.pth.tar'
torch.save(model.state_dict(), filename)
