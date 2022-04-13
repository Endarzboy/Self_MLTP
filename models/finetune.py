import sys

import torch
from torch import nn

sys.path.append('..')
from dataset import dataloader
from torchvision.models import video


class FineTune(nn.Module):  # Retraining the pre-trained model

    def __init__(self, weight, device, num_classes, net=None, pretrained=False):
        super(FineTune, self).__init__()

        model = None

        if net == 'r3d':
            model = video.r3d_18(pretrained=pretrained).to(device)
            model.load_state_dict(torch.load(weight, map_location='cpu'), strict=False)
        elif net == 'r21d':
            model = video.r2plus1d_18(pretrained=pretrained).to(device)
            model.load_state_dict(torch.load(weight, map_location='cpu'), strict=False)

        # If we want to use c3d model as a fixed feature extractor, we can uncomment the following lines of code.
        # for param in model.parameters():
        #     param.requires_grad = False  # if `True`: it's the same as fine-tuning the whole net without the fc layer.

        # Fetch the feature extractor part excluding the last fully connected layers.
        self.conv_layers = list(model.children())[:-1]
        self.base_model = nn.Sequential(*self.conv_layers)

        # Resetting the final three linear layers with output class of size 5
        self.classifier = nn.Linear(model.fc.in_features, num_classes)


    def forward(self, x):
        x = self.base_model(x)
        x = x.view(-1, 512)  # torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x


# if __name__ == '__main__':
#     from torchsummary import summary
#     chkpnt_c3d = "../checkpoints/2021-05-05_23-04_c3d_50/c3d_checkpoint-000040.pth"
#     model_ = FineTune(chkpnt_c3d, attrs, 'c3d').cuda()
#     input_ = torch.randn((2, 3, 16, 112, 112)).cuda()
#     output = model_(input_)
#     print("Shape of base output: ", output.size())

#     summary(model_, (3, 16, 112, 112))
