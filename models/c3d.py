"""C3D"""
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from torchsummary import summary


class C3DBN(nn.Module):
    """C3D with BN and pool5 to be AdaptiveAvgPool3d(1)."""

    def __init__(self, num_classes=3):
        super(C3DBN, self).__init__()
    
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.relu3b = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.relu4a = nn.ReLU()
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.relu4b = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.relu5a = nn.ReLU()
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.relu5b = nn.ReLU()

        self.pool5 = nn.AdaptiveAvgPool3d(1)
        
        # Multi-Label Classifier
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_classes)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.relu5a(x)
        x = self.conv5b(x)
        x = self.bn5b(x)
        x = self.relu5b(x)

        x = self.pool5(x)
        x = x.view(-1, 512)

        x = self.sigmoid(self.classifier(x))

        return x


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of weights
    """

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            # print (name, param.dataset)

            yield param


def get_2x_lr_params(model):
    """
    This generator returns all the parameters of bias
    """

    for name, param in model.named_parameters():
        if 'bias' in name and param.requires_grad:
            # print (name, param.dataset)

            yield param
    

# if __name__ == '__main__':
#     import torch
#     from torchsummary import summary
    
#     c3d = C3DBN(num_classes=3).cuda()
#     input_ = torch.randn((4, 1, 8, 112, 112)).cuda()
    
#     output = c3d(input_)
#     print("Region output size: ", output.size())
#     summary(c3d, (1, 8, 112, 112))
