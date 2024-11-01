import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        w = 224
        c1 = 5
        p1 = 2
        self.conv1 = nn.Conv2d(1, 16, c1)
        self.norm1 = nn.BatchNorm2d(16, affine=True)
        self.pool1 = nn.MaxPool2d(p1)
        w = self.new_w(w, c1, p1)

        c2 = 5
        p2 = 2
        self.conv2 = nn.Conv2d(16, 32, c2)
        self.norm2 = nn.BatchNorm2d(32, affine=True)
        self.pool2 = nn.MaxPool2d(p2)
        w = self.new_w(w, c2, p2)

        c3 = 5
        p3 = 2
        self.conv3 = nn.Conv2d(32, 64, c3)
        self.norm3 = nn.BatchNorm2d(64, affine=True)
        self.pool3 = nn.MaxPool2d(p3)
        w = self.new_w(w, c3, p3)

        c4 = 5
        p4 = 2
        self.conv4 = nn.Conv2d(64, 128, c4)
        self.norm4 = nn.BatchNorm2d(128, affine=True)
        self.pool4 = nn.MaxPool2d(p4)
        w = self.new_w(w, c4, p4)

        c5 = 5
        p5 = 2
        self.conv5 = nn.Conv2d(128, 256, c5)
        self.norm5 = nn.BatchNorm2d(256, affine=True)
        self.pool5 = nn.MaxPool2d(p5)
        w = self.new_w(w, c5, p5)

        print('Last layer dimensions', w)
        
        self.fc1 = nn.Linear(256 * w * w, 12288)
        self.fc2 = nn.Linear(12288, 136)
        
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)

    def new_w(self, w, c_kernel, p_kernel):
        return round(((w - c_kernel) + 1) / p_kernel)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop1(x)        

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)        
        x = self.pool3(x)
        x = self.drop1(x)        

        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.drop1(x)        

        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.pool5(x)
        x = self.drop1(x)
        
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(32,64,5)  
#         self.conv3 = nn.Conv2d(64,128,5)
#         self.conv4 = nn.Conv2d(128,256,5)
#         self.dropout = nn.Dropout(0.2)
#         self.fc1 = nn.Linear(256*10*10,12800)
#         self.fc2 = nn.Linear(12800,136)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.dropout(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.dropout(x)
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.dropout(x)
#         x = self.pool(F.relu(self.conv4(x)))
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x