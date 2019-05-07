import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 8, 3)
		self.conv1_bn = nn.BatchNorm2d(8)
		self.conv2 = nn.Conv2d(8, 8, 3)
		self.conv2_bn = nn.BatchNorm2d(8)   #base complete

		self.conv3 = nn.Conv2d(8, 16, 1, stride=2)
		self.conv3_bn = nn.BatchNorm2d(16)
		self.conv4 = depthwise_separable_conv(16, 16, 3, 1)
		self.conv4_bn = nn.BatchNorm2d(256)
		self.conv5 = depthwise_separable_conv(16, 16, 3, 1)
		self.conv5_bn = nn.BatchNorm2d(16)	#module 1 complete

		self.conv6 = nn.Conv2d(16, 32, 1, stride=2)
		self.conv6_bn = nn.BatchNorm2d(32)
		self.conv7 = depthwise_separable_conv(32, 32, 3, 1)
		self.conv7_bn = nn.BatchNorm2d(32)
		self.conv8 = depthwise_separable_conv(32, 32, 3, 1)
		self.conv8_bn = nn.BatchNorm2d(32)	#module 2 complete

		self.conv9 = nn.Conv2d(32, 64, 1, stride=2)
		self.conv9_bn = nn.BatchNorm2d(64)
		self.conv10 = depthwise_separable_conv(64, 64, 3, 1)
		self.conv10_bn = nn.BatchNorm2d(64)
		self.conv11 = depthwise_separable_conv(64, 64, 3, 1)
		self.conv11_bn = nn.BatchNorm2d(64)		#module 3 complete

		self.conv12 = nn.Conv2d(64, 128, 1, stride=2)
		self.conv12_bn = nn.BatchNorm2d(128)
		self.conv13 = depthwise_separable_conv(128, 128, 3, 1)
		self.conv13_bn = nn.BatchNorm2d(128)
		self.conv14 = depthwise_separable_conv(128, 128, 3, 1)
		self.conv14_bn = nn.BatchNorm2d(128)	#module 4 complete

		self.conv15 = nn.Conv2d(128, num_classes, 3, padding=1)
		self.conv15_bn = nn.BatchNorm2d(num_classes)
		# self.fc1 = nn.Linear(512 * 7 * 7, 120)
		# self.fc2 = nn.Linear(120, 48)
		# self.fc3 = nn.Linear(48, 3)
		self.max_pool = nn.MaxPool2d(3,2, ceil_mode=True)

	def forward(self, x):
		x = F.relu(self.conv1_bn(self.conv1(x)))
		x = F.relu(self.conv2_bn(self.conv2(x))) # base complete

		residual = F.relu(self.conv3_bn(self.conv3(x)))
		x = F.relu(self.conv4_bn(self.conv4(x)))
		x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
		x = x + residual 						# module 1 complete
		
		residual = F.relu(self.conv6_bn(self.conv6(x)))
		x = F.relu(self.conv7_bn(self.conv7(x)))
		x = self.pool(F.relu(self.conv8_bn(self.conv8(x))))
		x = x + residual 						# module 2 complete

		residual = F.relu(self.conv9_bn(self.conv9(x)))
		x = F.relu(self.conv10_bn(self.conv10(x)))
		x = self.pool(F.relu(self.conv11_bn(self.conv11(x))))
		x = x + residual 						# module 3 complete

		residual = F.relu(self.conv12_bn(self.conv12(x)))
		x = F.relu(self.conv13_bn(self.conv13(x)))
		x = self.pool(F.relu(self.conv14_bn(self.conv14(x))))
		x = x + residual 						# module 3 complete

		x = F.relu(self.conv15_bn(self.conv15(x)))
		x = torch.mean(x, (2,3))
		x.view(residual.shape[0], num_classes)
		return x
		