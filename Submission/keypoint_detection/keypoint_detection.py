
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import utils
import scipy.ndimage as ndimage
from PIL import Image
from torchvision import transforms
from skimage import io, transform
import matplotlib.pyplot as plt
import cv2
import os
from torch import nn, optim
import torch.nn.functional as F
import time
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


device_ids = [0, 1, 2, 3, 4, 5, 6]


# In[ ]:


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# In[ ]:


class Parameters:
    def __init__(self):
        self.image_dir = "../WFLW_images/"
        self.train_annotations = "../WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
        self.test_annotations = "../WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
        self.data_transform = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])


# In[ ]:


def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, market='.', c='g')


# In[ ]:


class keypoint_dataset(Dataset):
    def __init__(self, opt, typ='Train'):
        if typ is 'Train':
            self.keypoints = pd.read_csv(opt.train_annotations, sep=" ")
        else:
            self.keypoints = pd.read_csv(opt.test_annotations, sep=" ")
        self.opt = opt

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        image_name = self.keypoints.iloc[idx, -1]
        image_path = os.path.join(self.opt.image_dir, image_name)
        image = rgb2gray(io.imread(image_path))
        keypoints = self.keypoints.iloc[idx, :196].as_matrix()
        keypoints = keypoints.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': keypoints}
        if self.opt.data_transform is not None:
            sample = self.opt.data_transform(sample)
        return sample


# In[ ]:


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'keypoints': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        if len(image.shape) == 2:
            image = image[:, :, None]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}


# In[ ]:


class Keypoint_model(nn.Module):
    def __init__(self):
        super(Keypoint_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 600)
        self.fc3 = nn.Linear(600, 196)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112
        x = self.pool(F.relu(self.conv2(x)))  # 56
        x = self.pool(F.relu(self.conv3(x)))  # 28
        x = self.pool(F.relu(self.conv4(x)))  # 14
        x = self.pool(F.relu(self.conv5(x)))  # 7

        x = x.view(-1, 256 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# In[ ]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Keypoint_model()
model.to(device)
model = torch.nn.DataParallel(model, device_ids=device_ids)
torch.cuda.set_device(0)


# In[ ]:


parameters = Parameters()
train_dataset = keypoint_dataset(parameters)
test_dataset = keypoint_dataset(parameters)


# In[ ]:


train_dataloader = DataLoader(dataset=train_dataset, batch_size=5, num_workers=0, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=5, num_workers=0, shuffle=True)


# In[ ]:


def early_stopping(val_losses, epoch_threshold=10):
    if epoch_threshold > val_losses:
        return False
    latest_losses = val_losses[-epoch_threshold:]
    if len(set(latest_losses)) == 1:
        return True
    min_loss = min(val_losses)
    if min(latest_losses) < min(val_losses[:len(val_losses) - epoch_threshold]):
        return False
    else:
        return True


# In[ ]:


model_dir = 'saved_models/'
model_name = 'my_best_model.pt'
criterion = nn.MSELoss()
initial_lr = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)


def sqrt_lr_scheduler(optimizer, epoch, init_lr=initial_lr, steps=50):
    # Decay learning rate by square root of the epoch #
    if (epoch % steps) == 0:
        lr = init_lr / np.sqrt(epoch + 1)
        print('Adjusted learning rate: {:.6f}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer


# In[ ]:


def train(n_epochs, train_loader, val_loader, optimizer):
    model.train()
    train_losses, val_losses = [], []
    best_val_loss = float("INF")

    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0
        optimizer = sqrt_lr_scheduler(optimizer, epoch, initial_lr)
        for batch_i, data in enumerate(train_loader):
            # if batch_i > 2:
             #   break

            images = data['image'].to(device)
            keypts = data['keypoints'].to(device)
            keypts = keypts.view(keypts.size(0), -1)
            images, keypts = Variable(images).type(torch.cuda.FloatTensor), Variable(
                keypts).type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            output_pts = model(images)
            loss = criterion(output_pts, keypts)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print('Epoch: {}, Avg. Loss: {}'.format(epoch + 1, avg_train_loss))
        total_val_loss = 0
        for batch_i, data in enumerate(val_loader):
            # if batch_i > 2:
            #    break
            images = data['image'].to(device)
            keypts = data['keypoints'].to(device)
            keypts = keypts.view(keypts.size(0), -1)
            images, keypts = Variable(images).type(torch.cuda.FloatTensor), Variable(
                keypts).type(torch.cuda.FloatTensor)
            output_pts = model(images)
            loss = criterion(output_pts, keypts)
            total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), model_dir + model_name)
            print("val_loss improved from {} to {}, saving model to {}".format(best_val_loss,
                                                                               avg_val_loss,
                                                                               model_name))
            best_val_loss = avg_val_loss
        else:
            print("val_loss did not improve")
            print("took {:.2f}s; loss = {:.2f}; val_loss = {:.2f}".format(time.time() - start_time,
                                                                          avg_train_loss, avg_val_loss))
        if epoch > 100:
            if early_stopping(val_losses, 10):
                break
    print('Finished Training')
    return train_losses, val_losses


# In[ ]:


train(500, train_dataloader, test_dataloader, optimizer)
