import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

model = Keypoint_model()
criterion = nn.MSELoss()
optimizer = nn.Adam(model.parameters())

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
    
model_dir = 'saved_models/'
model_name = 'my_best_model.pt'

def train(n_epochs):
    model.train()
    train_losses, val_losses = [], []
    best_val_loss = float("INF")
    
    for epoch in range(n_epochs):
        print ("Epoch {}/{}".format(epoch + 1, n_epochs))
        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0
        
        for batch_i, data in enumerate(train_loader):
            optimizer.zero_grad()
            images = data['image']
            keypts = data['keypoints']
            keypts = keypts.view(key_pts.size(0), -1)
            images, keypts = Variable(images).type(torch.FloatTensor), Variable(keypts).type(torch.FloatTensor)
            output_pts = model(images)
            loss = criterion(output_pts, key_pts)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.data[0] 
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print('Epoch: {}, Avg. Loss: {}'.format(epoch + 1, avg_train_loss))
        total_val_loss = 0
        for batch_i, data in enumerate(val_loader):
            images = data['image']
            keypts = data['keypoints']
            keypts = keypts.view(key_pts.size(0), -1)
            images, keypts = Variable(images).type(torch.FloatTensor), Variable(keypts).type(torch.FloatTensor)
            output_pts = net(images)
            loss = criterion(output_pts, key_pts)
            total_val_loss += loss.data[0]
        vg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            torch.save(net.state_dict(), model_dir + model_name)
            print ("val_loss improved from {} to {}, saving model to {}".format(best_val_loss, 
                                                                                    avg_val_loss, 
                                                                                    model_name))
            best_val_loss = avg_val_loss
        else:
            print ("val_loss did not improve")
            print ("took {:.2f}s; loss = {:.2f}; val_loss = {:.2f}".format(time.time() - start_time, 
                                                                           avg_train_loss, avg_val_loss))
        if epoch > 100:
            if early_stopping(val_losses, 10):
                break   
    print('Finished Training')
    return train_losses, val_losses
