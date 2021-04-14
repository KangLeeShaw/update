import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models

import os
import numpy as np
from sklearn import metrics
from tqdm import trange, tqdm

import matplotlib.pyplot as plt
import nibabel as nib

import utilities as UT
from PIL import Image
from diversebranchblock import *

def prep_data(LABEL_PATH ,TEST_NUM):
    # This function is used to prepare train/test labels for 5-fold cross-validation
    TEST_LABEL = LABEL_PATH + '/fold_' + str(TEST_NUM) +'.csv'

    # combine train labels
    filenames = [LABEL_PATH + '/fold_0.csv',
           LABEL_PATH + '/fold_1.csv',
           LABEL_PATH + '/fold_2.csv',
           LABEL_PATH + '/fold_3.csv',
           LABEL_PATH + '/fold_4.csv', ]

    filenames.remove(TEST_LABEL)

    with open(LABEL_PATH + '/combined_train_list.csv', 'w') as combined_train_list:
        for fold in filenames:
            for line in open(fold, 'r'):
                combined_train_list.write(line)
    TRAIN_LABEL = LABEL_PATH + '/combined_train_list.csv'

    return TRAIN_LABEL, TEST_LABEL

class att(nn.Module):
    def __init__(self, input_channel):
        "the soft attention module"
        super(att,self).__init__()

        self.channel_in = input_channel

        self.convem1 = nn.Sequential(
            nn.Conv2d(
            in_channels=input_channel,
            out_channels=512,
            kernel_size=1),
            nn.ReLU()
            )
        self.convemx = nn.Sequential(
           nn.MaxPool2d(kernel_size=2,stride=2)
            )

        self.convem2 = nn.Sequential(
            nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=1),
            nn.ReLU()
            )
        self.convem3 = nn.Sequential(
            nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=1),
            nn.ReLU()
            )



        #vgg11 vgg16 mobile 用scale=2 vgg19 用size(6,6) Res18 34 用size（7，7）

        self.emup=nn.Sequential(
            nn.Upsample(scale_factor=2,
             mode='nearest',
             align_corners=None)
            )
        self.dbb1 = DiverseBranchBlock(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1,groups=1,deploy=False)
        self.dbb2 = DiverseBranchBlock(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1,groups=1,deploy=False)
        self.dbb3 = DiverseBranchBlock(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,groups=1,deploy=False)
        self.conv1 =nn.Sequential(
            nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1),
            nn.Softmax(dim=2)
            )


        '''
        self.emup=nn.Sequential(
            nn.Upsample(size=(7,7),
             mode='nearest',
             align_corners=None)
            )

        self.convem4 = nn.Sequential(
            nn.Conv2d(
            in_channels=512,
            out_channels=input_channel,
            kernel_size=1),
            nn.ReLU()
            )



        self.conv1a = nn.Sequential(
            nn.Conv2d(
            in_channels=input_channel,
            out_channels=512,
            kernel_size=3),
            nn.ReLU()
            )
        self.up1=nn.Sequential(
            nn.Upsample(size=(6,6),
             mode='nearest',
             align_corners=None)
            )
        self.conv1b = nn.Sequential(
            nn.Conv2d(
            in_channels=input_channel,
            out_channels=512,
            kernel_size=1),
            nn.ReLU()
            )
        self.conv2a = nn.Sequential(
            nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=3),
            nn.ReLU()
            )
        self.up2=nn.Sequential(
            nn.Upsample(size=(6,6),
             mode='nearest',
             align_corners=None)
            )
        self.conv2b = nn.Sequential(
            nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=1),
            nn.ReLU()
            )
        self.conv3a =nn.Sequential(
            nn.Conv2d(
            in_channels=1536,
            out_channels=64,
            kernel_size=3),
            nn.ReLU()
            )
        self.up3=nn.Sequential(
            nn.Upsample(size=(6,6),
             mode='nearest',
             align_corners=None)
            )
        self.conv3b =nn.Sequential(
            nn.Conv2d(
            in_channels=1536,
            out_channels=64,
            kernel_size=1),
            nn.ReLU()
            )
        self.conv4 =nn.Sequential(
            nn.Conv2d(
            in_channels=1664,
            out_channels=1,
            kernel_size=1),
            nn.Softmax(dim=2)
            )
        '''
    def forward(self, x):

        '''
        mask = x
        #print(mask.shape)
        mask = self.convem1(mask)
        #print(mask.shape)
        mask = self.convemx(mask)
        #print(mask.shape)
        mask = self.convem2(mask)
        #print(mask.shape)
        mask = self.convem3(mask)
        #print(mask.shape)
        mask = self.emup(mask)
        #print(mask.shape)
        mask11 = self.convem4(mask)
        #print(mask11.shape)
        mask1a = self.conv1a(mask11)
        mask1a=self.up1(mask1a)
        #print(mask1a.shape)
        mask1b = self.conv1b(mask11)
        #print(mask1b.shape)
        mask = torch.cat([mask1a,mask1b],axis=1)
        mask2a = self.conv2a(mask)
        mask2a=self.up2(mask2a)
        mask2b = self.conv2b(mask)
        mask = torch.cat([mask,mask2a,mask2b],axis=1)
        mask3a = self.conv3a(mask)
        mask3a=self.up3(mask3a)
        mask3b = self.conv3b(mask)
        mask = torch.cat([mask,mask3a,mask3b],axis=1)
        att = self.conv4(mask)
        #print(att.shape)
        '''
        mask = x
        #print(mask.shape)
        mask = self.convem1(mask)
        #print(mask.shape)
        mask = self.convemx(mask)
        #print(mask.shape)
        mask = self.convem2(mask)
        #print(mask.shape)
        mask = self.convem3(mask)
        #print(mask.shape)
        mask = self.emup(mask)
        mask=self.dbb1(mask)
        mask=self.dbb2(mask)
        mask=self.dbb3(mask)
        att=self.conv1(mask)
        #print(att.shape)

        output = torch.mul(mask, att)
        #print(output.shape)
        return output





class CNN(nn.Module):
    def __init__(self,
                 num_classes=2,
                 feature='Vgg11',
                 feature_shape=(512,7,7),
                 pretrained=True,
                 requires_grad=False):

        super(CNN, self).__init__()

        # Feature Extraction
        if(feature=='Alex'):
            self.ft_ext = models.alexnet(pretrained=pretrained)
            self.ft_ext_modules = list(list(self.ft_ext.children())[:-2][0][:9])

        elif(feature=='Res34'):
            self.ft_ext = models.resnet34(pretrained=pretrained)
            self.ft_ext_modules=list(self.ft_ext.children())[0:3]+list(self.ft_ext.children())[4:-2] # remove the Maxpooling layer

        elif(feature=='Res18'):
            self.ft_ext = models.resnet18(pretrained=pretrained)
            self.ft_ext_modules=list(self.ft_ext.children())[0:3]+list(self.ft_ext.children())[4:-2] # remove the Maxpooling layer

        elif(feature=='Vgg16'):
            self.ft_ext = models.vgg16(pretrained=pretrained)
            self.ft_ext_modules=list(self.ft_ext.children())[0][:30] # remove the Maxpooling layer
        elif(feature=='Vgg19'):
            self.ft_ext = models.vgg16(pretrained=pretrained)
            self.ft_ext_modules=list(self.ft_ext.children())[0][:36] # remove the Maxpooling layer
        elif(feature=='Vgg11'):
            self.ft_ext = models.vgg11(pretrained=pretrained)
            self.ft_ext_modules=list(self.ft_ext.children())[0][:19] # remove the Maxpooling layer

        elif(feature=='Mobile'):
            self.ft_ext = models.mobilenet_v2(pretrained=pretrained)
            self.ft_ext_modules=list(self.ft_ext.children())[0] # remove the Maxpooling layer

        self.ft_ext=nn.Sequential(*self.ft_ext_modules)
        for p in self.ft_ext.parameters():
            p.requires_grad = requires_grad

        # Classifier
        if(feature=='Alex'):
            feature_shape=(256,5,5)
        elif(feature=='Res34'):
            feature_shape=(512,7,7)
        elif(feature=='Res18'):
            feature_shape=(512,7,7)
        elif(feature=='Vgg16'):
            feature_shape=(512,6,6)
        elif(feature=='Vgg19'):
            feature_shape=(512,6,6)
        elif(feature=='Vgg11'):
            feature_shape=(512,6,6)
        elif(feature=='Mobile'):
            feature_shape=(1280,4,4)

        conv1_output_features = int(feature_shape[0])#vgg16 64

        fc1_input_features = int(conv1_output_features*feature_shape[1]*feature_shape[2])
        fc1_output_features = int(conv1_output_features*2)
        fc2_output_features = int(fc1_output_features/4)



        self.attn=att(conv1_output_features)
        self.flat = nn.Sequential(
            nn.Flatten(),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0],
                out_channels=conv1_output_features,
                kernel_size=1,
            ),
            nn.BatchNorm2d(conv1_output_features),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
             nn.Linear(2304,1024),
             nn.BatchNorm1d(fc1_output_features),
             nn.ReLU()
         )

        self.fc2 = nn.Sequential(
             nn.Linear(1024,256),
             nn.BatchNorm1d(fc2_output_features),
             nn.ReLU()
         )

        self.out = nn.Linear(fc2_output_features, num_classes)

    def forward(self, x, drop_prob=0.5):
        x = self.ft_ext(x)
        #print(x.size())

        x= self.attn(x)
        #print(x.shape)
        x=self.flat(x)
        #x = self.conv1(x)
        #print(x.shape)
        #x = x.view(-1,x.size(0))
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)
        x = nn.Dropout(drop_prob)(x)
        prob = self.out(x)

        return prob
def train(train_dataloader, val_dataloader, feature='Vgg11'):
    net = CNN(feature=feature).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)
#     opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma= 0.985)
#     scheduler = torch.optim.lr_scheduler.CyclicLR(opt,
#                                                   base_lr=LR,
#                                                   max_lr=0.001,
#                                                   step_size_up=100,
#                                                   cycle_momentum=False)
    loss_fcn = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHTS.to(device))

    t = trange(EPOCHS, desc=' ', leave=True)

    train_hist = []
    val_hist = []
    pred_result = []
    old_acc = 0
    old_auc = 0
    test_acc = 0
    best_epoch = 0
    test_performance = []
    for e in t:
        y_true = []
        y_pred = []

        val_y_true = []
        val_y_pred = []

        train_loss = 0
        val_loss = 0

        # training
        net.train()
        for step, (img, label, _) in enumerate(train_dataloader):
            img = img.float().to(device)
            label = label.long().to(device)
            opt.zero_grad()
            out = net(img)
            loss = loss_fcn(out, label)

            loss.backward()
            opt.step()

            label = label.cpu().detach()
            out = out.cpu().detach()
            y_true, y_pred = UT.assemble_labels(step, y_true, y_pred, label, out)

            train_loss += loss.item()

        train_loss = train_loss/(step+1)
        acc = float(torch.sum(torch.max(y_pred, 1)[1]==y_true))/ float(len(y_pred))
        auc = metrics.roc_auc_score(y_true, y_pred[:,1])
        f1 = metrics.f1_score(y_true, torch.max(y_pred, 1)[1])
        precision = metrics.precision_score(y_true, torch.max(y_pred, 1)[1])
        recall = metrics.recall_score(y_true, torch.max(y_pred, 1)[1])
        ap = metrics.average_precision_score(y_true, torch.max(y_pred, 1)[1]) #average_precision

        scheduler.step()

        # val
        net.eval()
        full_path = []
        with torch.no_grad():
            for step, (img, label, _) in enumerate(val_dataloader):
                img = img.float().to(device)
                label = label.long().to(device)
                out = net(img)
                loss = loss_fcn(out, label)
                val_loss += loss.item()

                label = label.cpu().detach()
                out = out.cpu().detach()
                val_y_true, val_y_pred = UT.assemble_labels(step, val_y_true, val_y_pred, label, out)

                for item in _:
                    full_path.append(item)

        val_loss = val_loss/(step+1)
        val_acc = float(torch.sum(torch.max(val_y_pred, 1)[1]==val_y_true))/ float(len(val_y_pred))
        val_auc = metrics.roc_auc_score(val_y_true, val_y_pred[:,1])
        val_f1 = metrics.f1_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_precision = metrics.precision_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_recall = metrics.recall_score(val_y_true, torch.max(val_y_pred, 1)[1])
        val_ap = metrics.average_precision_score(val_y_true, torch.max(val_y_pred, 1)[1]) #average_precision


        train_hist.append([train_loss, acc, auc, f1, precision, recall, ap])
        val_hist.append([val_loss, val_acc, val_auc, val_f1, val_precision, val_recall, val_ap])

        t.set_description("Epoch: %i, train loss: %.4f, train acc: %.4f, val loss: %.4f, val acc: %.4f, test acc: %.4f"
                          %(e, train_loss, acc, val_loss, val_acc, test_acc))


        if(old_acc<val_acc):
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true))/ float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1]) #average_precision

            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall, test_ap]

        if(old_acc==val_acc) and (old_auc<val_auc):
            old_acc = val_acc
            old_auc = val_auc
            best_epoch = e
            test_loss = 0
            test_y_true = val_y_true
            test_y_pred = val_y_pred

            test_loss = val_loss
            test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true))/ float(len(test_y_pred))
            test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
            test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
            test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1]) #average_precision

            test_performance = [best_epoch, test_loss, test_acc, test_auc, test_f1, test_precision, test_recall, test_ap]
    return train_hist, val_hist, test_performance, test_y_true, test_y_pred, full_pa
LABEL_PATH = './data'

GPU = 0
BATCH_SIZE = 16
EPOCHS = 150

LR = 0.0001
LOSS_WEIGHTS = torch.tensor([1., 1.])
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:'+str(GPU) if torch.cuda.is_available() else 'cpu')

 #DATA_PATH = '/data/scratch/gliang/data/adni/ADNI2_MRI_Feature/Alex_Layer-9_DynamicImage'
#FEATURE_SHAPE=(256,5,5)
#print('DATA_PATH:',DATA_PATH)

train_hist = []
val_hist = []
test_performance = []
test_y_true = np.asarray([])
test_y_pred = np.asarray([])
full_path = np.asarray([])
for i in range(0, 5):
    print('Train Fold', i)

    TEST_NUM = i
    TRAIN_LABEL, TEST_LABEL = prep_data(LABEL_PATH, TEST_NUM)

    train_dataset = Dataset_Early_Fusion(label_file=TRAIN_LABEL)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_dataset = Dataset_Early_Fusion(label_file=TEST_LABEL)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    cur_result = train(train_dataloader, val_dataloader)

    train_hist.append(cur_result[0])
    val_hist.append(cur_result[1])
    test_performance.append(cur_result[2])
    test_y_true = np.concatenate((test_y_true, cur_result[3].numpy()))
    if(len(test_y_pred) == 0):
        test_y_pred = cur_result[4].numpy()
    else:
        test_y_pred = np.vstack((test_y_pred, cur_result[4].numpy()))
    full_path = np.concatenate((full_path, np.asarray(cur_result[5])))

print(test_performance)

test_y_true = torch.tensor(test_y_true)
test_y_pred = torch.tensor(test_y_pred)
test_acc = float(torch.sum(torch.max(test_y_pred, 1)[1]==test_y_true.long()))/ float(len(test_y_pred))
test_auc = metrics.roc_auc_score(test_y_true, test_y_pred[:,1])
test_f1 = metrics.f1_score(test_y_true, torch.max(test_y_pred, 1)[1])
test_precision = metrics.precision_score(test_y_true, torch.max(test_y_pred, 1)[1])
test_recall = metrics.recall_score(test_y_true, torch.max(test_y_pred, 1)[1])
test_ap = metrics.average_precision_score(test_y_true, torch.max(test_y_pred, 1)[1])

print('ACC %.4f, AUC %.4f, F1 %.4f, Prec %.4f, Recall %.4f, AP %.4f'
      %(test_acc, test_auc, test_f1, test_precision, test_recall, test_ap))
