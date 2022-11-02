
import torch
from torch import nn
from models import SAINT

from data import data_prep, DataSetCatCon, partical_train_data

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, imputations_acc_justy
from augmentations import embedding_mask, embed_data_mask_train

import os
import numpy as np

parser = argparse.ArgumentParser()


parser.add_argument('--dataset', default='qsar_bio', type=str,
                    choices=['1995_income', 'bank_marketing', 'qsar_bio', 'online_shoppers', 'blastchar'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='col', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default=1, type=int)

parser.add_argument('--pretrain_epochs', default=50, type=int)

parser.add_argument('--active_log',default=False,action='store_true')
parser.add_argument('--pretrain', default=True, action='store_true')
parser.add_argument('--pt_tasks', default=['mask_one'], type=str, nargs='*',
                    choices=['mask_multi',"mask_one"])
parser.add_argument('--patial_train', default=50, type=float)
parser.add_argument('--mask_prob', default=0.5, type=float)
parser.add_argument('--forzen', default=False)

parser.add_argument('--ssl_avail_y', default=0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead']) ##
parser.add_argument('--lam1', default=1, type=float)
parser.add_argument('--lam2', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])


opt = parser.parse_args()
torch.manual_seed(opt.set_seed)

# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


device = "cpu"



print(f"Device is {device}.")

modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.dataset, opt.run_name)
os.makedirs(modelsave_path, exist_ok=True)

if opt.active_log:
    import wandb

    if opt.ssl_avail_y > 0 and opt.pretrain:
        wandb.init(project="saint_ssl", group=opt.run_name,
                   name=opt.run_name + '_' + str(opt.attentiontype) + '_' + str(opt.dataset))
    else:
        wandb.init(project="name "+ opt.dataset, group=opt.run_name,
                   name=opt.run_name + '_' + str(opt.attentiontype) + '_' + str(opt.dataset))
    wandb.config.update(opt)


# mask parameters are used to similate missing data scenrio. Set to default 0s otherwise. (pt_mask_params is for pretraining)


pt_mask_params = {
    "method": "pt_mask_params",
    "mask_prob": opt.mask_prob,
    "avail_train_y": 0,
    "test_mask": 0
}


print('Downloading and processing the dataset, it might take some time.')

cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std,train_size = data_prep(
    opt.dataset, opt.set_seed)
if opt.patial_train != 0:
    X_train, y_train,_, _ = partical_train_data(opt.set_seed, X_train, y_train, con_idxs, howmany=opt.patial_train)##0.9219


continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)



train_bsize = opt.batchsize


train_ds = DataSetCatCon(X_train, y_train, cat_idxs, continuous_mean_std, is_pretraining=True)
trainloader = DataLoader(train_ds, batch_size=train_bsize, shuffle=True)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, continuous_mean_std, is_pretraining=True)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs, continuous_mean_std, is_pretraining=True)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False)

# Creating a different dataloader for the pretraining.
if opt.pretrain:
    _, cat_idxs, _, X_train_pt, y_train_pt, _, _, _, _, train_mean, train_std,train_size = data_prep(opt.dataset, opt.set_seed,
                                                                                          pt_mask_params)
    ctd = np.array([train_mean, train_std]).astype(np.float32)

    pt_train_ds = DataSetCatCon(X_train_pt, y_train_pt, cat_idxs, ctd, is_pretraining=True)
    pt_trainloader = DataLoader(pt_train_ds, batch_size=opt.batchsize, shuffle=True)

cat_dims = np.append(np.array(cat_dims), np.array([2])).astype(
    int)  # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification
model = SAINT(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=opt.embedding_size,
    dim_out=1,
    depth=opt.transformer_depth,
    heads=opt.attention_heads,
    attn_dropout=opt.attention_dropout,
    ff_dropout=opt.ff_dropout,
    mlp_hidden_mults=(4, 2),
    continuous_mean_std=continuous_mean_std,
    cont_embeddings=opt.cont_embeddings,
    attentiontype=opt.attentiontype,
    final_mlp_style=opt.final_mlp_style,
    y_dim=2
)

criterion = nn.CrossEntropyLoss().to(device)
model.to(device)




if opt.pretrain:

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    print("Pretraining begins!")
    for epoch in range(opt.pretrain_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(pt_trainloader, 0):
            loss = 0

            optimizer.zero_grad()

            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            _,x_categ_enc_mask, x_cont_enc_mask = embedding_mask(x_categ, x_cont, cat_mask, con_mask, model, opt.pt_tasks)

            cat_outs, con_outs = model(x_categ_enc_mask, x_cont_enc_mask) ## 9,6
            con_outs = torch.cat(con_outs, dim=1)
            l2 = criterion2(con_outs, x_cont)
            l1 = 0
            for j in range(len(cat_dims) - 1):
                l1 += criterion1(cat_outs[j], x_categ[:, j])
            loss += opt.lam1 * l1 + opt.lam2 * l2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch: {epoch}, Running Loss: {running_loss}')
        if opt.active_log:
            wandb.log({'pt_epoch': epoch, 'pretrain_epoch_loss': running_loss
                       })

# torch.save(model.state_dict(),'%s/model.pth' % (modelsave_path))

# model.load_state_dict(torch.load("model.pth"))
# model.eval()

## Only train the Classification layer

if opt.forzen :
    for par in model.parameters():
        par.requires_grad = False
    for par in model.mlpfory.parameters():
        par.requires_grad = True


params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.AdamW(params,lr=opt.lr)



best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0

print('Training begins now.')
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont.
        x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
        # We are converting the data to embeddings in the next step
        _ , x_categ_enc, x_cont_enc = embed_data_mask_train(x_categ, x_cont, cat_mask, con_mask,model)
        reps = model.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:,len(cat_dims)-1,:]
        y_outs = model.mlpfory(y_reps)
        loss = criterion(y_outs,x_categ[:,len(cat_dims)-1])*10
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if opt.active_log:
        wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss,
        'loss': loss.item()
        })

    if epoch%5==0:
            model.eval()
            with torch.no_grad():

                accuracy, auroc = imputations_acc_justy(model, validloader, device)
                test_accuracy, test_auroc = imputations_acc_justy(model, testloader, device)


                print('[EPOCH %d] VALID ACCURACY: %.4f, VALID AUROC: %.4f' %
                    (epoch + 1, accuracy,auroc ))
                print('[EPOCH %d] TEST ACCURACY: %.4f, TEST AUROC: %.4f' %
                    (epoch + 1, test_accuracy,test_auroc ))
                print('[EPOCH %d] RUNNING LOSS: %.4f' % (epoch + 1, running_loss))
                if opt.active_log:
                    wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })
                    wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })
                if auroc > best_valid_auroc:
                    best_valid_auroc = auroc
                    best_test_auroc = test_auroc
                    best_test_accuracy = test_accuracy
                    # torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            model.train()



total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
print('AUROC on best model:  %.4f' %(best_test_auroc))

if opt.active_log:
    wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc ,
    'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })

print( "pretrain="  +("True" if opt.pretrain == 1 else "False") +" patial_train="+str(opt.patial_train) + " mask_prob=" + str(opt.mask_prob))
