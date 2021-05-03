import numpy as np
import torch
from model import GMCF
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import pickle
import time
import math
import pandas as pd


def train(args, data_info, show_loss):
    train_loader = data_info['train']
    val_loader= data_info['val']
    test_loader= data_info['test']
    feature_num = data_info['feature_num']
    train_num, val_num, test_num = data_info['data_num']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    model = GMCF(args, feature_num, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            weight_decay=1e-5, 
            lr=args.lr
    )
    crit = torch.nn.BCELoss()

    print([i.size() for i in filter(lambda p: p.requires_grad, model.parameters())])
    print('start training...')
    for step in range(args.n_epoch):
        # training
        loss_all = 0
        edge_all = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            output = model(data)
            label = data.y
            label = label.to(device)
            baseloss = crit(torch.squeeze(output), label)
            loss = baseloss
            loss_all += data.num_graphs * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cur_loss = loss_all / train_num 

        # evaluation
        val_auc, val_logloss, val_ndcg5, val_ndcg10 = evaluate(model, val_loader, device)
        test_auc, test_logloss, test_ndcg5, test_ndcg10 = evaluate(model, test_loader, device)

        print('Epoch: {:03d}, Loss: {:.5f}, AUC: {:.5f}/{:.5f}, Logloss: {:.5f}/{:.5f}, NDCG@5: {:.5f}/{:.5f} NDCG@10: {:.5f}/{:.5f}'.
          format(step, cur_loss, val_auc, test_auc, val_logloss, test_logloss, val_ndcg5, test_ndcg5, val_ndcg10, test_ndcg10))


def evaluate(model, data_loader, device):
    model.eval()

    predictions = []
    labels = []
    user_ids = []
    edges_all = [0, 0]
    with torch.no_grad():
        for data in data_loader:
            _, user_id_index = np.unique(data.batch.detach().cpu().numpy(), return_index=True)
            user_id = data.x.detach().cpu().numpy()[user_id_index]
            user_ids.append(user_id)

            data = data.to(device)
            pred = model(data)
            pred = pred.squeeze().detach().cpu().numpy().astype('float64')
            if pred.size == 1:
                pred = np.expand_dims(pred, axis=0)
            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)
    user_ids = np.concatenate(user_ids, 0)

    ndcg5 = cal_ndcg(predictions, labels, user_ids, 5)
    ndcg10 = cal_ndcg(predictions, labels, user_ids, 10)
    auc = roc_auc_score(labels, predictions)
    logloss = log_loss(labels, predictions)

    return auc, logloss, ndcg5, ndcg10

def cal_ndcg(predicts, labels, user_ids, k):
    d = {'user': np.squeeze(user_ids), 'predict':np.squeeze(predicts), 'label':np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    ndcg = []
    for user_id in user_unique:
        user_srow = df.loc[df['user'] == user_id]
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            #print('less than 2', user_id)
            continue
        #supred = [upred] if len(upred)>1 else [upred + [-1]]  # prevent error occured if only one sample for a user
        ulabel = user_srow['label'].tolist()
        #sulabel = [ulabel] if len(ulabel)>1 else [ulabel +[1]]

        ndcg.append(ndcg_score([ulabel], [upred], k=k)) 

    return np.mean(np.array(ndcg))

