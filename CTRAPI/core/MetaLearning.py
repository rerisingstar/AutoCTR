import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from pruned_search_space import cal_SearchSpace_size

class MetaDataSet(Dataset):
    def __init__(self, data):
        super(MetaDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        dataset_id = int(self.data[index, 0])
        solution_id = int(self.data[index, 1])
        auc_loss = torch.tensor(self.data[index, 2:])
        return dataset_id, solution_id, auc_loss


class MetaPredictor(nn.Module):
    def __init__(self, meta_path, em_size=32, device='cpu'):
        super(MetaPredictor, self).__init__()
        self.device = device
        self.em_size = em_size

        self.read_meta_info(meta_path)
        ds_num = 10
        st_num = cal_SearchSpace_size()
        self.Demb = nn.Embedding(ds_num, em_size).to(device)
        self.Semb = nn.Embedding(st_num, em_size).to(device)

        hidden_size = em_size
        self.dnnD = nn.Sequential(
            # nn.Linear(em_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        self.dnnS = nn.Sequential(
            # nn.Linear(em_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ).to(device)
        self.output = nn.Sequential(
            nn.Linear(hidden_size*2, 2),
            nn.Sigmoid(),
        ).to(device)

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, dataset_id, solution_emb):
        '''
        :param dataset_id: tensor(B)
        :param solution_emb: tensor(B, E)
        :return: predict_auc_loss: (B, 2)
        '''
        # dataset_id, solution_id = torch.chunk(input, 2, dim=1)

        dt_emb = self.Demb(dataset_id) #(B, E)
        # sl_emb = self.Semb(solution_id).squeeze()

        dt_ten = self.dnnD(dt_emb) #(B, H)
        sl_ten = self.dnnS(solution_emb)

        cat_ten = torch.cat((dt_ten, sl_ten), dim=1) #(B, H*2)

        predict_auc_loss = self.output(cat_ten).to(torch.float32) #(B, 2)
        return predict_auc_loss

    def read_meta_info(self, meta_path, batch_size=32):
        # feature_names = ['dataset_id', 'solution_id', 'test_auc', 'test_logloss']

        with open(meta_path, 'r') as f:
            meta_info = json.load(f)
        meta_info = [t for t in meta_info if t[3] < 2]
        data = np.array(meta_info)

        mms = MinMaxScaler(feature_range=(0, 1))
        data[:, 3] = mms.fit_transform(data[:, 3].reshape(-1, 1)).reshape(1, -1)

        self.train_data, self.test_data = train_test_split(data, test_size=0.2, random_state=2022)

        self.train_loader = DataLoader(MetaDataSet(self.train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(MetaDataSet(self.test_data), batch_size=batch_size, shuffle=True)

    def evaluate(self, test_loader):
        with torch.no_grad():
            pred_ans = []
            test_loss = 0
            for _, (di, si, y) in enumerate(test_loader):
                di = di.to(self.device)
                si = si.to(self.device)
                solution_emb = self.Semb(si).squeeze()
                y = y.to(self.device).to(torch.float32)

                y_pred = self(di, solution_emb)

                test_loss += self.loss_func(y_pred, y)

            test_loss /= len(test_loader)
            return test_loss


    def construct(self, epochs=10):
        '''
        :param data_sets: list, ['str']
        :param solutions: list, [int] index of every solutions
        :param aucs: list, test_auc of every (dataset, solution)
        :return:
        '''
        batch_size = 32
        epochs = 10

        steps_per_epoch = len(self.train_data) // batch_size + 1
        print(f"{len(self.train_data)} samples in total, {steps_per_epoch} steps per epoch")
        for epoch in range(epochs):
            totalloss = 0
            with tqdm(enumerate(self.train_loader)) as t:
                for _, (di, si, y) in t:
                    di = di.to(self.device)
                    si = si.to(self.device)
                    solution_emb = self.Semb(si).squeeze()
                    y = y.to(self.device).to(torch.float32)

                    y_pred = self(di, solution_emb)

                    loss = self.loss_func(y_pred, y)
                    totalloss += loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            totalloss /= steps_per_epoch
            test_loss = self.evaluate(self.test_loader)
            print(f"epoch-{epoch+1}, loss={totalloss}, test_loss={test_loss}")


if __name__ == '__main__':
    # model_code = [('linear', 2, 'sigmoid'), 'cat', ('linear', 1, 'sigmoid')]
    meta_predicter = MetaPredictor(device='cuda:6', meta_path='../meta_full.json')
    meta_predicter.construct()

    torch.save(meta_predicter, 'Meta.model')







