import torch
import numpy as np
from math import sqrt, sin, cos, pi, asin
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

def earth_distance(lat_lng1, lat_lng2):

    lat1, lng1 = [float(l)*pi/180 for l in lat_lng1]
    lat2, lng2 = [float(l)*pi/180 for l in lat_lng2]

    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds

def instantiate_model(oa2centroid, oa2features, dim_input, device=torch.device("cpu"), dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=False):
    model = NN_MultinomialRegression(dim_input, dim_hidden,  'deepgravity', dropout_p=dropout_p, device=device)
    return model
                
class FlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 list_IDs: List[str],\
                 tileid2oa2features2vals: Dict,\
                 o2d2flow: Dict,\
                 oa2features: Dict,\
                 oa2centroid: Dict,\
                 dim_dests: int,\
                 frac_true_dest: float
                ) -> None:
        'Initialization'
        self.list_IDs = list_IDs
        self.tileid2oa2features2vals = tileid2oa2features2vals
        self.o2d2flow = o2d2flow
        self.oa2features = oa2features
        self.oa2centroid = oa2centroid
        self.dim_dests = dim_dests
        self.frac_true_dest = frac_true_dest
        self.oa2tile = {oa:tile for tile,oa2v in tileid2oa2features2vals.items() for oa in oa2v.keys()}

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get_features(self, oa_origin, oa_destination):            #Putting together the FEATURES of the origin and destination into a 2n+1-dimensional vector
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        dist_od = earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])

        return oa2features[oa_origin] + oa2features[oa_destination] + [dist_od]

    def get_flow(self, oa_origin, oa_destination):                #Get the real mobility flows from li to lj
        o2d2flow = self.o2d2flow
        try:
            return o2d2flow[oa_origin][oa_destination]
        except KeyError:
            return 0

    def get_destinations(self, oa, size_train_dest, all_locs_in_train_region):        #Get other cities in the same "tile" as the current city
        o2d2flow = self.o2d2flow
        frac_true_dest = self.frac_true_dest
        try:
            true_dests_all = list(o2d2flow[oa].keys())
        except KeyError:
            true_dests_all = []
        size_true_dests = min(int(size_train_dest * frac_true_dest), len(true_dests_all))
        size_fake_dests = size_train_dest - size_true_dests

        true_dests = np.random.choice(true_dests_all, size=size_true_dests, replace=False)
        fake_dests_all = list(set(all_locs_in_train_region) - set(true_dests))
        fake_dests = np.random.choice(fake_dests_all, size=size_fake_dests, replace=False)

        dests = np.concatenate((true_dests, fake_dests))
        np.random.shuffle(dests)
        return dests

    def get_X_T(self, origin_locs, dest_locs):

        X, T = [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                X[-1] += [self.get_features(i, j)]
                T[-1] += [self.get_flow(i, j)]

        teX = torch.from_numpy(np.array(X)).float()
        teT = torch.from_numpy(np.array(T)).float()
        return teX, teT

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        tileid2oa2features2vals = self.tileid2oa2features2vals
        dim_dests = self.dim_dests
        oa2tile = self.oa2tile

        # Select sample (tile)
        sampled_origins = [self.list_IDs[index]]
        tile_ID = oa2tile[sampled_origins[0]]

        all_locs_in_train_region = list(tileid2oa2features2vals[tile_ID].keys())
        size_train_dest = min(dim_dests, len(all_locs_in_train_region))
        sampled_dests = [self.get_destinations(oa, size_train_dest, all_locs_in_train_region) for oa in sampled_origins]

        sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests)

        return sampled_trX, sampled_trT, sampled_origins, sampled_dests

    def __getitem_tile__(self, index: int) -> Tuple[Any, Any]:
        'Generates one sample of data (one tile)'

        tileid2oa2features2vals = self.tileid2oa2features2vals
        dim_dests = self.dim_dests
        tile_ID = self.list_IDs[index]
        sampled_origins = list(tileid2oa2features2vals[tile_ID].keys())

        # Select a subset of OD pairs
        train_locs = sampled_origins
        all_locs_in_train_region = train_locs
        size_train_dest = min(dim_dests, len(all_locs_in_train_region))
        sampled_dests = [self.get_destinations(oa, size_train_dest, all_locs_in_train_region)
                         for oa in sampled_origins]

        # get the features and flows
        sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests)

        return sampled_trX, sampled_trT

class GLM_MultinomialRegression(torch.nn.Module):
    def __init__(self, dim_w, device=torch.device("cpu")):
        super(GLM_MultinomialRegression, self).__init__()

        self.device = device
        self.linear1 = torch.nn.Linear(dim_w, 1)

    def forward(self, vX):
        out = self.linear1(vX)
        return out

    def loss(self, out, vT):
        lsm = torch.nn.LogSoftmax(dim=1)
        return -( vT * lsm(torch.squeeze(out, dim=-1)) ).sum()

    def negative_loglikelihood(self, tX, tT):
        return self.loss(self.forward(tX), tT).item()

    def train_one(self, optimizer, tX, tY):
        
        # Reset gradient
        optimizer.zero_grad()

        NlogL = 0.

        num_batches = len(tX)
        for k in range(num_batches):
            if 'cuda' in self.device.type:
                x = Variable(torch.from_numpy(np.array(tX[k])).cuda(), requires_grad=False)
                y = Variable(torch.from_numpy(np.array(tY[k])).cuda(), requires_grad=False)
            else:
                x = Variable(torch.from_numpy(np.array(tX[k])), requires_grad=False)
                y = Variable(torch.from_numpy(np.array(tY[k])), requires_grad=False)

            # Forward
            fx = self.forward(x)
            NlogL += self.loss(fx, y)

        # Backward
        NlogL.backward()

        # Update parameters
        optimizer.step()

        return NlogL.item()  #NlogL.data[0]

    def predict_proba(self, x):
        sm = torch.nn.Softmax(dim=1)
        #probs = sm(torch.squeeze(self.forward(x), dim=2))
        probs = sm(torch.squeeze(self.forward(x), dim=-1))
        if 'cuda' in self.device.type:
            return probs.cpu().detach().numpy()
        else:
            return probs.detach().numpy()

    def average_OD_model(self, tX, tT):
        p = self.predict_proba(tX)
        if 'cuda' in self.device.type:
            #tot_out_trips = tT.sum(dim=1).cpu().detach().numpy()
            tot_out_trips = tT.sum(dim=-1).cpu().detach().numpy()
        else:
            #tot_out_trips = tT.sum(dim=1).detach().numpy()
            tot_out_trips = tT.sum(dim=-1).detach().numpy()
        model_od = (p.T * tot_out_trips).T
        return model_od

    def predict(self, x_val):
        x = Variable(x_val, requires_grad=False)
        output = self.forward(x)
        #return output.data.numpy().argmax(axis=1)
        return output.data.argmax(axis=1)
    
def common_part_of_commuters(values1, values2, numerator_only=False):
    if numerator_only:
        tot = 1.0
    else:
        tot = (np.sum(values2) + np.sum(values2))
    if tot > 0:
        return 2.0 * np.sum(np.minimum(values1, values2)) / tot
    else:
        return 0.0
    
class NN_OriginalGravity(GLM_MultinomialRegression):

    def __init__(self, dim_input, df='exponential', device=torch.device("cpu")):

        super(GLM_MultinomialRegression, self).__init__()
        #         super().__init__(self)
        self.device = device
        self.df = df
        self.dim_input = dim_input
        self.linear_out = torch.nn.Linear(dim_input, 1)

    def forward(self, vX):
        out = self.linear_out(vX)
        return out

    def get_features(self, oa_origin, oa_destination, oa2features, oa2centroid, df):
        return get_features_original_gravity(oa_origin, oa_destination, oa2features, oa2centroid, df=df)

    def get_X_T(self, origin_locs, dest_locs, oa2features, oa2centroid, o2d2flow, verbose=False):
       
        X, T = [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                X[-1] += [self.get_features(i, j, oa2features, oa2centroid, self.df)]
              
                T[-1] += [get_flow(i, j, o2d2flow)]

        if 'cuda' in self.device.type:
            teX = torch.from_numpy(np.array(X)).float().cuda()
            teT = torch.from_numpy(np.array(T)).float().cuda()
        else:
            teX = torch.from_numpy(np.array(X)).float()
            teT = torch.from_numpy(np.array(T)).float()

        return teX, teT

    def get_cpc(self, teX, teT, numerator_only=False):
        if 'cuda' in self.device.type:
            flatten_test_observed = teT.cpu().detach().numpy().flatten()
        else:
            flatten_test_observed = teT.detach().numpy().flatten()
        model_OD_test = self.average_OD_model(teX, teT)
        flatten_test_model = model_OD_test.flatten()
        cpc_test = common_part_of_commuters(flatten_test_observed, flatten_test_model, 
                                            numerator_only=numerator_only)
        return cpc_test
    
class NN_OriginalGravity(GLM_MultinomialRegression):

    def __init__(self, dim_input, df='exponential', device=torch.device("cpu")):

        super(GLM_MultinomialRegression, self).__init__()
        #         super().__init__(self)
        self.device = device
        self.df = df
        self.dim_input = dim_input
        self.linear_out = torch.nn.Linear(dim_input, 1)

    def forward(self, vX):
        out = self.linear_out(vX)
        return out

    def get_features(self, oa_origin, oa_destination, oa2features, oa2centroid, df):
        return get_features_original_gravity(oa_origin, oa_destination, oa2features, oa2centroid, df=df)

    def get_X_T(self, origin_locs, dest_locs, oa2features, oa2centroid, o2d2flow, verbose=False):
       
        X, T = [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                X[-1] += [self.get_features(i, j, oa2features, oa2centroid, self.df)]
              
                T[-1] += [get_flow(i, j, o2d2flow)]

        if 'cuda' in self.device.type:
            teX = torch.from_numpy(np.array(X)).float().cuda()
            teT = torch.from_numpy(np.array(T)).float().cuda()
        else:
            teX = torch.from_numpy(np.array(X)).float()
            teT = torch.from_numpy(np.array(T)).float()

        return teX, teT

    def get_cpc(self, teX, teT, numerator_only=False):
        if 'cuda' in self.device.type:
            flatten_test_observed = teT.cpu().detach().numpy().flatten()
        else:
            flatten_test_observed = teT.detach().numpy().flatten()
        model_OD_test = self.average_OD_model(teX, teT)
        flatten_test_model = model_OD_test.flatten()
        cpc_test = common_part_of_commuters(flatten_test_observed, flatten_test_model, numerator_only=numerator_only)
        return cpc_test

class NN_MultinomialRegression(NN_OriginalGravity):

    def __init__(self, dim_input, dim_hidden, df, dropout_p=0.35,  device=torch.device("cpu")):

        super(NN_OriginalGravity, self).__init__(dim_input, device=device)

        self.df = df

        self.device = device

        p = dropout_p

        self.linear1 = torch.nn.Linear(dim_input, dim_hidden)
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(p)

        self.linear2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(p)

        self.linear3 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(p)

        self.linear4 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu4 = torch.nn.LeakyReLU()
        self.dropout4 = torch.nn.Dropout(p)

        self.linear5 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu5 = torch.nn.LeakyReLU()
        self.dropout5 = torch.nn.Dropout(p)

        self.linear6 = torch.nn.Linear(dim_hidden, dim_hidden // 2)
        self.relu6 = torch.nn.LeakyReLU()
        self.dropout6 = torch.nn.Dropout(p)

        self.linear7 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu7 = torch.nn.LeakyReLU()
        self.dropout7 = torch.nn.Dropout(p)

        self.linear8 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu8 = torch.nn.LeakyReLU()
        self.dropout8 = torch.nn.Dropout(p)

        self.linear9 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu9 = torch.nn.LeakyReLU()
        self.dropout9 = torch.nn.Dropout(p)

        self.linear10 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu10 = torch.nn.LeakyReLU()
        self.dropout10 = torch.nn.Dropout(p)

        self.linear11 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu11 = torch.nn.LeakyReLU()
        self.dropout11 = torch.nn.Dropout(p)

        self.linear12 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu12 = torch.nn.LeakyReLU()
        self.dropout12 = torch.nn.Dropout(p)

        self.linear13 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu13 = torch.nn.LeakyReLU()
        self.dropout13 = torch.nn.Dropout(p)

        self.linear14 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu14 = torch.nn.LeakyReLU()
        self.dropout14 = torch.nn.Dropout(p)

        self.linear15 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu15 = torch.nn.LeakyReLU()
        self.dropout15 = torch.nn.Dropout(p)

        self.linear_out = torch.nn.Linear(dim_hidden // 2, 1)

    def forward(self, vX):
        lin1 = self.linear1(vX)
        h_relu1 = self.relu1(lin1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        h_relu2 = self.relu2(lin2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        h_relu3 = self.relu3(lin3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        h_relu4 = self.relu4(lin4)
        drop4 = self.dropout4(h_relu4)

        lin5 = self.linear5(drop4)
        h_relu5 = self.relu5(lin5)
        drop5 = self.dropout5(h_relu5)

        lin6 = self.linear6(drop5)
        h_relu6 = self.relu6(lin6)
        drop6 = self.dropout6(h_relu6)

        lin7 = self.linear7(drop6)
        h_relu7 = self.relu7(lin7)
        drop7 = self.dropout7(h_relu7)

        lin8 = self.linear8(drop7)
        h_relu8 = self.relu8(lin8)
        drop8 = self.dropout8(h_relu8)

        lin9 = self.linear9(drop8)
        h_relu9 = self.relu9(lin9)
        drop9 = self.dropout9(h_relu9)

        lin10 = self.linear10(drop9)
        h_relu10 = self.relu10(lin10)
        drop10 = self.dropout10(h_relu10)

        lin11 = self.linear11(drop10)
        h_relu11 = self.relu11(lin11)
        drop11 = self.dropout11(h_relu11)

        lin12 = self.linear12(drop11)
        h_relu12 = self.relu12(lin12)
        drop12 = self.dropout12(h_relu12)

        lin13 = self.linear13(drop12)
        h_relu13 = self.relu13(lin13)
        drop13 = self.dropout13(h_relu13)

        lin14 = self.linear14(drop13)
        h_relu14 = self.relu14(lin14)
        drop14 = self.dropout14(h_relu14)

        lin15 = self.linear15(drop14)
        h_relu15 = self.relu15(lin15)
        drop15 = self.dropout15(h_relu15)

        out = self.linear_out(drop15)
        return out
