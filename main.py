import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim 

import pandas as pd


class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.hidden_size = [256, 128, 64, len(labels)]
        self.net_seq = nn.Sequential(
            nn.Linear(num_feature, self.hidden_size[0], bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_size[0], self.hidden_size[1], bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_size[1], self.hidden_size[2], bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_size[2], self.hidden_size[3], bias=True),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.net_seq(input)


if __name__ == '__main__':
    df = pd.read_csv('train.csv')

    num_feature, num_data = 0, len(df['id'])
    labels = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3,}
    for item in df.columns:
        if 'feature' in item:
            num_feature += 1
    total_mat = df.values
    feature_max = 
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)



# sample = df.sample(n=5, axis=0)
# df.as_matrix()
# np.array(df)

# print(sample.loc)
