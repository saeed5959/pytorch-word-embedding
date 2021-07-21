import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# making a model class for embedding
class cbow(nn.Module):
    def __init__(self, voc_size, embed_size):
        super(cbow, self).__init__()
        self.embed_size = embed_size
        self.voc_size = voc_size

        self.embedlayer = nn.Embedding(self.voc_size, self.embed_size)
        self.linearlayer = nn.Linear(self.embed_size, self.voc_size)

    def forward(self, x):
        input_embed = self.embedlayer(x)
        input_embed_sum = torch.sum(input_embed, dim=1)
        out = self.linearlayer(input_embed_sum)
        out = F.log_softmax(out, dim=1)
        return out

    def make_embed(self, x):
        x_embed = self.embedlayer(x)
        return x_embed

# making a dataset class : 1) context : contain 2 words before target word and 2 words after target word
#                          2) target : contain target word
class dataset_word(Dataset):
    def __init__(self, data_path, window):
        with open(data_path, 'r') as f:
            self.data = f.read().replace("\n", " ").split(" ")[:-1]

        self.word_to_index = {word: index for index, word in enumerate(self.data)}

        self.context = []
        self.target = []
        for i in range(window, len(self.data) - window):
            self.context.append([self.data[i - window], self.data[i - window + 1], self.data[i - window + 3],
                                 self.data[i - window + 4]])
            self.target.append(self.data[i - window + 2])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        context = [self.word_to_index[word] for word in self.context[item]]
        target = [self.word_to_index[self.target[item]]]
        return torch.tensor(context) , torch.tensor(target)

#path of your text
input_path = "/home/saeed/text.txt"
data = dataset_word(data_path=input_path,window=2)
# model
model = cbow(20,8)
loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-10)

epochs = 5
for epoch in range(1,epochs):
    print("epoch "+str(epoch)+"**")
    dataloader = DataLoader(data, batch_size=2, shuffle=True)
    for num, (context, target) in enumerate(dataloader):
        out = model(context)
        loss_out = loss(out, target.view(2))

        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()

        print("loss : " + str(loss_out.item()))