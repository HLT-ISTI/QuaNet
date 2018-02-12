import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from torch.autograd import Variable
import numpy as np
from dataset_loader import TextCollectionLoader
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix

dataset = 'reuters21578'
vectorizer = 'tfidf'
#sublinear_tf = True
feat_sel = 1000


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_p=0.1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        # self.fc3 = nn.Linear(int(hidden_size/2), num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout_p = dropout_p

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = F.dropout(out, self.dropout_p, training=self.training)
        out = self.fc2(out)
        # out = F.relu(out)
        # out = F.dropout(out, self.dropout_p, training=self.training)
        # out = self.fc3(out)
        return out

use_cuda = torch.cuda.is_available()
print('CUDA:', use_cuda)

print('loading ' + dataset)
data = TextCollectionLoader(dataset=dataset, vectorizer='tfidf', feat_sel=feat_sel, rep_mode='dense', top_categories=90)


Xtr, ytr = data.get_devel_set()
nD,nF = Xtr.shape
nC = ytr.shape[1]

hidden_size = 512
#batch_size = 100
learning_rate = 0.001
weight_decay = 0.0001
num_steps = 10000
#half = True

net = Net(input_size=nF*nF, hidden_size=hidden_size, num_classes=nC)
if use_cuda:
    net.cuda()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

accuracy = nn.L1Loss()

# Training the Model
def train():
    doc_indexes = np.arange(nD)
    loss_ave = 0
    ave_steps = 10
    net.train(mode=True)
    for i in range(1, num_steps+1):
        np.random.shuffle(doc_indexes)
        n = np.random.randint(int(0.1*nD),nD)
        Xsample = Xtr[doc_indexes[:n]]
        ysample = ytr[doc_indexes[:n]]
        sample_prevalences = np.sum(ysample, axis=0)/n

        X = Variable(torch.from_numpy(Xsample).float())
        y = Variable(torch.from_numpy(sample_prevalences).float())
        if use_cuda:
            X, y = X.cuda(), y.cuda()

        X = torch.mm(X.t(), X).view(1,-1)
        y = y.view(1,-1)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_ave += loss.data[0]
        if i % ave_steps == 0:
            print('Step: [%d/%d], Loss: %.12f' % (i, num_steps, loss_ave / ave_steps))
            loss_ave = 0

        if i % 1000 == 0:
            test()

# Test the Model
def test():
    net.train(mode=False)
    Xte, yte = data.get_test_set()
    nDte = Xte.shape[0]
    true_prevalences = np.sum(yte, axis=0) / nDte

    X = Variable(torch.from_numpy(Xte).float())
    y = Variable(torch.from_numpy(true_prevalences).float())
    if use_cuda:
        X, y = X.cuda(), y.cuda()

    X = torch.mm(X.t(), X).view(1,-1)
    y = y.view(1,-1)

    estimated_prevalences = net(X)
    print('estimated:',estimated_prevalences)
    print('true:',y)

    acc = accuracy(estimated_prevalences, y)
    print('Accuracy %.8f' % acc[0])




def test_svm():
    print('computing svm...')
    svm = GridSearchCV(OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1), refit=True,
                       param_grid={'estimator__C': [1, 10, 100, 1000]})

    Xtr, ytr = data.get_devel_set()
    Xte, yte = data.get_test_set()
    Xtr = csr_matrix(Xtr)
    Xte = csr_matrix(Xte)
    Xtr.sort_indices()
    Xte.sort_indices()
    nDte = Xte.shape[0]
    true_prevalences = np.sum(yte, axis=0) / nDte
    true_prevalences = Variable(torch.from_numpy(true_prevalences).float())

    svm.fit(Xtr, ytr)
    estimated_prevalences = svm.predict(Xte)
    estimated_prevalences = np.sum(estimated_prevalences, axis=0) / nDte

    estimated_prevalences = Variable(torch.from_numpy(estimated_prevalences).float())
    if use_cuda:
        estimated_prevalences = estimated_prevalences.cuda()
        true_prevalences = true_prevalences.cuda()
    acc = accuracy(estimated_prevalences, true_prevalences)
    print(estimated_prevalences)
    print('SVM Accuracy %.8f' % acc[0])

train()
test()
test_svm()



