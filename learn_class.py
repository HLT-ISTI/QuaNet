import argparse
from time import time

from quantification.helpers import *
from quantification.nets.classification import LSTMTextClassificationNet
from util.helpers import *


def main(args):
    args = parseargs(args)
    if '.[data].' in args.output:
        args.output = args.output.replace('.[data].','.'+args.data+'.')

    create_if_not_exists(args.output)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loadDataset(dataset=args.data, vocabularysize=args.vocabularysize)
    print('x_train shape:', x_train.shape, 'prev', np.mean(y_train))
    print('x_val shape:', x_val.shape, 'prev', np.mean(y_val))
    print('x_test shape:', x_test.shape, 'prev', np.mean(y_test))

    class_lstm_layers = 1
    classes = 2

    status_every = 20
    test_every = 100

    class_net = LSTMTextClassificationNet(args.vocabularysize, args.embeddingsize, classes, args.hiddensize,
                                          class_lstm_layers, args.linlayers, args.dropout)
    class_net = class_net.cuda() if use_cuda else class_net
    print(class_net)

    criterion = torch.nn.MSELoss()
    optimizier = torch.optim.Adam(class_net.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    x_train_pos, x_train_neg, _,_ = split_pos_neg(x_train, y_train)

    x_val, y_val = prepare_classification(x_val, y_val)
    x_test, y_test = prepare_classification(x_test, y_test)
    best_val_accuracy = -1

    loss_sum, accuracy_sum = 0, 0
    t_init = time()
    patience = PATIENCE
    for step in range(1, args.maxiter + 1):
        class_net.train()

        prevalence = np.random.rand()

        x, y, _ = sample_data(x_train_pos, x_train_neg, prevalence, args.batchsize)
        x, y = prepare_classification(x, y)

        optimizier.zero_grad()
        yhat = class_net.forward(x)
        class_loss = criterion(yhat, y)
        class_loss.backward()
        optimizier.step()

        loss_sum += class_loss.data[0]
        accuracy_sum += accuracy(y, yhat)

        if step % status_every == 0:
            print('step {}\tloss {:.5f}\t accuracy {:.5f}\t v {:.2f} steps/s'
                  .format(step, loss_sum / status_every, accuracy_sum / status_every, status_every/(time()-t_init)))
            loss_sum, accuracy_sum = 0, 0
            t_init = time()

        if step % test_every == 0:
            class_net.eval()
            y_hat = class_batched_predictions(class_net, x_val)
            accuracy_val = accuracy(y_val, y_hat)

            y_hat = class_batched_predictions(class_net, x_test)
            accuracy_test = accuracy(y_test, y_hat)
            print('ValAcc {:.5f}\tTestAcc {:.5f}[patience {}]'.format(accuracy_val, accuracy_test, patience))

            if accuracy_val > best_val_accuracy:
                print('\tsaving model to', args.output)
                torch.save(class_net, args.output)
                best_val_accuracy=accuracy_val
                patience = PATIENCE
            else:
                patience-=1
                if patience==0:
                    print('Early stop after {} loss checks without improvement'.format(PATIENCE))
                    break

def parseargs(args):
    parser = argparse.ArgumentParser(description='Learn Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data',
                        help='Name of the dataset. Valid ones are: imdb, hp, kindle')
    parser.add_argument('-v', '--vocabularysize',
                        help='Maximum length of the vocabulary', type=int, default=5000)
    parser.add_argument('-e', '--embeddingsize',
                        help='Size of the word embeddings', type=int, default=100)
    parser.add_argument('-H', '--hiddensize',
                        help='Size of the LSTM hidden layers', type=int, default=128)
    parser.add_argument('-d', '--dropout',
                        help='Drop probability for dropout', type=float, default=0.5)
    parser.add_argument('-l', '--linlayers',
                        help='Linear layers on top of the LSTM output', type=int, default=[1024, 100], nargs='+')
    parser.add_argument('-I', '--maxiter',
                        help='Maximum number of iterations', type=int, default=20000)
    parser.add_argument('-O', '--output',
                        help='Path to the output file containing the model parameters', type=str,
                        default='./class_net.[data].pt')
    parser.add_argument('--lr',
                        help='learning rate', type=float, default=0.0001)
    parser.add_argument('--weightdecay',
                        help='weight decay', type=float, default=1e-4)
    parser.add_argument('--batchsize',
                        help='batch size', type=float, default=100)

    return parser.parse_args(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
