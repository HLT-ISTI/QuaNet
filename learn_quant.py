import argparse
from time import time
from plot_correction import plot_corr, plot_loss
from quantification.helpers import *
from nets.quantification import LSTMQuantificationNet
from util.helpers import *
# from inntt import *


def eval_metric(metric, prevs, *methods):
    return [metric(prevs, method_i) for method_i in methods]


def main(args):
    print(args)
    args = parseargs(args)
    if '.[data].' in args.output:
        args.output = args.output.replace('.[data].','.'+args.data+'.')
    args.plotdir = os.path.join(args.plotdir, args.data)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loadDataset(dataset=args.data, vocabularysize=args.vocabularysize)
    print('x_train shape:', x_train.shape, 'prev', np.mean(y_train))
    print('x_val shape:', x_val.shape, 'prev', np.mean(y_val))
    print('x_test shape:', x_test.shape, 'prev', np.mean(y_test))

    class_net = torch.load(args.model)
    class_net = class_net.cuda() if use_cuda else class_net

    print('creating val_yhat and test_yhat')
    val_yhat = predict(class_net, x_val, args.use_embeddings)
    test_yhat = predict(class_net, x_test, args.use_embeddings)
    val_tpr = tpr(val_yhat, y_val)
    val_fpr = fpr(val_yhat, y_val)
    val_ptpr = ptpr(val_yhat, y_val)
    val_pfpr = pfpr(val_yhat, y_val)

    quant_lstm_layers = 1
    classes = 2
    input_size = classes if not args.use_embeddings else classes + 100 #todo: take from class-model

    quant_net = LSTMQuantificationNet(input_size, args.hiddensize, quant_lstm_layers,
                                      args.linlayers, drop_p=args.dropout,
                                      stats_in_lin_layers=args.stats_layer,
                                      stats_in_sequence=args.stats_lstm)

    quant_net = quant_net.cuda() if use_cuda else quant_net
    print(quant_net)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(quant_net.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    train_prev = np.mean(y_train)

    test_samples = (21 if args.include_bounds else 19) * 100 #todo: clarify
    prevs_range = define_prev_range(args.include_bounds)

    val_yhat_pos, val_yhat_neg, val_pos_ids, val_neg_ids = split_pos_neg(val_yhat, y_val)
    test_yhat_pos, test_yhat_neg, test_pos_ids, test_neg_ids = split_pos_neg(test_yhat, y_test)

    test_sample_yhat, test_sample_y, test_sample_prev, test_sample_stats, _ = \
        quantification_uniform_sampling(test_pos_ids, test_neg_ids, test_yhat_pos, test_yhat_neg,
                                        val_tpr, val_fpr, val_ptpr, val_pfpr,
                                        input_size, test_samples, args.samplelength, prevs_range=prevs_range)

    true_prevs = compute_true_prevalence(test_sample_prev)

    print('Computing classify & count based methods (cc, acc, pcc, apcc) for {} samples of the test set'.format(
        test_samples))
    cc_prevs, acc_prevs = compute_classify_count(test_sample_yhat, val_tpr, val_fpr, probabilistic=False)
    pcc_prevs, apcc_prevs = compute_classify_count(test_sample_yhat, val_ptpr, val_pfpr, probabilistic=True)

    print('Evaluate classify & count based methods with MSE')
    mse_samples = eval_metric(mse, true_prevs, cc_prevs, pcc_prevs, acc_prevs, apcc_prevs)

    print('Init quant_net training:')
    status_every = 10
    test_every = 100

    best_mse = None
    loss_sum = 0
    patience = PATIENCE
    losses = []
    t_init = time()
    for step in range(1, args.maxiter + 1):

        sample_length = min(10 + step // 10, MAX_SAMPLE_LENGTH) if args.incremental else args.samplelength
        adjust_learning_rate(optimizer, step, each=args.maxiter/2, initial_lr=args.lr)

        batch_yhat, batch_y, batch_p, stats, _ = \
            quantification_uniform_sampling(val_pos_ids, val_neg_ids, val_yhat_pos, val_yhat_neg,
                                            val_tpr, val_fpr, val_ptpr, val_pfpr,
                                            input_size, args.batchsize, sample_length, prevs_range=prevs_range)
        quant_net.train()
        optimizer.zero_grad()
        batch_phat = quant_net.forward(batch_yhat, stats)
        quant_loss = criterion(batch_phat, batch_p)
        quant_loss.backward()
        optimizer.step()

        loss = quant_loss.data[0]
        loss_sum += loss
        losses.append(loss)

        if step % status_every == 0:
            print('step {}\tloss {:.5}\tsample_length {}\tv {:.2f}'
                     .format(step, loss_sum / status_every, sample_length, status_every / (time() - t_init)))
            loss_sum = 0
            t_init = time()

        if step % test_every == 0 and sample_length==args.samplelength:
            quant_net.eval()

            test_batch_phat = quant_batched_predictions(quant_net, test_sample_yhat, test_sample_stats, batchsize=args.batchsize)

            # prevalence by sampling test -----------------------------------------------------------------------------
            net_prevs = []
            printmax = -1
            for i in range(test_samples):
                net_prevs.append(float(test_batch_phat[i,0]))

                if i < printmax:
                    print('\tsampling-test {}/{}:\tp={:.3f}\tcc={:.3f}\tacc={:.3f}\tpcc={:.3f}\tapcc={:.3f}\tnet={:.3f}'
                      .format(i,test_samples,test_sample_prev[i].data[0], cc_prevs[i], acc_prevs[i], pcc_prevs[i], apcc_prevs[i], net_prevs[i]))
                elif i == printmax:
                    print('\t...{} omitted'.format(test_samples-i))

            mse_net_sample = mse(true_prevs, net_prevs)

            print('Samples MSE:\tcc={:.5f} pcc={:.5f} acc={:.5f} apcc={:.5f} net={:.5f}'
                  .format(mse_samples[0], mse_samples[1], mse_samples[2], mse_samples[3], mse_net_sample))

            print('patience',patience)

            # plots ---------------------------------------------------------------------------------------------------
            methods = np.array([cc_prevs, acc_prevs, pcc_prevs, apcc_prevs, net_prevs])
            labels = ['cc', 'acc', 'pcc', 'apcc', 'net']
            title=args.data.upper() if args.data!='kindle' else args.data.title()
            plot_corr(true_prevs, methods, labels, savedir=args.plotdir, savename='corr.png', train_prev=train_prev, test_prev=None, title=title)#test_prev)
            plot_loss(range(step), losses, savedir=args.plotdir, savename='loss.png')

            if best_mse is None or mse_net_sample < best_mse:
                print('\tsaving model in ' + args.output)
                torch.save(quant_net, args.output)
                best_mse = mse_net_sample
                patience = PATIENCE
            else:
                patience -= 1
                if patience == 0:
                    print('Early stop after {} loss checks without improvement'.format(PATIENCE))
                    break

def parseargs(args):
    parser = argparse.ArgumentParser(description='Learn Quantifier Correction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data',
                        help='Name of the dataset. Valid ones are: imdb, hp, kindle')
    parser.add_argument('model',
                        help='Path to the classifier model')
    parser.add_argument('-v', '--vocabularysize',
                        help='Maximum length of the vocabulary', type=int, default=5000)
    parser.add_argument('-E', '--use-embeddings',
                        help='use the embeddings as input to the quantifier', default=False, action='store_true')
    parser.add_argument('-H', '--hiddensize',
                        help='Size of the LSTM hidden layers', type=int, default=64)
    parser.add_argument('-d', '--dropout',
                        help='Drop probability for dropout', type=float, default=0.5)
    parser.add_argument('-l', '--linlayers',
                        help='Linear layers on top of the LSTM output', type=int, default=[1024,512], nargs='+')
    parser.add_argument('--incremental',
                        help='Activates the incremental mode for the sample_length', default=False, action='store_true')
    parser.add_argument('-I', '--maxiter',
                        help='Maximum number of iterations', type=int, default=20000)
    parser.add_argument('-S', '--samplelength',
                        help='Length of the samples (in number of documents)', type=int, default=500)
    parser.add_argument('-O', '--output',
                        help='Path to the output file containing the model parameters', type=str,
                        default='./quant_net.[data].pt')
    parser.add_argument('--lr',
                        help='learning rate', type=float, default=0.0001)
    parser.add_argument('--weightdecay',
                        help='weight decay', type=float, default=1e-4)
    parser.add_argument('--batchsize',
                        help='batch size', type=float, default=19*5)
    parser.add_argument('--plotdir',
                        help='Path to the plots', type=str, default='../plots')
    parser.add_argument('--results',
                        help='Path to the results', type=str, default='../results.txt')
    parser.add_argument('--stats-layer',
                        help='Concatenates the statistics (tpr,fpr,cc,acc,pcc,apcc) to the dense representation of the quantification',
                        default=False, action='store_true')
    parser.add_argument('--stats-lstm',
                        help='Concatenates the statistics (tpr,fpr,cc,acc,pcc,apcc) to sequence in input to the LSTM',
                        default=False, action='store_true')
    parser.add_argument('--include-bounds',
                        help='Include the bounds 0 and 1 in the sampling. If otherwise, the sampling will be done from'
                             '0.05 to 0.95', default=False, action='store_true')


    return parser.parse_args(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
