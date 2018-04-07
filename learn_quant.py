import argparse,ntpath
from time import time
from nets.quantification import LSTMQuantificationNet
from plot_correction import plot_corr, plot_loss, plot_bins
from quantification.helpers import *
from util.helpers import *
# from inntt import *

def compute_classify_count_methods(test_batch_yhat, test_batch_p, val_tpr, val_fpr):

    test_samples = test_batch_yhat.shape[0]

    prevs = test_batch_p[:, 0].data
    prevs = prevs.cpu().numpy() if use_cuda else prevs.numpy()

    cc_prevs, pcc_prevs = [], []
    for i in range(test_samples):
        cc_prevs.append(classify_and_count(np.asarray(test_batch_yhat[i, :, :].data)))
        pcc_prevs.append(probabilistic_classify_and_count(np.asarray(test_batch_yhat[i, :, :].data)))

    acc_prevs = [adjusted_quantification(cc, val_tpr, val_fpr) for cc in cc_prevs]
    apcc_prevs = [adjusted_quantification(pcc, val_tpr, val_fpr) for pcc in pcc_prevs]

    return prevs, cc_prevs, pcc_prevs, acc_prevs, apcc_prevs


def eval_metric(metric, prevs, *methods):
    return [metric(prevs, method_i) for method_i in methods]


def main(args):
    print(args)
    args = parseargs(args)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loadDataset(dataset=args.data, vocabularysize=args.vocabularysize)
    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)

    class_net = torch.load(args.model)
    class_net = class_net.cuda() if use_cuda else class_net

    print('creating val_yhat and test_yhat')
    val_yhat = predict(class_net, x_val, args.use_embeddings)
    test_yhat = predict(class_net, x_test, args.use_embeddings)
    val_tpr = tpr(val_yhat, y_val)
    val_fpr = fpr(val_yhat, y_val)

    quant_lstm_layers = 1
    classes = 2
    input_size = classes if not args.use_embeddings else classes + 32

    quant_net = LSTMQuantificationNet(classes, input_size, args.hiddensize, quant_lstm_layers,
                                      args.linlayers, drop_p=args.dropout,
                                      stats_in_lin_layers=args.stats_layer,
                                      stats_in_sequence=args.stats_lstm)

    quant_net = quant_net.cuda() if use_cuda else quant_net
    print(quant_net)

    quant_loss_function = torch.nn.MSELoss()
    quant_optimizer = torch.optim.Adam(quant_net.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    with open('quant_net_hist.txt', mode='w', encoding='utf-8') as outputfile, \
            open('quant_net_test.txt', mode='w', encoding='utf-8') as testoutputfile:

        train_prev = np.mean(y_train)
        test_prev = np.mean(y_test)
        ntests = y_test.shape[0]

        test_samples = 10000

        val_yhat_pos, val_yhat_neg = split_pos_neg(val_yhat, y_val)
        test_yhat_pos, test_yhat_neg = split_pos_neg(test_yhat, y_test)

        test_sample_yhat, test_sample_y, test_sample_prev, test_sample_stats = \
            quantification_batch(test_yhat_pos, test_yhat_neg, val_tpr, val_fpr, input_size, test_samples, sample_length=500)


        print('Computing classify & count based methods (cc, acc, pcc, apcc) for {} samples of the test set'.format(test_samples))
        prevs, cc_prevs, pcc_prevs, acc_prevs, apcc_prevs = compute_classify_count_methods(test_sample_yhat, test_sample_prev, val_tpr, val_fpr)

        print('Evaluate classify & count based methods with MAE and MSE')
        mae_samples = eval_metric(mae, prevs, cc_prevs, pcc_prevs, acc_prevs, apcc_prevs)
        mse_samples = eval_metric(mse, prevs, cc_prevs, pcc_prevs, acc_prevs, apcc_prevs)

        print('Computing classify & count based methods (cc, acc, pcc, apcc) for the test set')
        cc = classify_and_count(test_yhat)
        pcc = probabilistic_classify_and_count(test_yhat)
        acc = adjusted_quantification(cc, val_tpr, val_fpr)
        apcc = adjusted_quantification(pcc, val_tpr, val_fpr)
        mae_test = eval_metric(mae, [test_prev], [cc], [pcc], [acc], [apcc])
        mse_test = eval_metric(mse, [test_prev], [cc], [pcc], [acc], [apcc])

        print('Init quant_net training:')
        status_every = 10
        test_every = 200

        best_loss = None
        quant_loss_sum = 0
        patience = 0
        losses = []
        t_init = time()
        for step in range(1, args.maxiter + 1):

            sample_length = min(10 + step // 10, MAX_SAMPLE_LENGTH) if args.incremental else 500
            adjust_learning_rate(quant_optimizer, step, each=args.maxiter/2, initial_lr=args.lr)

            batch_yhat, batch_y, batch_p, stats = quantification_batch(val_yhat_pos, val_yhat_neg,
                                                                       val_tpr, val_fpr, input_size,
                                                                       args.batchsize, sample_length)
            quant_net.train()
            quant_optimizer.zero_grad()
            batch_phat = quant_net.forward(batch_yhat, stats)
            quant_loss = quant_loss_function(batch_phat, batch_p)
            quant_loss.backward()
            quant_optimizer.step()

            loss = quant_loss.data[0]
            quant_loss_sum += loss
            losses.append(loss)

            if step % status_every == 0:
                printtee('step {}\tloss {:.5}\tsample_length {}\tv {:.2f}'
                         .format(step, quant_loss_sum / status_every, sample_length, status_every / (time() - t_init)),
                         outputfile)
                quant_loss_sum = 0
                t_init = time()

            if step % test_every == 0 and sample_length==500:
                quant_net.eval()

                test_batch_phat = quant_batched_predictions(quant_net, test_sample_yhat, test_sample_stats, batchsize=args.batchsize)

                # prevalence by sampling test -----------------------------------------------------------------------------
                net_prevs, anet_prevs = [], []
                printmax = 10
                for i in range(test_samples):
                    net_prevs.append(float(test_batch_phat[i, 0]))

                    if i < printmax:
                        printtee('\tsampling-test {}/{}:\tp={:.3f}\tcc={:.3f}\tacc={:.3f}\tpcc={:.3f}\tapcc={:.3f}\tnet={:.3f}'
                          .format(i,test_samples,
                                  test_sample_prev[i, 0].data[0], cc_prevs[i], acc_prevs[i], pcc_prevs[i], apcc_prevs[i], net_prevs[i]),
                                 outputfile)
                    elif i == printmax: printtee('\t...{} omitted'.format(test_samples-i), outputfile)

                mae_net_sample = mae(prevs, net_prevs)
                mse_net_sample = mse(prevs, net_prevs)

                printtee('Samples MAE:\tcc={:.5f} pcc={:.5f} acc={:.5f} apcc={:.5f} net={:.5f}'
                      .format(mae_samples[0], mae_samples[1], mae_samples[2], mae_samples[3], mae_net_sample), testoutputfile)

                printtee('Samples MSE:\tcc={:.5f} pcc={:.5f} acc={:.5f} apcc={:.5f} net={:.5f}'
                      .format(mse_samples[0], mse_samples[1], mse_samples[2], mse_samples[3], mse_net_sample), testoutputfile)

                # plots ---------------------------------------------------------------------------------------------------
                methods = [cc_prevs, acc_prevs, pcc_prevs, apcc_prevs, net_prevs]
                labels = ['cc', 'acc', 'pcc', 'apcc', 'net']
                plot_corr(prevs, methods, labels, savedir=args.plotdir, savename='corr.png',
                          train_prev=train_prev, test_prev=test_prev)
                plot_bins(prevs, methods, labels, mae, bins=10,
                          savedir=args.plotdir, savename='bins_' + mae.__name__ + '.png')
                plot_bins(prevs, methods, labels, mse, bins=10,
                          savedir=args.plotdir, savename='bins_' + mse.__name__ + '.png')
                plot_loss(range(step), losses,
                          savedir=args.plotdir, savename='loss.png')

                # full test -----------------------------------------------------------------------------------------------
                #test_batch_phat = quant_batched_predictions(quant_net, test_batch_yhat, test_stats, batchsize=args.batchsize)
                net_prev_test = 0
                for test_batch_yhat_i, _, _, test_stats_i in \
                        create_fulltest_batch(test_yhat, y_test, val_tpr, val_fpr, input_size, batch_size=args.batchsize, sample_length=sample_length):
                    test_batch_phat = quant_net.forward(test_batch_yhat_i, test_stats_i)
                    ntest_in_batch = test_batch_yhat_i.shape[0]*test_batch_yhat_i.shape[1]
                    estim_batch_prev = np.mean(test_batch_phat[:, 0].data.cpu().numpy())
                    net_prev_test +=  estim_batch_prev * (ntest_in_batch/ntests)

                net_prev_test = np.mean(net_prev_test)
                mae_net_test = mae([test_prev], [net_prev_test])
                mse_net_test = mse([test_prev], [net_prev_test])

                printtee('\nFullTest prevalence cc={:.4f} pcc={:.4f} acc={:.4f} apcc={:.4f} net={:.4f} [train_prev={:.4f} test_prev={:.4f}]'
                    .format(cc, pcc, acc, apcc, net_prev_test, train_prev, test_prev), testoutputfile)

                printtee('FullTest MAE:\tcc={:.5f} pcc={:.5f} acc={:.5f} apcc={:.5f} net={:.5f}'
                         .format(mae_test[0], mae_test[1], mae_test[2], mae_test[3], mae_net_test),
                         testoutputfile)

                printtee('FullTest MSE:\tcc={:.5f} pcc={:.5f} acc={:.5f} apcc={:.5f} net={:.5f}'
                         .format(mse_test[0], mse_test[1], mse_test[2], mse_test[3], mse_net_test),
                         testoutputfile)

                if best_loss is None or quant_loss_sum < best_loss:
                    print('\tsaving model in ' + args.output)
                    torch.save(quant_net, args.output)
                    best_loss = quant_loss_sum
                    best_metrics = {'mae_net_sample':mae_net_sample, 'mse_net_sample':mse_net_sample,
                                    'mae_net_test': mae_net_test, 'mse_net_test': mse_net_test}
                    patience = 0
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stop after 20 loss checks without improvement')
                        break

    qr = QuantificationResults(train_prev=train_prev, test_prev=test_prev)
    qr.add_results('mae', 'sample', *mae_samples, best_metrics['mae_net_sample'])
    qr.add_results('mse', 'sample', *mse_samples, best_metrics['mse_net_sample'])
    qr.add_results('mae', 'full', *mae_test, best_metrics['mae_net_test'])
    qr.add_results('mse', 'full', *mse_test, best_metrics['mse_net_test'])

    add_header = (not os.path.exists(args.results))
    with open(args.results, 'a') as res:
        if add_header:
            res.write('data\t'+qr.header()+'\n')
        dataset_name = ntpath.basename(args.data)
        res.write(dataset_name+'\t'+qr.show()+'\n')

    return qr

def parseargs(args):
    parser = argparse.ArgumentParser(description='Learn Quantifier Correction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data',
                        help='Path to the corpus file')
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
                        help='Linear layers on top of the LSTM output', type=int, default=[1024, 512], nargs='+')
    parser.add_argument('--incremental',
                        help='Activates the incremental mode for the sample_length', default=False, action='store_true')
    parser.add_argument('-I', '--maxiter',
                        help='Maximum number of iterations', type=int, default=10000)
    parser.add_argument('-O', '--output',
                        help='Path to the output file containing the model parameters', type=str,
                        default='./quant_net.pt')
    parser.add_argument('--lr',
                        help='learning rate', type=float, default=0.0001)
    parser.add_argument('--weightdecay',
                        help='weight decay', type=float, default=0)
    parser.add_argument('--batchsize',
                        help='batch size', type=float, default=100)
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

    return parser.parse_args(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
