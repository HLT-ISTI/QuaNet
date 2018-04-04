from nets.quantification import LSTMQuantificationNet
from plot_correction import plot_corr, plot_loss, plot_bins
from helper import *
# from inntt import *

dataset = 'hp'
max_features = 5000
use_document_embeddings_from_classifier = True
class_steps = 20000

(x_train, y_train), (x_val, y_val), (x_test, y_test) = loadDataset(dataset, max_features=max_features)
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

with open('class_' + get_name(class_steps, info=dataset), mode='br') as modelfile:
    class_net = torch.load(modelfile)
    class_net = class_net.cuda() if use_cuda else class_net


print('creating val_yhat and test_yhat')
val_yhat = predict(class_net, x_val, use_document_embeddings_from_classifier)
test_yhat = predict(class_net, x_test, use_document_embeddings_from_classifier)
val_tpr = tpr(val_yhat, y_val)
val_fpr = fpr(val_yhat, y_val)

quant_lstm_hidden_size = 64
quant_lstm_layers = 1
quant_lin_layers_sizes = [1024, 512]  # [128,64,32]
classes = 2
stats_in_lin_layers = True
stats_in_sequence = True
drop_p = 0.5
input_size = classes if not use_document_embeddings_from_classifier else classes + 32

quant_net = LSTMQuantificationNet(classes, input_size, quant_lstm_hidden_size, quant_lstm_layers,
                                  quant_lin_layers_sizes, drop_p=drop_p,
                                  stats_in_lin_layers=stats_in_lin_layers,
                                  stats_in_sequence=stats_in_sequence)

quant_net = quant_net.cuda() if use_cuda else quant_net
print(quant_net)

lr = 0.0001
weight_decay = 0.00001

quant_loss_function = torch.nn.MSELoss()
quant_optimizer = torch.optim.Adam(quant_net.parameters(), lr=lr, weight_decay=weight_decay)

print('init quantification')
with open('quant_net_hist.txt', mode='w', encoding='utf-8') as outputfile, \
        open('quant_net_test.txt', mode='w', encoding='utf-8') as testoutputfile:

    # if interactive:
    #    innt = InteractiveNeuralTrainer()
    #    innt.add_optim_param_adapt('ws', quant_optimizer, 'lr', inc_factor=10.)
    #    innt.add_optim_param_adapt('da', quant_optimizer, 'weight_decay', inc_factor=2.)
    #    innt.start()

    quant_steps = 10000
    status_every = 10
    test_every = 200
    save_every = 1000
    test_samples = 1000
    batch_size = 100
    interactive = False

    quant_loss_sum = 0
    t_init = time()
    val_yhat_pos = val_yhat[y_val == 1]
    val_yhat_neg = val_yhat[y_val != 1]
    test_yhat_pos = test_yhat[y_test == 1]
    test_yhat_neg = test_yhat[y_test != 1]
    losses = []
    for step in range(1, quant_steps + 1):

        sample_length = min(10 + step // 10, MAX_SAMPLE_LENGTH)

        batch_yhat, batch_y, batch_p, stats = create_batch_(val_yhat_pos, val_yhat_neg,
                                                            val_tpr, val_fpr, input_size,
                                                            batch_size, sample_length)

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
            printtee('step {}\tloss {:.5}\tv {:.2f}'
                     .format(step, quant_loss_sum / status_every, status_every / (time() - t_init)), outputfile)
            quant_loss_sum = 0
            t_init = time()

        if step % test_every == 0:
            quant_net.eval()

            test_batch_yhat, test_batch_y, test_batch_p, stats = create_batch_(test_yhat_pos, test_yhat_neg, val_tpr,
                                                                               val_fpr, input_size, test_samples,
                                                                               sample_length)
            test_batch_phat = quant_net.forward(test_batch_yhat, stats)

            # prevalence by sampling test -----------------------------------------------------------------------------
            prevs, net_prevs, cc_prevs, pcc_prevs, acc_prevs, apcc_prevs, anet_prevs = [], [], [], [], [], [], []
            for i in range(test_samples):

                net_prev = float(test_batch_phat[i, 0])
                cc_prev = classify_and_count(np.asarray(test_batch_yhat[i, :, :].data))
                pcc_prev = probabilistic_classify_and_count(np.asarray(test_batch_yhat[i, :, :].data))

                anet_prev = adjusted_quantification(net_prev, val_tpr, val_fpr)
                acc_prev = adjusted_quantification(cc_prev, val_tpr, val_fpr)
                apcc_prev = adjusted_quantification(pcc_prev, val_tpr, val_fpr)

                prevs.append(test_batch_p[i, 0].data[0])

                net_prevs.append(net_prev)
                cc_prevs.append(cc_prev)
                pcc_prevs.append(pcc_prev)
                anet_prevs.append(anet_prev)
                acc_prevs.append(acc_prev)
                apcc_prevs.append(apcc_prev)

                printtee('step {}\tp={:.3f}\tcc={:.3f}\tacc={:.3f}\tpcc={:.3f}\tapcc={:.3f}\tnet={:.3f}\tanet={:.3f}'
                      .format(step, test_batch_p[i, 0].data[0], cc_prev, acc_prev, pcc_prev, apcc_prev, net_prev,
                              anet_prev), outputfile)

            printtee('Average MAE:\tcc={:.5f}\tacc={:.5f}\tpcc={:.5f}\tapcc={:.5f}\tnet={:.5f}\tanet={:.5f}'
                  .format(mae(prevs, cc_prevs), mae(prevs, acc_prevs), mae(prevs, pcc_prevs), mae(prevs, apcc_prevs),
                          mae(prevs, net_prevs), mae(prevs, anet_prevs)), testoutputfile)
            printtee('Average MSE:\tcc={:.5f}\tacc={:.5f}\tpcc={:.5f}\tapcc={:.5f}\tnet={:.5f}\tanet={:.5f}'
                  .format(mse(prevs, cc_prevs), mse(prevs, acc_prevs), mse(prevs, pcc_prevs), mse(prevs, apcc_prevs),
                          mse(prevs, net_prevs), mse(prevs, anet_prevs)), testoutputfile)

            # plots ---------------------------------------------------------------------------------------------------
            methods = [cc_prevs, acc_prevs, pcc_prevs, apcc_prevs, net_prevs]
            labels = ['cc', 'acc', 'pcc', 'apcc', 'net']
            plot_corr(prevs, methods, labels, savedir='../plots', savename='corr.png')  # 'step_' + str(step) + '.png')
            plot_bins(prevs, methods, labels, mae, bins=10,
                      savedir='../plots', savename='bins_' + mae.__name__ + '.png')
            plot_bins(prevs, methods, labels, mse, bins=10,
                      savedir='../plots', savename='bins_' + mse.__name__ + '.png')
            plot_loss(range(step), losses, savedir='../plots', savename='loss.png')

            # full test -----------------------------------------------------------------------------------------------
            p_ave = []
            true_prev = np.mean(y_test)
            cc = classify_and_count(test_yhat)
            acc = adjusted_quantification(cc, val_tpr, val_fpr)
            pcc = probabilistic_classify_and_count(test_yhat)
            apcc = adjusted_quantification(pcc, val_tpr, val_fpr)
            for test_batch_yhat, test_batch_y, test_batch_p, stats in \
                    create_fulltest_batch(test_yhat, y_test, val_tpr,val_fpr, input_size, batch_size=batch_size, sample_length=sample_length):
                test_batch_phat = quant_net.forward(test_batch_yhat, stats)
                ntest_in_batch = test_batch_yhat.shape[0]*test_batch_yhat.shape[1]
                p_ave.append(np.mean(test_batch_phat[:, 0].data.cpu().numpy()) ) # * ntest_in_batch

            p_ave = np.mean(p_ave)
            printtee('FullTest prevalence cc={:.4f} acc={:.4f} pcc={:.4f} apcc={:.4f} net={:.4f} true_prev={:.4f}'
                  .format(cc, acc, pcc, apcc, p_ave, true_prev), testoutputfile)

        if step % save_every == 0:
            filename = get_name(step, dataset)
            print('saving to', filename)
            with open(filename, mode='bw') as modelfile:
                torch.save(quant_net, modelfile)
