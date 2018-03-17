
# TODO: wrap all baselines in the same interface with common interface

tr_prev = variable_from_numpy(np.mean(ytr, axis=0))
te_prev = variable_from_numpy(np.mean(yte, axis=0))
mae_naive = mean_absolute_error(tr_prev, te_prev)[0]
print('train_prevalence:', tr_prev)
print('test_prevalence:', te_prev)
print('Naive-MAE:\t%.8f' % mae_naive)
# sys.exit()
