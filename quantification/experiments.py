
dataset, categories, features, weight, log = parse_dataset_code(args.dataset)
print('loading ' + dataset, categories, features, weight, log)
#data = TextCollectionLoader(dataset=dataset, vectorizer=weight, sublinear_tf=log, feat_sel=features, rep_mode='dense', top_categories=categories)
data = ReviewsDataset.load('/home/moreo/pytecs/datasets/build/online/hp/Mat2008_1OnlineS3F.pkl')
fs = RoundRobin(k=features)
data.Xtr = fs.fit_transform(data.Xtr, data.ytr)
if data.ismultipletest():
    for i,Xtei in enumerate(data.Xte):
        data.Xte[i] = fs.transform(Xtei)
else:
    data.Xte = fs.transform(data.Xte)


nF = data.num_features()
nC = data.num_categories()

net = Net(input_size=int((nF+1)*nF/2), hidden_size=args.hidden, num_classes=nC)
net = net.cuda() if args.cuda else net

# Evaluation measures
mean_absolute_error = nn.L1Loss()

Xtr, ytr = data.get_devel_set()
Xte, yte = data.get_test_set()
if len(ytr.shape)==1:
    ytr = ytr.reshape(-1,1)
    yte = yte.reshape(-1, 1)

tr_prev = variable_from_numpy(np.mean(ytr, axis=0))
te_prev = variable_from_numpy(np.mean(yte, axis=0))
mae_naive = mean_absolute_error(tr_prev, te_prev)[0]
print('train_prevalence:', tr_prev)
print('test_prevalence:', te_prev)
print('Naive-MAE:\t%.8f' % mae_naive)
#sys.exit()

train(Xtr, ytr, net, evaluation_measure=mean_absolute_error, Xte=Xte, yte=yte, num_steps=args.iter, learning_rate=args.lr, weight_decay=args.weight_decay)
mae_net = net.evaluation(Xte, yte, evaluation_measure=mean_absolute_error, verbose=True)

svm = SVMclassifyAndCount(class_weight='balanced')
svm.fit(Xtr, ytr)
mae_svm = svm.evaluation(Xte, yte, evaluation_measure=mean_absolute_error, verbose=True)

winner = wilcoxon_comparison(Xte, yte, net, svm, eval_metric=mean_absolute_error, lower_is_better=True)
if winner:
    print('Wilcoxon test, winner: ' + winner.__class__.__name__)

print('Net-MAE:\t%.8f' % mae_net)
print('SVM-MAE:\t%.8f' % mae_svm)
print('Naive-MAE:\t%.8f' % mae_naive)