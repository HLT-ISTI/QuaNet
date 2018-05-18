import numpy as np

# TODO: evaluation trials-policies
# TODO: evaluation metrics
# TODO: use, as an evaluation metric, the p-values of two proportions in the Z-distribution (for each category);
# ...and figure out how to sum them up. This should not be difficult because all them share the number of documents (population size)
def count(y):
    return (np.mean(y, axis=0))

def evaluation(estimated_prevalences, true_prevalences, evaluation_measure):
    estimated_prevalences = self.predict_prevalences(X)
    estimated_prevalences = variable_from_numpy(estimated_prevalences)

    true_prevalences = count(y)
    true_prevalences = variable_from_numpy(true_prevalences)

    mae = evaluation_measure(estimated_prevalences, true_prevalences)

    mae = mae[0].data
    mae = mae.cpu() if args.cuda else mae
    return mae.numpy()[0]

def wilcoxon_comparison(X, y, method1, method2, eval_metric, lower_is_better=True, until_p = 0.05, max_iter=100):
    from scipy.stats import wilcoxon
    signficance_detected = False
    results1, results2 = [], []
    while not signficance_detected:
        Xs, ys = sample_collection(X,y)

        results1.append(method1.evaluation(Xs, ys, eval_metric))
        results2.append(method2.evaluation(Xs, ys, eval_metric))

        if len(results1) > 30:
            _,p = wilcoxon(results1, results2)
            signficance_detected = (p < until_p)
            print(results1[-1], results2[-1], signficance_detected)
        else:
            print(results1[-1],results2[-1])
        if len(results1) > max_iter:
            break

    if signficance_detected:
        r1ave = np.mean(results1)
        r2ave = np.mean(results2)
        lower,bigger = (method1,method2) if r1ave < r2ave else (method2,method1)
        return lower if lower_is_better else bigger

    print('Wilcoxon indetermined after {} iterations'.format(max_iter))
    return None


def random_split_evaluation(X, y, methods, eval_metrics, nsplits=10, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    nD = y.shape[0]
    doc_indexes = np.random.permutation(nD)
    results = np.zeros((nsplits, len(methods), len((eval_metrics))), dtype=float)
    for s in range(nsplits):
        split_indexes = doc_indexes[s*nD//nsplits:(s+1)*nD//nsplits]
        Xs, ys = X[split_indexes], y[split_indexes]

        for i, method_i in enumerate(methods):
            for j,eval_j in enumerate(eval_metrics):
                results[s,i,j] = method_i.evaluation(Xs, ys, eval_j)

    return np.mean(results, axis=0), np.std(results, axis=0)
