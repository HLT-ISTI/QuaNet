import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv('../../results/eval_results_bound.txt', index_col=0)

mu_std_cnt = lambda scores: (np.mean(scores),np.std(scores),[scores.tolist()])
#getall = lambda scores: [scores.tolist()]

piv = pd.pivot_table(df, values=['score'],index=['method'],columns=['metric','dataset'], aggfunc=mu_std_cnt)

#statistical significance by column
stat_sig_test = ttest_rel
def pval_interpretation(pval):
    print(pval)
    if pval < 0.005:
        return 0 # are distinguishable at a very high conf level
    elif pval < 0.05:
        return 1 # are distinguishable, but at a low conf level
    else:
        return 2 # are indistinguishable

def result2str(mu,std,dags,isbest,precision=4):
    dags = '\dag'*dags
    strval = '{:.3f} $\pm$ {:.3f}{}'.format(mu,std,dags)
    if isbest:
        strval = '\\textbf{'+strval+'}'
    return strval


score_is_error = True
best = np.argmin if score_is_error else np.argmax
for col in piv.columns:
    col_index = piv[col]
    values = col_index.values
    means,stds,results_lists = zip(*values)
    best_pos = best(means)

    # all others are compared against the best one with a statistical test
    n_methods = len(means)
    for pos in range(n_methods):
        if pos != best_pos:
            _,pval=stat_sig_test(results_lists[pos][0], results_lists[best_pos][0])
            n_dags = pval_interpretation(pval)
        else:
            n_dags = 0
        col_index.iloc[pos] = result2str(means[pos],stds[pos],n_dags,isbest=(pos==best_pos or means[pos]==means[best_pos]))


#reorder
piv = piv.reindex_axis(['cc','pcc','acc','apcc','net'], axis='rows')
piv = piv.reindex_axis(['mae','mse','mnkld','mrae'], axis='columns', level='metric')
piv = piv.reindex_axis(['hp','kindle','imdb'], axis='columns', level='dataset')

#postprocessing
latex = piv.to_latex(escape=False)
latex = latex.replace('hp','HP').replace('kindle','Kindle').replace('imdb','IMDB')
latex = '\\begin{table}\n' \
        + latex \
        + '\n\caption{}' \
        + '\n\\end{table}'
print(latex)



