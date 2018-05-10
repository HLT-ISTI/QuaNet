from quantification.helpers import *
from learn_class import main as classifier
from learn_quant import main as quantifier
from os.path import join
import glob, ntpath
import subprocess
from subprocess import PIPE, STDOUT


class_common_args = []
quant_common_args = ['--use-embeddings', '--stats-lstm', '--stats-layer',  '--incremental']

dataset_dir = '../datasets/build/online'

for datagroup in ['hp','kindle']:

    write_header = True
    with open('./results_'+datagroup+'.csv', 'w') as fo:

        for datapath in glob.glob(join(dataset_dir, datagroup,'Seq*OnlineS3F.pkl')):
            dataset_name = ntpath.basename(datapath)
            modelname = dataset_name.replace('/','_').replace('.pkl', '')

            classmodelpath = '../models/class_' + datagroup + '_' + modelname + '.pt'
            quantmodelpath = '../models/quant_' + datagroup + '_' + modelname + '.pt'

            plotdir = join('../plots', datagroup, modelname)

            class_args = [datapath,'--output', classmodelpath] + class_common_args
            #classifier(class_args) #seems to create memory leaks
            p = subprocess.run(['python3','learn_class.py'] + class_args)

            quant_args = [datapath, classmodelpath, '--plotdir', plotdir, '--output',quantmodelpath] + quant_common_args
            #quant_results = quantifier(quant_args) #seems to create memory leaks
            p = subprocess.run(['python3', 'learn_quant.py'] + quant_args + ['--results','./results_'+datagroup+'.csv'])

            #if write_header:
            #    fo.write('data\t'+quant_results.header()+'\n')
            #    write_header = False
            #fo.write(datagroup+'-'+dataset_name+'\t'+quant_results.show()+'\n')
            #fo.flush()


