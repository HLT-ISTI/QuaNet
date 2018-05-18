#!/usr/bin/env bash
set -x

#quantumconf="--stats-layer --stats-lstm"

for run in {0..9} ; do
for data in "kindle" "hp" "imdb" ; do
	classmodel=../models/class_net.$data.$run.pt
	quantmodel=../models/quant_net.$data.$run.pt
	quantmodelE=../models/quant_netE.$data.$run.pt
	quantmodelS=../models/quant_netS.$data.$run.pt
	quantmodelES=../models/quant_netES.$data.$run.pt
	
#	python3 learn_class.py $data --output $classmodel

#	python3 learn_quant.py $data $classmodel -E --stats-layer --output $quantmodelES --include-bounds
	python3 eval_quant.py $data $classmodel $quantmodelES $run -E --plotdir ../eval_plots_formanbounds_EM --results ../results/formanbounds_em.txt --result-note $run --include-bounds

#	python3 learn_quant.py $data $classmodel --output $quantmodel --include-bounds
#	python3 eval_quant.py $data $classmodel $quantmodel $run --plotdir ../eval_plots_bound --results ../results/eval_results_bound.txt --result-note $run --net-only

#	python3 learn_quant.py $data $classmodel -E --output $quantmodelE --include-bounds
#	python3 eval_quant.py $data $classmodel $quantmodelE $run -E --plotdir ../eval_plots_bound --results ../results/eval_results_bound.txt --result-note $run --net-only --include-bounds

#	python3 learn_quant.py $data $classmodel --stats-layer --output $quantmodelS --include-bounds
#	python3 eval_quant.py $data $classmodel $quantmodelS $run --plotdir ../eval_plots_bound --results ../results/eval_results_bound.txt --result-note $run --net-only --include-bounds
done
done
