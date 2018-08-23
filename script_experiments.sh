#!/usr/bin/env bash
set -x

for run in {0..9} ; do
	for data in "kindle" "hp" "imdb" ; do
		classmodel=../models/class_net.$data.$run.pt
		quantmodelES=../models/quant_netES.$data.$run.pt
	
		python3 learn_class.py $data --output $classmodel

		python3 learn_quant.py $data $classmodel -E --stats-layer --output $quantmodelES --include-bounds
		python3 eval_quant.py $data $classmodel $quantmodelES $run -E --plotdir ../plots --results ../results/scores.txt --result-note $run --include-bounds

	done
done
