list:
	@grep '^[^#[:space:]].*:' Makefile

run-all:
	python src/main.py --input graph/beta/facebook.csv --output emb/facebook.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1  &&\
	python src/main.py --input graph/beta/amazon.csv --output emb/amazon.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1  &&\
	python src/main.py --input graph/beta/grid.csv --output emb/grid.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1 &&\
	python src/main.py --input graph/beta/lastfm.csv --output emb/lastfm.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1 &&\
	python src/main.py --input graph/beta/reddit.csv --output emb/reddit.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1

run-abcd-1000:
	python src/main.py --input graph/beta/x3.csv --output emb/x3.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1 &&\
	python src/main.py --input graph/beta/x4.csv --output emb/x4.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1 &&\
	python src/main.py --input graph/beta/x5.csv --output emb/x5.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1 &&\
	python src/main.py --input graph/beta/x6.csv --output emb/x6.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1 &&\
	python src/main.py --input graph/beta/x7.csv --output emb/x7.emb --num-walks 10 --walk-length 50 --window-size 5 --dimensions 16 --OPT1 True --OPT2 True --OPT3 True --workers 1