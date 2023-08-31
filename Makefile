list:
	@grep '^[^#[:space:]].*:' Mak=t3le

run-all:
	kedro run --runner=ParallelRunner &&\
 	kedro run --env=grid --runner=ParallelRunner &&\
	kedro run --env=lastfm --runner=ParallelRunner &&\
	kedro run --env=reddit --runner=ParallelRunner &&\
	kedro run --env=amazon --runner=ParallelRunner &&\
	kedro run --env=x3_o1000 --runner=ParallelRunner &&\
	kedro run --env=x4_o1000 --runner=ParallelRunner &&\
	kedro run --env=x5_o1000 --runner=ParallelRunner &&\
	kedro run --env=x6_o1000 --runner=ParallelRunner &&\
	kedro run --env=x7_o1000 --runner=ParallelRunner

run-all-f:
	kedro run --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=grid --pipeline=t1 --runner=ParallelRunner  &&\
	kedro run --env=lastfm --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=reddit --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=amazon --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x3_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x4_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x5_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x6_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x7_o1000 --pipeline=t1 --runner=ParallelRunner

run-all-i:
	kedro run --env=x3_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x4_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x5_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x6_o1000 --pipeline=t1 --runner=ParallelRunner &&\
	kedro run --env=x7_o1000 --pipeline=t1 --runner=ParallelRunner