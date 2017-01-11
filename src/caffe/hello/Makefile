# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

# Added for opencv
CFLAGS+= `pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv` -lglog -lboost_system -lboost_thread -lboost_filesystem -DUSE_OPENCV 

# Added for mxnet
#export MXNET_ROOT=`pwd`/../../../../mxnet
export MXNET_ROOT=/dk/cnn/caffe

CFLAGS+= -I$(MXNET_ROOT)/include 
LDFLAGS+=$(MXNET_ROOT)/.build_release/lib/libcaffe.so
CC=g++

#image-classification-predict: image-classification-predict.o
#	g++ -O3 -o image-classification-predict image-classification-predict.o $(LDFLAGS)

#image-classification-predict.o: image-classification-predict.cc
#	g++ -O3 -c image-classification-predict.cc ${CFLAGS}
	
all:
	g++ -W hello.cc  $(CFLAGS)  $(LDFLAGS) -o hello 
clean: 
	rm image-classification-predict
	rm -f *.d *.o

.cc.o:
	echo [CC] $<
	g++ -W $<  $(CFLAGS)  $(LDFLAGS) -o $@

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./
