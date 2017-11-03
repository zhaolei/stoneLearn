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
export CAFFE_ROOT=/ds/local/caffe/install

CFLAGS+= -I$(CAFFE_ROOT)/include 
LDFLAGS+=$(CAFFE_ROOT)/lib/libcaffe.so
CC=g++

#image-classification-predict: image-classification-predict.o
#	g++ -O3 -o image-classification-predict image-classification-predict.o $(LDFLAGS)

#image-classification-predict.o: image-classification-predict.cc
#	g++ -O3 -c image-classification-predict.cc ${CFLAGS}
	
all:
	g++ -W hello.cc $(LDFLAGS) $(CFLAGS)   -o hello 
clean: 
	rm image-classification-predict
	rm -f *.d *.o

.cc.o:
	echo [CC] $<
	g++ -W $<  $(CFLAGS)  $(LDFLAGS) -o $@

