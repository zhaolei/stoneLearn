
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall -Wsign-compare

export CAFFE_ROOT=/ds/github/caffe/install

CFLAGS+= -D CPU_ONLY -lcaffe -lstdc++ -lglog

CFLAGS+= -I$(CAFFE_ROOT)/include 
#LDFLAGS+=$(CAFFE_ROOT)/lib/libcaffe.so
LDFLAGS+=-L$(CAFFE_ROOT)/lib/
CC=g++

all:main.cpp
	$(CC) -o blob main.cpp $(LDFLAGS) $(CFLAGS)  
clean:
	rm blob
