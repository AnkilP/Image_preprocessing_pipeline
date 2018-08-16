
CC = g++
CFLAGS = -g -Wall
SRCS = adaptive_manifold.hpp
PROG = adaptive_manifold

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)