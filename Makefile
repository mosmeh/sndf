#!/bin/make

all: learn

learn:
	g++ -std=c++11 -O3 -Wall learn.cpp stochastic_decision_tree.cpp -lgflags -o learn

clean:
	$(RM) learn
