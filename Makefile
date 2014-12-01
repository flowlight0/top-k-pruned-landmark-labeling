CXX = g++
CXXFLAGS = -g -Wall -Wextra -Ilib -O3 -msse

LIB = -lgtest -lgtest_main -lpthread
OBJ = src/top_k_pruned_landmark_labeling.o

all: bin bin/construct_index bin/k_distance bin/test 

bin:
	mkdir -p bin

bin/construct_index: sample/construct_index_main.cpp ${OBJ}
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB) -Isrc

bin/k_distance: sample/k_distance_main.cpp ${OBJ}
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB) -Isrc

bin/test: src/top_k_pruned_landmark_labeling_test.cpp lib/gtest/gtest-all.cc lib/gtest/gtest_main.cc ${OBJ}
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB) 


.PHONY:	test clean

test: bin/test
	./$<

clean:
	rm -rf bin
	rm ${OBJ}
