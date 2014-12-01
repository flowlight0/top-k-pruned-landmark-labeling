#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "top_k_pruned_landmark_labeling.hpp"
using namespace std;

void read_graph(const char *graph_file, vector<pair<int, int> > &es){
  ifstream ifs(graph_file);
  if (!ifs.good()){
    cerr << "Error: Cannot open " << graph_file << "." << endl;
    exit(EXIT_FAILURE);
  }

  es.clear();
  for (int u, v; ifs >> u >> v;){
    es.push_back(make_pair(u, v));
  }

  ifs.close();
}

int main(int argc, char **argv) {
  if (argc != 5) {
    cerr << "Usage: " << argv[0] << " (graph_file) (K) (D) (index_file)" << endl;
    exit(EXIT_FAILURE);
  }

  const char   *graph_file = argv[1];
  const size_t  K          = atoi(argv[2]);
  const bool    directed   = atoi(argv[3]) != 0;
  const char   *index_file = argv[4];
  
  vector<pair<int, int> > es;
  read_graph(graph_file, es);
  
  TopKPrunedLandmarkLabeling kpll;
  kpll.ConstructIndex(es, K, directed);
  
  ofstream ofs(index_file);
  
  if (!ofs.good()){
    cerr << "Error: Cannot open " << index_file << "." << endl;
    exit(EXIT_FAILURE);
  }
  
  kpll.StoreIndex(ofs);
  ofs.close();
  
  exit(EXIT_SUCCESS);
}
