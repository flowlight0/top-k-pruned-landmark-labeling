#include <cstdlib>
#include <iostream>
#include "top_k_pruned_landmark_labeling.hpp"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " (index_file)" << endl;
    exit(EXIT_FAILURE);
  }
  
  TopKPrunedLandmarkLabeling kpll;
  
  if (!kpll.LoadIndex(argv[1])) {
    cerr << "error: Load failed" << endl;
    exit(EXIT_FAILURE);
  }
  
  for (int u, v; cin >> u >> v; ) {
    vector<int> dist;
    kpll.KDistanceQuery(u, v, dist);
    
    cout << dist.size();
    for (size_t i = 0; i < dist.size(); i++){
      cout << " " << dist[i];
    }
    cout << endl;
  }
  exit(EXIT_SUCCESS);
}
