#include "top_k_pruned_landmark_labeling.hpp"
#include <vector>
#include <cassert>
#include <queue>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "gtest/gtest.h"
using namespace std;

const int V = 30;  
const int K = 30;

class TopKNaive{
  typedef vector<vector<int> > Graph;
  Graph graph;
    
public:
    
  TopKNaive(const vector<pair<int, int> > &es, size_t, bool directed){
    size_t V = 0;
    for (size_t i = 0; i < es.size(); i++){
      V = max(V, (size_t)es[i].first  + 1);
      V = max(V, (size_t)es[i].second + 1);
    }
    graph.resize(V);
    
    for (size_t i = 0; i < es.size(); i++){
      graph[es[i].first].push_back(es[i].second);

      if (!directed){
        graph[es[i].second].push_back(es[i].first);
      }
    }
  }

  uint8_t KDistanceQuery(int s, int t, uint8_t k, vector<int> &ret) {
    const size_t N = graph.size();
        
    vector<vector<int> > dist(N);
    priority_queue<pair<int, int > > que;
    que.push(make_pair(0, s));
        
    while (!que.empty()) {
      int v = que.top().second;
      int c = -que.top().first;
      que.pop();
      
      if (dist[v].size() >= k)  continue;
      
      dist[v].push_back(c);
      for(size_t i = 0; i < graph[v].size(); i++){
        que.push(make_pair(-(1 + c), graph[v][i]));
      }
    }
        
    ret = dist[t];
    return dist[t].size();
  }
};

void CheckAllVertexPairs(const vector<pair<int, int> > &es, size_t V, uint8_t K, bool directed){
  TopKNaive ksp1(es, K, directed);
  TopKPrunedLandmarkLabeling ksp2;
  ASSERT_TRUE(ksp2.ConstructIndex(es, K, directed));
  
  for (size_t u = 0; u < V; u++){
    for (size_t v = 0; v < V; v++){
      vector<int> d1;
      vector<int> d2;
      
      ksp1.KDistanceQuery(u, v, K, d1);
      ksp2.KDistanceQuery(u, v, K, d2);

      if (d1 != d2){
        cerr << u << " " << v << " " << (int)K << endl;
        for (size_t i = 0; i < d1.size(); i++) cerr << " " <<  d1[i];
        cerr << "d1:";
        cerr << endl;

        cerr << "d2: ";
        for (size_t i = 0; i < d2.size(); i++) cerr << " " << d2[i];
        cerr << endl;
      }
      ASSERT_EQ(d1, d2);
    }
  }
}

vector<pair<int, int> > GenerateLine(int V){
  vector<pair<int, int> > es;
  for (int i = 0; i < V - 1; i++){
    es.push_back(pair<int, int>(i, i + 1));
  }
  return es;
}

vector<pair<int, int> > GenerateRing(int V){
  vector<pair<int, int> > es;
  for(int i = 0; i < V; i++){
    es.push_back(pair<int, int>(i, (i + 1) % V));
  }
  return es;
}

vector<pair<int, int> > GenerateRandom(int V, int seed){
  srand(seed);
  vector<pair<int, int> > es;
        
  for(int i = 0; i < V; i++){
    for(int j = i + 1; j < V; j++){
      double x = rand() / (double)RAND_MAX;
      if(x < 0.5){
        es.push_back(pair<int, int>(i, j));
      }
    }
  }
  return es;
}

int ReadEdges(const char *filename, vector<pair<int, int> > &es){
  ifstream ifs(filename);
  
  if (!ifs.good()) return -1;
    
  int u, v, V = 0;
    
  while (ifs >> u >> v){
    es.push_back(make_pair(u, v));
    V = max(V, max(u, v) + 1);
  }

  ifs.close();

  return V;
}

TEST(TOPK_TEST, U_LINE){
  vector<pair<int, int> > es = GenerateLine(V);
  CheckAllVertexPairs(es, V, K, false);
}

TEST(TOPK_TEST, U_RING){
  vector<pair<int, int> > es = GenerateRing(V);  
  CheckAllVertexPairs(es, V, K, false);
}

TEST(TOPK_TEST, U_RANDOM){
  vector<pair<int, int> > es = GenerateRandom(V, 0);  
  CheckAllVertexPairs(es, V, K, false);
}

TEST(TOPK_TEST, U_HANDMADE){
  vector<pair<int, int> > es;
  int V = ReadEdges("sample/example.txt", es);
  ASSERT_TRUE(V >= 0);
    
  for (int k = 1; k < 128; k <<= 1){
    CheckAllVertexPairs(es, V, k, false);
  }
}

TEST(TOPK_TEST, D_LINE){
  vector<pair<int, int> > es = GenerateLine(V);  
  CheckAllVertexPairs(es, V, K, true);
}

TEST(TOPK_TEST, D_RING){
  vector<pair<int, int> > es = GenerateRing(V);
  TopKPrunedLandmarkLabeling ksp;
  ASSERT_FALSE(ksp.ConstructIndex(es, K, true));
}

TEST(TOPK_TEST, D_RANDOM){
  vector<pair<int, int> > es = GenerateRandom(V, 0);
  CheckAllVertexPairs(es, V, K, true);
}

TEST(TOPK_TEST, D_HANDMADE){
  vector<pair<int, int> > es;
  int V = ReadEdges("sample/example.txt", es);
  ASSERT_TRUE(V >= 0);
    
  for (int k = 1; k < 128; k <<= 1){
    CheckAllVertexPairs(es, V, k, true);
  }
}


void CheckIndexIO(const vector<pair<int, int> > &es, int V, int K, bool directed){
    
  TopKPrunedLandmarkLabeling ksp1;
  ksp1.ConstructIndex(es, K, directed);
    
  const char *index_file = "sample/index_file";
    
  {
    ofstream ofs(index_file);
    ASSERT_TRUE(ofs);
    ASSERT_TRUE(ksp1.StoreIndex(ofs));
    ofs.close();
  }
    
  ifstream ifs(index_file);
  TopKPrunedLandmarkLabeling ksp2;
  ksp2.LoadIndex(ifs);
    
  for (int u = 0; u < V; u++){
    for (int v = 0;  v < V; v++){
      vector<int> d1;
      vector<int> d2;
      ksp1.KDistanceQuery(u, v, K, d1);
      ksp2.KDistanceQuery(u, v, K, d2);
      ASSERT_EQ(d1, d2) << u << " " << v << endl;
    }
  }
}

TEST(INDEX_IO_TEST, U_LINE){
  vector<pair<int, int> > es = GenerateLine(V);;  
  CheckIndexIO(es, V, K, false);
}

TEST(INDEX_IO_TEST, U_RING){
  vector<pair<int, int> > es = GenerateRing(V);  
  CheckIndexIO(es, V, K, false);
}

TEST(INDEX_IO_TEST, U_RANDOM){
  int num_graphs = 5;
    
  for (int i = 0; i < num_graphs; i++){
    vector<pair<int, int> > es = GenerateRandom(V, i);
    CheckIndexIO(es, V, K, false);
  }
}

TEST(INDEX_IO_TEST, U_HANDMADE0){
  vector<pair<int, int> > es;
  int V = ReadEdges("sample/example.txt", es);
  ASSERT_TRUE(V >= 0);
    
  for (int k = 1; k < 128; k <<= 1){
    CheckIndexIO(es, V, k, false);
  }
}


TEST(INDEX_IO_TEST, D_LINE){
  vector<pair<int, int> > es = GenerateLine(V);;  
  CheckIndexIO(es, V, K, true);
}

TEST(INDEX_IO_TEST, D_RING){
  vector<pair<int, int> > es = GenerateRing(V);  
  CheckIndexIO(es, V, K, true);
}

TEST(INDEX_IO_TEST, D_RANDOM){
  int num_graphs = 5;
    
  for (int i = 0; i < num_graphs; i++){
    vector<pair<int, int> > es = GenerateRandom(V, i);
    CheckIndexIO(es, V, K, true);
  }
}

TEST(INDEX_IO_TEST, D_HANDMADE0){
  vector<pair<int, int> > es;
  int V = ReadEdges("sample/example.txt", es);
  ASSERT_TRUE(V >= 0);
    
  for (int k = 1; k < 128; k <<= 1){
    CheckIndexIO(es, V, k, true);
  }
}
