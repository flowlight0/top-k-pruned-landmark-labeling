#include "top_k_pruned_landmark_labeling.hpp"
#include <iostream>
#include <climits>
#include <xmmintrin.h>
#include <cassert>
#include <cstdlib>
#include <queue>
#include <algorithm>
#include <memory.h>
using namespace std;

const uint8_t TopKPrunedLandmarkLabeling::INF8 = std::numeric_limits<uint8_t>::max() / 2;

template <typename T> inline bool ReAlloc(T*& ptr, size_t nmemb){
  ptr = (T*)realloc(ptr, nmemb * sizeof(T));
  return ptr != NULL;
}

template <typename T> inline void EraseVector(std::vector<T> &vec){
  std::vector<T>().swap(vec);
}

  
double GetCurrentTimeSec(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ラベルiの距離配列の先頭アドレスの取得
inline uint8_t* TopKPrunedLandmarkLabeling::index_t::
GetDistArray(size_t i) const {
  size_t off = i & dist_array_t::mask;      
  const dist_array_t &dn = d_array[i >> dist_array_t::width];
  return dn.addr + (off == 0 ? 0 : dn.offset[off - 1]);
}

// ラベルiの距離配列の長さ
inline uint8_t TopKPrunedLandmarkLabeling::index_t::
DistArrayLength(size_t i) const {
  size_t off = i & dist_array_t::mask;      
  const dist_array_t &da = d_array[i >> dist_array_t::width];
  return off == 0 ? da.offset[0] : da.offset[off] - da.offset[off - 1];
}

// ラベルiの距離配列の要素数がnmembに領域を割り当て
// ラベルiに対する処理後j < iとなるラベルjに対して呼び出されない。

inline bool TopKPrunedLandmarkLabeling::index_t::
ReAllocDistArray(size_t i, size_t nmemb){
  size_t off = i & dist_array_t::mask;
  dist_array_t &da = d_array[i >> dist_array_t::width];
  size_t n = nmemb + (off == 0 ? 0 : da.offset[off-1]);
  
  if(da.offset[off] < n){
    da.offset[off] = n;
    return ReAlloc(da.addr, n);
  }else{
    assert(da.addr[da.offset[off] - 1] == std::numeric_limits<uint8_t>::max() / 2);
    return true;
  }
}

// 辺集合とKを受け取りグラフの頂点を次数順に並び替えた後Indexing

bool TopKPrunedLandmarkLabeling::
ConstructIndex(const vector<pair<int, int> > &es, size_t K, bool directed){
  Free();

  this->V = 0;
  this->K = K;
  this->directed = directed;
  
  for (size_t i = 0; i < es.size(); i++){
    V = std::max(V, (size_t)std::max(es[i].first, es[i].second) + 1);
  }
  
  for (int dir = 0; dir < 1 + directed; dir++){
    graph[dir].resize(V);
  }
    
  // renaming
  {
    vector<std::pair<int, int> > deg(V, std::make_pair(0, 0));
        
    for (size_t i = 0; i < V; i++) deg[i].second = i;
        
    for (size_t i = 0; i < es.size(); i++){
      deg[es[i].first ].first++;
      deg[es[i].second].first++;
    }
        
    sort(deg.begin(), deg.end(), greater<pair<int, int> >());
    alias.resize(V);
        
    for (size_t i = 0; i < V; i++) alias[deg[i].second] = i;
        
    for (size_t i = 0; i < es.size(); i++){
      graph[0][alias[es[i].first]].push_back(alias[es[i].second]);
      
      if (directed){
        graph[1][alias[es[i].second]].push_back(alias[es[i].first]);
      } else {
        graph[0][alias[es[i].second]].push_back(alias[es[i].first]);
      }
    }
  }
  
  Init();

  return Labeling();
}

TopKPrunedLandmarkLabeling::
~TopKPrunedLandmarkLabeling(){
  Free();
}


int TopKPrunedLandmarkLabeling::
KDistanceQuery(int s, int t, size_t k){
  vector<int> dists;
  return KDistanceQuery(s, t, k, dists);
}

int TopKPrunedLandmarkLabeling::
KDistanceQuery(int s, int t, size_t k, vector<int> &ret){
  ret.clear();
  s = alias[s];
  t = alias[t];
  size_t pos1 = 0;
  size_t pos2 = 0;

  vector<int> count(30, 0);
  // cerr << directed << " " << s << " " << t << endl;
  const index_t &ids = index[directed][s];
  const index_t &idt = index[0][t];
  
  uint32_t *ls = ids.label;
  uint32_t *lt = idt.label;
  
  for (;;){
    if (ls[pos1] == lt[pos2]){
      uint32_t W = ls[pos1];
      if (W == V) break;
            
      uint8_t *dcs = ids.GetDistArray(pos1);
      uint8_t *dct = idt.GetDistArray(pos2);
            
      for (size_t i = 0; dcs[i] != INF8; i++){
        for (size_t j = 0; dct[j] != INF8; j++){
          for (size_t m = 0; m < loop_count[W].size(); m++){
            uint8_t d_tmp = ids.offset[pos1] + idt.offset[pos2] + i + j + m;
            uint8_t c_tmp = loop_count[W][m] - (m ? loop_count[W][m-1] : 0);
            if (count.size() <= d_tmp) count.resize(d_tmp + 1, 0);
            count[d_tmp] += (int)dcs[i] * dct[j] * c_tmp;
          }
        }
      }
      pos1++, pos2++;
    } else {
      if (ls[pos1] < lt[pos2]){
        pos1++;
      } else {
        pos2++;
      }
    }
  }
  
  for (size_t i = 0; i < count.size(); i++){
    while (ret.size() < k && count[i]-- > 0){
      ret.push_back(i);
    }
  }

  return ret.size() < k ? INT_MAX : 0;
}

// ラベル等に使用したメモリ量を計算
size_t TopKPrunedLandmarkLabeling::
IndexSize(){
  size_t sz = 0;
    
  sz += sizeof(int) * alias.size();		      // alias
    
  for(size_t v = 0; v < V; v++){
    sz += sizeof(uint8_t ) * loop_count[v].size(); // loopcount

    sz += sizeof(uint32_t) * graph[0][v].size();	  // graph
    sz += sizeof(uint32_t) * graph[1][v].size();	 
  }
  
  // index's size
  for (int dir = 0; dir < 1 + directed; dir++){
    for(size_t v = 0; v < V; v++){
      sz += sizeof(index_t);
      sz += index[dir][v].length * sizeof(uint32_t); // index[i].label
      sz += index[dir][v].length * sizeof(uint8_t ); // index[i].offset
        
      // index[i].d_array
      for(int pos = 0; index[dir][v].label[pos] != V; pos++){
        if((pos & dist_array_t::mask) == 0){
          sz += sizeof(dist_array_t);
        }
        int j = 0;
        do{
          sz += sizeof(uint8_t);	
        }while(index[dir][v].GetDistArray(pos)[j++] != INF8);
      }
    }
  }
  return sz;
}

double TopKPrunedLandmarkLabeling::
AverageLabelSize(){
  double total = 0;
  for (int dir = 0; dir < 1 + directed; dir++){
    for (size_t v = 0; v < V; v++){
      total += index[dir][v].length;
    }
  }
  return total / V;
}


// 一時配列とラベルの初期化
void TopKPrunedLandmarkLabeling::
Init(){
  tmp_pruned.resize(V, false);
  tmp_offset.resize(V, INF8);
  tmp_count .resize(V, 0);
  tmp_s_offset.resize(V, INF8); tmp_s_offset.push_back(0);
  tmp_s_count .resize(V);
  for (int j = 0; j < 2; j++) tmp_dist_count[j].resize(V, 0);
    
  loop_count.resize(V);
  
  for (int dir = 0; dir < 1 + directed; dir++){
    index[dir] = (index_t*) malloc(sizeof(index_t) * V);
        
    for (size_t v = 0; v < V; v++){
      index[dir][v].label    = (uint32_t*)malloc(sizeof(uint32_t) * 1);
      index[dir][v].label[0] = V;
      index[dir][v].length   = 0;
      index[dir][v].offset   = NULL;
      index[dir][v].d_array  = NULL;
    }
  }
}

void TopKPrunedLandmarkLabeling::
Free(){
  for (int dir = 0; dir < 1 + directed; dir++){
    for (size_t v = 0; v < V; v++){
      index_t &idv = index[dir][v];
        
      free(idv.label);
        
      if (idv.offset != NULL) free(idv.offset);
      
      if (idv.d_array != NULL){
        for (size_t i = 0; i < idv.length - 1; i += dist_array_t::size){
          free(idv.d_array[i / dist_array_t::size].addr);
        }
        free(idv.d_array);
      }
    }
    free(index[dir]);
  }
}

bool TopKPrunedLandmarkLabeling::
Labeling(){
  
  bool status = true;
  
  loop_count_time = -GetCurrentTimeSec();
  for(size_t v = 0; v < V; v++){
    CountLoops(v, status);
  }
  loop_count_time += GetCurrentTimeSec();
  
  indexing_time = -GetCurrentTimeSec();
  for(size_t v = 0; v < V; v++){

    // compute L_in
    PrunedBfs(v, false, status);
        
    if (directed){
      //  compute L_out
      PrunedBfs(v, true, status);
    }
  }
  indexing_time += GetCurrentTimeSec();

  return status;
}

void TopKPrunedLandmarkLabeling::
CountLoops(uint32_t s, bool &status){
  size_t  count = 0;
  int     curr  = 0;
  int     next  = 1;
  uint8_t dist  = 0;
  
  std::queue<uint32_t> node_que[2];
  vector<uint32_t>     updated;
  const vector<vector<uint32_t> > &fgraph = graph[0];
    
  node_que[curr].push(s);
  updated.push_back(s);
  tmp_dist_count[curr][s] = 1;

  // 2つのqueueを距離が１増えるごとに入れ替えて探索していく。
  for (;;){
    if (dist == INF8 && status){
      cerr << "Warning: Self loops become too long." << endl;
      status = false;
    }
        
    while (!node_que[curr].empty() && count < K){
      uint32_t v = node_que[curr].front(); node_que[curr].pop();
      uint8_t  c = tmp_dist_count[curr][v]; // 始点からvに距離distで来るパスの数
      tmp_dist_count[curr][v] = 0;
      if (c == 0) continue;
      
      if (v == s){
        loop_count[s].resize(dist + 1, 0);
        loop_count[s][dist] += c;
        count += c;
      }
      
      for (size_t i = 0; i < fgraph[v].size(); i++){
        uint32_t to = fgraph[v][i];
                
        if (tmp_count[to] == 0){
          updated.push_back(to);
        }
	
        if (to >= s && tmp_count[to] < K){
          tmp_count[to] += c;
          node_que[next].push(to);
          tmp_dist_count[next][to] += c;
        }
      }
    }
    if(node_que[next].empty() || count >= K) break;
    swap(curr, next);
    dist++;
  }
  
  for(size_t i = 1; i < loop_count[s].size(); i++){
    loop_count[s][i] += loop_count[s][i-1];
  }
  assert(loop_count[s][0] == 1);
  ResetTempVars(s, updated, false);
}

void TopKPrunedLandmarkLabeling::
PrunedBfs(uint32_t s, bool rev, bool &status){
  SetStartTempVars(s, rev);
  
  int     curr = 0;
  int     next = 1;
  uint8_t dist = 0;
    
  std::queue<uint32_t> node_que[2];
  vector<uint32_t>     updated;
    
  node_que[curr].push(s);
  tmp_dist_count[curr][s] = 1;
  updated.push_back(s);
  const vector<vector<uint32_t> > &graph_ = graph[rev];
    
  for (;;){
    if (dist == INF8 && status){
      cerr << "Warning: Distance from a source node becomes too long." << endl;
      status = false;
    }
    
    
    while (!node_que[curr].empty()){
            
      uint32_t v = node_que[curr].front(); node_que[curr].pop();
      uint8_t  c = tmp_dist_count[curr][v];
      tmp_dist_count[curr][v] = 0;
            
      if(c == 0 || tmp_pruned[v]) continue;
      tmp_pruned[v] = Pruning(v, dist, rev);
      // cerr << "Pruning done" << endl;
            
      if(tmp_pruned[v]) continue;
            
      if(tmp_offset[v] == INF8){
        // Make new label for a node v
        tmp_offset[v] = dist;    
        AllocLabel(v, s, dist, c, rev);
      }else{
        // assert(s != 3);
        ExtendLabel(v, s, dist, c, rev);
      }
            
      for(size_t i = 0; i < graph_[v].size(); i++){
        uint32_t to  = graph_[v][i];
        if(tmp_count[to] == 0){
          updated.push_back(to);
        }
                
        if(to > s && tmp_count[to] < K){
          tmp_count[to] += c;
          node_que[next].push(to);
          tmp_dist_count[next][to] += c;
        }
      }
    }
        
    if (node_que[next].empty()) break;
    swap(curr, next);
    dist++;
  }
  // cerr <<"#visited nodes: " << num_of_labeled_vertices[s] << endl;
  ResetTempVars(s, updated, rev);
};

// 累積和計算のための配列tmp_s_countを計算し枝刈りを高速化
inline void TopKPrunedLandmarkLabeling::
SetStartTempVars(uint32_t s, bool rev){
  const index_t &ids = index[directed && !rev][s];
    
  for(size_t pos = 0; ids.label[pos] != V; pos++){
    int w = ids.label[pos];
    tmp_s_offset[w] = ids.offset[pos];
    
    vector<uint8_t> tmp_v;
    for(size_t i = 0; ids.GetDistArray(pos)[i] != INF8; i++){
      tmp_v.push_back(ids.GetDistArray(pos)[i]);
    }
    tmp_s_count[w].resize(tmp_v.size() + loop_count[w].size() - 1, 0);

    for(size_t i = 0; i < tmp_v.size(); i++){
      for(size_t j = 0; j < loop_count[w].size(); j++){
        tmp_s_count[w][i+j] += tmp_v[i] * loop_count[w][j];
      }
    }
  }
}

// 更新されたテーブルをもとに戻す
inline void TopKPrunedLandmarkLabeling::
ResetTempVars(uint32_t s, const vector<uint32_t> &updated, bool rev){
  // cerr << rev << " " << s << " " << V << endl;
  const index_t &ids = index[directed && !rev][s];

  // cerr << ids.length << " " << ids.label[0] << endl;
  for(size_t pos = 0; ids.label[pos] != V; pos++){
    int w = ids.label[pos];
    tmp_s_offset[w] = INF8;
    tmp_s_count[w].clear();
  }
  
  for(size_t i = 0; i < updated.size(); i++){
    tmp_count [updated[i]] = 0;
    tmp_offset[updated[i]] = INF8;
    tmp_pruned[updated[i]] = false;
    for(int j = 0; j < 2; j++) tmp_dist_count[j][updated[i]] = 0;
  }
}


// bfsの始点sからvへの距離d以下のパスの個数がK個以上なら枝刈り
inline bool TopKPrunedLandmarkLabeling::
Pruning(uint32_t v,  uint8_t d, bool rev){
  const index_t &idv = index[rev][v];
    
  _mm_prefetch(idv.label , _MM_HINT_T0);
  _mm_prefetch(idv.offset, _MM_HINT_T0);
    
  size_t pcount = 0;

  // cerr << "Pruning start" << endl;
  for (size_t pos = 0;; pos++){
    uint32_t w = idv.label[pos];
    // cerr << "label: " << w << " " << (int)tmp_s_offset[w] << endl;
        
    if (tmp_s_offset[w] == INF8) continue;
    if (w == V) break;
        
    const vector<uint8_t> &dcs = tmp_s_count[w];
    const uint8_t         *dcv = idv.GetDistArray(pos);
        
    int l = dcs.size() - 1;
    int c = d - tmp_s_offset[w] - idv.offset[pos];
        
    // tmp_s_countテーブルを利用してループを１重に
    for (int i = 0; i <= c && dcv[i] != INF8; i++){
      pcount += (int)dcs[std::min(c - i, l)] * dcv[i];
    }
    
    if (pcount >= K) return true;
  }
  return false;
}

inline void TopKPrunedLandmarkLabeling::
AllocLabel(uint32_t v, uint32_t start, uint8_t dist, uint8_t count, bool dir){
  index_t &idv = index[dir][v];
    
  size_t size = ++idv.length;
  size_t last = size - 1;
  
  ReAlloc(idv.label, size + 1);
  idv.label[last] = start;
  idv.label[size] = V;
  
  ReAlloc(idv.offset, size);
  idv.offset[last] = dist;
  
  if (last % dist_array_t::size == 0){
    int ds = (size + dist_array_t::size - 1) / dist_array_t::size;
    int dl = ds - 1;
    ReAlloc(idv.d_array, ds);
    idv.d_array[dl].addr = NULL;
    memset(idv.d_array[dl].offset, 0, sizeof(idv.d_array[dl].offset));
  }
    
  idv.ReAllocDistArray (last, 2);
  // cerr << (unsigned long long )idv.GetDistArray(last) << " " << (int)idv.DistArrayLength(last) << endl;
  idv.GetDistArray(last)[0] = count;
  idv.GetDistArray(last)[1] = INF8;
}

inline void TopKPrunedLandmarkLabeling::
ExtendLabel(uint32_t v, uint32_t start, uint8_t dist, uint8_t count, bool dir){
  index_t &idv = index[dir][v];
    
  assert(idv.length > 0);
  size_t last     = idv.length - 1;
    
  assert(idv.DistArrayLength(last) > 0);
  size_t cur_size = idv.DistArrayLength(last);

  assert(dist >= tmp_offset[v]);
  size_t new_size = dist - tmp_offset[v] + 2;
  
  assert(idv.label[last] == start);
    
  if (new_size > cur_size){
    idv.ReAllocDistArray(last, new_size);

    assert(idv.GetDistArray(last)[cur_size - 1] == INF8);
    for (size_t pos = cur_size - 1; pos < new_size; pos++){
      idv.GetDistArray(last)[pos] = 0;
    }
    idv.GetDistArray(last)[new_size-1] = INF8;
  }
  idv.GetDistArray(last)[new_size - 2] += count;
}


bool TopKPrunedLandmarkLabeling::StoreIndex(ofstream &ofs){
#define WRITE_BINARY(value) (ofs.write((const char*)&(value), sizeof(value)))

  // number of vertices V, parameter K
  WRITE_BINARY(V);
  WRITE_BINARY(K);
  WRITE_BINARY(directed);
    
  // mapping  between id in original graph and constructed index
  for (size_t v = 0; v < V; v++){
    WRITE_BINARY(alias[v]);
  }
  
  // loop count index
  for (size_t v = 0; v < V; v++){
    size_t length = loop_count[v].size();
    WRITE_BINARY(length);
    for (size_t i = 0; i < loop_count[v].size(); i++){
      WRITE_BINARY(loop_count[v][i]);
    }
  }

  for (int dir = 0; dir < 1 + directed; dir++){
    for (size_t v = 0; v <  V; v++){
      const index_t &idv = index[dir][v];

      assert(idv.length > 0);
      WRITE_BINARY(idv.length);
            
      assert(idv.label[idv.length] == V);
      for (size_t i = 0; i < idv.length; i++){
        WRITE_BINARY(idv.label[i]);
      }

      for (size_t i = 0; i < idv.length; i++){
        WRITE_BINARY(idv.offset[i]);
      }
        
      for (size_t i = 0; idv.label[i] != V;){
        dist_array_t da = idv.d_array[i / dist_array_t::size];
            
        size_t ni = i;
        size_t offset = 0;
        while (ni < i + dist_array_t::size && idv.label[ni] != V){
          size_t j = 0;
          do {
            WRITE_BINARY(idv.GetDistArray(ni)[j]);
            offset++;
          } while (idv.GetDistArray(ni)[j++] != INF8);
          assert(offset == da.offset[ni & dist_array_t::mask]);
          
          ni++;
        }
        i = ni;
      }
    }
  }
  return ofs.good();
}

bool TopKPrunedLandmarkLabeling::StoreIndex(const char *file){
  ofstream ofs(file);
  bool status = StoreIndex(ofs);
  ofs.close();
  return status;
}

bool TopKPrunedLandmarkLabeling::LoadIndex(ifstream &ifs){
  Free();
  
#define READ_BINARY(value) (ifs.read((char*)&(value), sizeof(value)))
  READ_BINARY(V);
  READ_BINARY(K);
  READ_BINARY(directed);

  loop_count_time = 0;
  indexing_time   = 0;
  alias.resize(V);
    
  for (size_t v = 0; v < V; v++){
    READ_BINARY(alias[v]);
  }
    
  loop_count.resize(V);
  for (size_t v = 0; v < V; v++){
        
    size_t length;
    READ_BINARY(length);
        
    loop_count[v].resize(length);
    for (size_t i = 0; i < length; i++){
      READ_BINARY(loop_count[v][i]);
    }
  }

  for (int dir = 0; dir < 1 + directed; dir++){
    index[dir] = (index_t *)malloc(V * sizeof(index_t));

    for (size_t v = 0; v < V; v++){
      index_t &idv = index[dir][v];
      uint32_t length;
      READ_BINARY(length);
        
      assert(length > 0);
      idv.length = length;
      idv.label  = (uint32_t *) malloc(sizeof(uint32_t) * length + 1);
      idv.offset = (uint8_t *) malloc(sizeof(uint8_t) * length);
      idv.label[length] = V;

      for (size_t i = 0; i < length; i++){
        READ_BINARY(idv.label[i]);
      }

      for (size_t i = 0; i < length; i++){
        READ_BINARY(idv.offset[i]);
      }

      size_t num_da = (length + dist_array_t::size - 1) /dist_array_t::size;
      idv.d_array = (dist_array_t *)malloc(num_da * sizeof(dist_array_t));

      for (size_t i = 0; idv.label[i] != V;){
        dist_array_t &da = idv.d_array[i / dist_array_t::size];
                
        vector<uint8_t> d_count;
            
        size_t ni = i;
        while (ni < i + dist_array_t::size && idv.label[ni] != V){
          uint8_t count = 0;
                
          do {
            READ_BINARY(count);
            d_count.push_back(count);
          } while(count != INF8);
          da.offset[ni - i] = d_count.size();
          ni++;
        }
        
        da.addr = (uint8_t *)malloc(d_count.size() * sizeof(uint8_t));
                
        for (size_t j = 0; j < d_count.size(); j++){
          da.addr[j] = d_count[j];
        }
        i = ni;
      }
    }
  }
  return ifs.good();
}


bool TopKPrunedLandmarkLabeling::LoadIndex(const char *file){
  ifstream ifs(file);
  bool status = LoadIndex(file);
  ifs.close();
  return status;
}
