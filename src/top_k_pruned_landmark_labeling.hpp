#ifndef TOP_K_PRUNED_LANDMARK_LABELING_H
#define TOP_K_PRUNED_LANDMARK_LABELING_H

#include <vector>
#include <sys/time.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <limits>

class TopKPrunedLandmarkLabeling{
  
  struct dist_array_t{
    static const size_t width = 4;
    static const size_t size  = 1 << width;
    static const size_t mask  = size - 1;
    uint8_t* addr;
    uint16_t offset[size];
  };
  
  struct index_t{
    uint32_t     *label;
    uint32_t      length;
    uint8_t      *offset;
    dist_array_t *d_array;
    inline uint8_t *GetDistArray(size_t i) const ;
    inline uint8_t DistArrayLength(size_t i) const;    
    inline bool ReAllocDistArray(size_t i, size_t nmemb);
  };
  
public:
  TopKPrunedLandmarkLabeling() :
    V(0), K(0), directed(0), loop_count_time(0), indexing_time(0){
    index[0] = index[1] = NULL;
  }
  
  ~TopKPrunedLandmarkLabeling();
  
  int KDistanceQuery(int s, int t, uint8_t k, std::vector<int> &ret);
  int KDistanceQuery(int s, int t, std::vector<int> &ret){ return KDistanceQuery(s, t, K, ret); }
  int KDistanceQuery(int s, int t, uint8_t k);
  int KDistanceQuery(int s, int t){ return KDistanceQuery(s, t, K); }
  
  bool   ConstructIndex(const std::vector<std::pair<int, int> > &es, size_t K, bool directed);
  bool   StoreIndex(std::ofstream &ofs);
  bool   StoreIndex(const char *file);
  bool   LoadIndex(std::ifstream &ifs);
  bool   LoadIndex(const char *file);
  
  double IndexingTime()  const { return indexing_time; }
  double LoopCountTime() const { return loop_count_time; }
  size_t NumOfVertex ()      { return V; }
  size_t IndexSize();
  double AverageLabelSize();
  
private:
  size_t V;
  uint8_t K;
  bool directed;
  // We assume that the diameter of a given network is less than 128.    
  static const uint8_t INF8; 
  
  double loop_count_time;
  double indexing_time;
  
  std::vector<uint32_t> alias;
  std::vector<std::vector<uint32_t> > graph[2];
  std::vector<std::vector<uint8_t> > loop_count;
  // index[0] corresponds to L_in. index[1] corresponds to L_out
  index_t *index[2];
  
  std::vector<bool>     tmp_pruned;
  std::vector<uint32_t> tmp_offset;
  std::vector<uint32_t> tmp_count;
  std::vector<uint8_t>  tmp_dist_count[2];
  std::vector<uint8_t>  tmp_s_offset;
  std::vector<std::vector<uint8_t> > tmp_s_count;
  
  void Init();
  void Free();
  bool Labeling();
  void CountLoops(uint32_t s, bool &status);
  void PrunedBfs (uint32_t s, bool dir, bool &status);
  inline void SetStartTempVars(uint32_t s, bool rev);
  inline void ResetTempVars(uint32_t s, const std::vector<uint32_t> &updated, bool rev);
  inline bool Pruning(uint32_t v, uint8_t d, bool rev);  
  inline void AllocLabel(uint32_t v, uint32_t s, uint8_t d, uint8_t dc, bool rev);
  inline void ExtendLabel(uint32_t v, uint32_t s, uint8_t d, uint8_t dc, bool rev);
};


#endif /* TOP_K_PRUNED_LANDMARK_LABELING_H */


