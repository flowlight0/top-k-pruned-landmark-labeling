Top-K Pruned Landmark Labeling
==============================
Top-K Pruned Landmark Labeling is a fast algorithm for answering top-k distance query on real-world networks, such as social networks and web graphs. 

## Usage 
### From CUI Interface

    $ make
    $ bin/construct_index sample/example.txt 16 0 sample/index_file     #compute an index that answer top 16 shortest distance on an undirected graph specified by sample/example.txt.
    $ bin/k_distance sample/index_file <<< "9 12"
    16 1 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4

* Execute `make` to build programs.
* Execute `bin/construct_index` to construct an index from a given graph. The constructed index is stored in index_file.
* Execute `bin/k_distance` to compute top-k distance using the index generated by `bin/construct_index`.
  For each vertex pair s and t, it output one line. First number l in the line means min{K, #general paths between s and t}.
  Following l numbers are top-l distance in the nondecreasing order.
    
### From Your Program
    
    TopKPrunedLandmarkLabeling kpll;
    kpll.ConstructIndex(edge_list, K, false);   // if a graph is undirected.
    kpll.ConstructIndex(edge_list, K, true);    // if a graph is directed.
    
    vector<int> k_distance;
    kpll.KDistanceQuery(2, 3, k_distance);      // top-k distance is stored in k_distance.
    kpll.StoreIndex(index_file);                // store an index in index_file.

* Call `ConstructIndex` to construct an index from a given edge list.
* Call `KDistanceQuery` to answer the top-k distance query.
* Call `StoreIndex` to store the constructed index.
* 
### Warning
Since the dialeter of real-world graphs, such as social networks and web graphs, are really small, out implementation exploit this property to save the memory usage. Therefore, when a diameter of a given graph is too large, it may return a wrong answer.

## Reference
Takuya Akiba, Takanori Hayashi, Nozomi Nori, Yoichi Iwata, and Yuichi Yoshida,  **Efficient Top-k Shortest-Path Distance Queries on Large Networks by Pruned Landmark Labeling**.
In *AAAI 2015*, to appear. 
