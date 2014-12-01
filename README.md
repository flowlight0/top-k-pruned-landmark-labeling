Top-K Pruned Landmark Labeling
==============================
Top-K Pruned Landmark Labeling is a fast algorithm for answering top-k distance query on real-world networks, such as social networks and web graphs.

## Usage 
### From CUI Interface

    $ make
    $ bin/construct_index sample/example.txt 16 0 sample/index_file
    $ bin/k_distance sample/index_file <<< "9 12"
    16 1 2 3 3 3 3 3 3 3 3 4 4 4 4 4 4
    
### From Your Program
    
    TopKPrunedLandmarkLabeling kpll;
    kpll.ConstructIndex(edge_list, K, false);   // if a graph is undirected.
    kpll.ConstructIndex(edge_list, K, true);    // if a graph is directed.
    
    vector<int> k_distance;
    kpll.KDistanceQuery(2, 3, k_distance);      // top-k distance is stored in k_distance.
    kpll.StoreIndex(index_file);                // store an index in index_file.

## Reference
Takuya Akiba, Takanori Hayashi, Nozomi Nori, Yoichi Iwata and Yuichi Yoshida,  **Efficient Top-k Shortest-Path Distance Queries on Large Networks by Pruned Landmark Labeling**.
In *AAAI 2015*, to appear. 
