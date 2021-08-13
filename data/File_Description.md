###  Folder : graph_data
-  `graph.edgelist`:
    a list of all edges of the build graph
    
    format: edge_id, src_node_id, dst_node_id
    
    !!! this is the file which we used to rebuild the graph everytime

-  `id_node_map.pickle`:
    (graph node id : node domain) of the built graph, e.g, {0:'fb.com'}


-  `id_edge_map_list.pickle`:
    a list of edge of the built graph
    
    example: {'id': 284, 'mongo_id': '5ea24047853cdbecc9c4b887', 'site_id': '5', 'site': 'qq.com', 'src': 'www.qq.com', 'dst': 'mat1.gtimg.com', 'url': 'https:...', 'track': '0'}
    
    note: `id` is the edge id in the graph
 



###  Folder : feat_data/raw_onehot_lists
-  `unique_methods`
-  `unique_types`
-  `unique_req_headers`
-  `unique_resp_headers`
-  `track_req_headers`
-  `track_resp_headers`
-  `char_Embeddings.npy`: derived by Google 1b-lm, contains size of 16 embedding for each char

   

