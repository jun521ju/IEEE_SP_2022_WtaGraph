############  Folder : graph_data
graph.edgelist
    a list of all edges of the build graph
    format: edge_id, src_node_id, dst_node_id
    !!! this is the file which we used to rebuild the graph everytime

id_node_map.pickle
    (graph node id : node domain) of the built graph. E.g, 
    format: {0:'fb.com'}
    acts like the nodes collection in mongodb


id_edge_map_list.pickle
    a list of edge of the built graph
    format: {'id': 284, 'mongo_id': '5ea24047853cdbecc9c4b887', 'site_id': '5', 'site': 'qq.com', 'src': 'www.qq.com', 'dst': 'mat1.gtimg.com', 'url': 'https:...', 'track': '0'}
    note: id is the edge id in the graph
    acts like the edges collection in mongodb



############  Folder : feat_data
subfolder: raw_onehot_lists/
    unique_methods
    unique_types
    unique_req_headers 
    unique_resp_headers
    track_req_headers
    track_resp_headers
        Above 6 files are extracted from req_data mongodb and used as one-hot
        vectors for feature extraction

    char_Embeddings.npy
        derived by Google 1b-lm, contains size 16 embedding for each char

    all_whois.csv
        whois record extracted from PETS work

