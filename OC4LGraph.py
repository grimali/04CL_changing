import pm4py
import pandas as pd

import numpy as np
from sklearn.neighbors import NearestNeighbors
from encode.baseline import run_onehot

from Levenshtein import distance as levenshtein_distance

from sklearn.neighbors import kneighbors_graph
import networkx as nx
from multiprocessing import Pool, cpu_count

from scipy.sparse import csr_matrix
import pywt

import numpy as np
from Levenshtein import distance as levenshtein_distance
from scipy.spatial.distance import euclidean
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from pathlib import Path

def get_transitions(dataframe):
    dfg, start_activities, end_activities = pm4py.discover_dfg(dataframe, case_id_key='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    
    activities = list(set([activity for pair in dfg.keys() for activity in pair]))
    
    # Initialize a matrix with zeros
    dfg_matrix = pd.DataFrame(0, index=activities, columns=activities)
    
    # Fill the matrix with the DFG transitions
    for (activity_from, activity_to), count in dfg.items():
        dfg_matrix.at[activity_from, activity_to] = count
    return dfg_matrix    

def get_trace_transition_matrix(dataframe):
    # Get all unique activities (nodes) in the DataFrame
    activities = list(dataframe['concept:name'].unique())

    # Group the DataFrame by cases (traces)
    grouped = dataframe.groupby("case:concept:name")
    
    transition_matrix_all = []
    case_ids = []  # To store the case names
    
    for case_id, group in grouped:
        transition_matrix = pd.DataFrame(0, index=activities, columns=activities)
        
        # Sort the group by timestamp
        group = group.sort_values(['time:timestamp'])

        # Iterate over the activities in the trace
        for i in range(len(group) - 1):
            activity_from = group.iloc[i]['concept:name']
            activity_to = group.iloc[i + 1]['concept:name']
            transition_matrix.at[activity_from, activity_to] += 1
        # Append the case ID and transition matrix to the lists
        case_ids.append(case_id)
        transition_matrix_all.append(transition_matrix.values.flatten())
    
    # Create a DataFrame with the case IDs as the first column
    transition_matrix_df = pd.DataFrame(transition_matrix_all, columns=[f"{a}->{b}" for a in activities for b in activities])
    transition_matrix_df.insert(0, "case", case_ids)
    
    return transition_matrix_df


def get_knn_distance_matrix(binary_data, n_neighbors=3, metric='euclidean'):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    binary_data.columns = binary_data.columns.astype(str)
    knn.fit(binary_data)
    # Compute the distance matrix
    distance_matrix = knn.kneighbors_graph(binary_data, mode='distance').toarray()

    # Fill the diagonal with zeros since the distance from a point to itself is zero
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

def compute_distance_matrix(data):
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = edit_distance_levenshtein(data[i], data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Distance matrix is symmetric
    
    return distance_matrix

def encode_histogram_with_haar(histogram):
    """
    Encodes a histogram using the wavelet packet transform with the Haar wavelet.
    
    Args:
    histogram (np.ndarray): The input histogram to encode.
    
    Returns:
    np.ndarray: The encoded histogram using Haar wavelet transform.
    """
    # Ensure histogram is a numpy array with float values
    data = np.array(histogram, dtype=float)
    
    # Create a wavelet packet object using the Haar wavelet
    wavelet = pywt.Wavelet('haar')
    packet = pywt.WaveletPacket(data, wavelet, mode='symmetric', maxlevel=2)
    
    # Fetch the coefficients
    coeffs = packet.get_level(2, order='freq')  # Get all nodes of level 2 (could adjust level as needed)
    
    # Flatten the coefficients into a single array
    encoded_data = np.hstack([node.data for node in coeffs])
    
    return encoded_data

def extract_knn_graph(distance_matrix, n_neighbors):
    """
    Extract the k-NN graph from a distance matrix.
    
    Parameters:
    distance_matrix (np.array): Distance matrix.
    n_neighbors (int): Number of neighbors to use for KNN.
    
    Returns:
    networkx.Graph: The k-NN graph.
    """
    # Create a k-NN graph from the distance matrix
    knn_graph = kneighbors_graph(distance_matrix, n_neighbors=n_neighbors, metric='precomputed', mode='connectivity',)
    
    # Convert the sparse matrix to a networkx graph
    G = nx.Graph(csr_matrix(knn_graph))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True) #per creare le componenti giganti
    G0 = G.subgraph(Gcc[0])
    return G
    


def edit_distance_levenshtein(vec1, vec2):
    """
    Compute the Levenshtein distance between two sequences using the python-Levenshtein library.
    
    Parameters:
    vec1 (list or str): First vector (can be numeric or categorical, but should be converted to string if numeric).
    vec2 (list or str): Second vector (same as vec1).
    
    Returns:
    int: The Levenshtein distance between vec1 and vec2.
    """
    return Levenshtein.distance(vec1, vec2)

def edit_distance(vec1, vec2):
    """
    Compute the edit distance between two non-binary vectors with reduced space complexity.
    Optimized for equal-length vectors using numpy for faster array operations.
    
    Parameters:
    vec1 (np.array): First vector (can be numeric or categorical).
    vec2 (np.array): Second vector (can be numeric or categorical).
    
    Returns:
    int: The edit distance between vec1 and vec2.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length.")

    n = len(vec1) + 1
    prev_row = np.arange(n)
    current_row = np.zeros(n, dtype=int)

    for i in range(1, n):
        current_row[0] = i
        current_row[1:] = np.minimum(
            prev_row[1:] + 1,
            np.minimum(current_row[:-1] + 1, prev_row[:-1] + (vec1[i-1] != vec2))
        )
        prev_row[:] = current_row

    return prev_row[-1]

def get_bins_split_by_max_case(df, time_column, case_column):
    # Ensure the time_column is in datetime format

    df[time_column] = pd.to_datetime(df[time_column])
    
    # Calculate the duration of each case by finding the max and min times for each case
    grouped = df.groupby(case_column)[time_column].agg([min, max])
    grouped['duration'] = (grouped['max'] - grouped['min']).dt.total_seconds()  # Calculate duration in seconds
    
    # Find the longest case duration
    longest_case_duration = grouped['duration'].max()

    # Determine the total duration of the dataset from the earliest to the latest timestamp
    total_duration = (df[time_column].max() - df[time_column].min()).total_seconds()

    # Calculate how many intervals of size 2 * longest_case_duration fit into the total duration
    if longest_case_duration == 0:
        return 0  # Prevent division by zero if all cases have identical start and end times
    
    num_intervals = int(total_duration / (4*longest_case_duration))
    
    return num_intervals

def _pairwise_distances(args):
        i, s_act_i, s_tra_i, vec_time_i, act_strings, tra_strings, time_vectors, w_act, w_tra, w_time = args
        row = np.zeros(len(act_strings))
        for j in range(i + 1, len(act_strings)):
            dist_act = levenshtein_distance(s_act_i, act_strings[j])
            dist_tra = levenshtein_distance(s_tra_i, tra_strings[j])
            dist_time = euclidean(vec_time_i, time_vectors[j])
            dist = w_act * dist_act + w_tra * dist_tra + w_time * dist_time
            row[j] = dist
        return i, row

def compute_distance_matrix_opt(act_features, tra_features, time_features, w_act=1.0, w_tra=1.0, w_time=1.0):
        
        cases = act_features.index
        n_samples = len(cases)
        distance_matrix = np.zeros((n_samples, n_samples))

        # Preprocess for faster access
        act_strings = act_features.astype(str).agg(''.join, axis=1).tolist()
        tra_strings = tra_features.astype(str).agg(''.join, axis=1).tolist()
        time_vectors = time_features.to_numpy()

        args_list = [
            (i, act_strings[i], tra_strings[i], time_vectors[i], act_strings, tra_strings, time_vectors, w_act, w_tra, w_time)
            for i in range(n_samples)
        ]

        with Pool(int(cpu_count()/2)) as pool:
            for i, row in tqdm(pool.imap_unordered(_pairwise_distances, args_list), total=n_samples, desc="Computing Distance Matrix"):
                distance_matrix[i] = row

        # Fill lower triangle
        i_lower = np.tril_indices(n_samples, -1)
        distance_matrix[i_lower] = distance_matrix.T[i_lower]

        return distance_matrix

def time_histogram_by_case(df, time_column, case_column, bins=16):
    """
    Extracts a histogram of time from a DataFrame's timestamp column,
    grouped by a case identifier (case:concept:name), with specified number of bins,
    using the global min and max time across the entire dataframe.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    time_column (str): The name of the column containing timestamps.
    case_column (str): The name of the column representing the case identifier.
    bins (int): The number of bins to use (default is 16).
    
    Returns:
    pd.DataFrame: A DataFrame containing the histogram counts for each case.
    """
    # Ensure the time column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Calculate the global min and max time across the entire dataframe
    min_time = df[time_column].min()
    
    # Convert timestamps to seconds (or another suitable unit like minutes)
    time_seconds = (df[time_column] - min_time).dt.total_seconds()
    
    # Define the global bin edges based on the entire dataframe's min and max time
    bin_edges_global = np.linspace(time_seconds.min(), time_seconds.max(), bins + 1)
    
    # Initialize a list to store the histogram data for each case
    histograms_data = []
    
    # Loop over each unique case
    for case in df[case_column].unique():
        # Filter the dataframe by case
        case_df = df[df[case_column] == case]
        
        # Convert case timestamps to seconds using the global min_time
        case_time_seconds = (case_df[time_column] - min_time).dt.total_seconds()
        
        # Calculate the histogram for the case using global bin edges
        hist, _ = np.histogram(case_time_seconds, bins=bin_edges_global)
        
        # Append the histogram data to the list, including the case name
        histograms_data.append([case] + hist.tolist())
        
    # Define column names, including the case column and the bin columns
    columns = ["case"] + [f'B{i+1}' for i in range(bins)]
    
    # Convert the histograms data to a DataFrame
    histograms_df = pd.DataFrame(histograms_data, columns=columns)
        
    return histograms_df



class OC4LGraph():
    def __init__(self, ocel_path, ocel_case_notion, config, k=3, n_bins_time_features=16, use_wavelet = True, plot=False):
        self.ocel_path = ocel_path
        self.ocel_case_notion = ocel_case_notion
        self.plot = plot
        self.graphs = []
        self.config = config
        self.k = k
        self.use_wavelet = use_wavelet
        self.n_bins_time_features = n_bins_time_features
        self.process()

    def load_xes(path):
        log = pm4py.read_xes(path)
        df = pm4py.convert_to_dataframe(log)
        return df
        

    def process(self):
        database_name = Path(self.ocel_path).stem
        print("Database name:", database_name)
        if self.ocel_path.endswith(".xes"):
            log = pm4py.read_xes(self.ocel_path)

            # Flatten to dataframe with common column names
            df_features = pm4py.convert_to_dataframe(log)
            df_features["case:concept:name"] = df_features["case:concept:name"].astype(str)  # Ensure strings
            df_features = df_features.filter(["case:concept:name","concept:name","time:timestamp"])
            #df_features = df_features.filter(["EventID","concept:name","time:timestamp"])
            df_features.columns = ["case:concept:name","concept:name","time:timestamp"]
            df_features["concept:name"] = (
                df_features["concept:name"]
                .astype(str)  # ensure string type
                .str.replace("-", "", regex=False)  # remove all "-"
                .str.replace("0_", "", regex=False)
                .str.replace("(", "", regex=False)
                .str.replace(")", "", regex=False)
                .str.replace(" ", "_", regex=False)  # replace " " with "_"
            )
        else:            
            if 'sqlite2' in self.ocel_path:
                print("sqlite2")
                ocel = pm4py.read_ocel2_sqlite(self.ocel_path)
            elif 'sqlite' in self.ocel_path:
                print("sqlite format assumed", self.ocel_path)
                ocel = pm4py.read_ocel_sqlite(self.ocel_path)
            else:
                print("json format assumed")
                ocel = pm4py.read_ocel2_json(self.ocel_path)
            filtered_ocel = pm4py.filter_ocel_object_attribute(ocel, 'ocel:type', [self.ocel_case_notion])
            df_filtered_ocel = filtered_ocel.get_extended_table().explode('ocel:type:' + self.ocel_case_notion).drop_duplicates().sort_values(["ocel:timestamp"])
            df_features = df_filtered_ocel.filter(["ocel:eid","ocel:activity","ocel:timestamp"])
            df_features.columns = ["case:concept:name","concept:name","time:timestamp"]
            df_features["case:concept:name"] = df_features["case:concept:name"].explode()
        
        print("Dataframe shape:", df_features.shape)

        self.config["vector_size"] = df_features["concept:name"].unique()
        act_features = run_onehot(self.config, log=df_features)
        act_features.columns = ["case"] + list(df_features["concept:name"].unique())
        tra_features = get_trace_transition_matrix(df_features)

        print('Bins for time features:', self.n_bins_time_features)
        if self.n_bins_time_features == 0:
           self.n_bins_time_features = get_bins_split_by_max_case(df_features, "time:timestamp", "case:concept:name")
        time_features = time_histogram_by_case(df_features, "time:timestamp", "case:concept:name", bins=self.n_bins_time_features)

        print('Wavelets', self.n_bins_time_features)
        if self.use_wavelet:
            time_features_w = time_features.iloc[:, 1:].apply(lambda row: encode_histogram_with_haar(row.values), axis=1)
            num_columns = len(time_features_w.iloc[0])
            wavelet_feature_columns = [f"W_{i}" for i in range(num_columns)]
            if not isinstance(time_features_w, pd.DataFrame):
               time_features_w = pd.DataFrame(time_features_w.tolist(), columns=wavelet_feature_columns)

            time_features_w = time_features_w.round(2)
            #time_features_w = pd.DataFrame(time_features_w.tolist(), columns=wavelet_feature_columns).round(2)
            time_features = pd.concat([time_features["case"], time_features_w], axis=1)

        # Ensure the "case" column is set as the index for all DataFrames
        act_features.set_index("case", inplace=True)
        tra_features.set_index("case", inplace=True)
        time_features.set_index("case", inplace=True)

        print('Dist Matrix', self.n_bins_time_features)
        distance_matrix = compute_distance_matrix_opt(
            act_features * 1,
            tra_features * 1,
            time_features * 2
        )
        
        labels = act_features.index
        str_labels = [str(l) for l in labels]
        G = extract_knn_graph(distance_matrix, n_neighbors=self.k)
        #self.graphs.append(G) #Graph without node labels
        
        #nx.set_node_attributes(G, labels, "Node")
        G_relabelled = nx.relabel_nodes(G, dict(zip(G.nodes, str_labels)), copy=True)
        self.graphs.append(G_relabelled)

        #time_label = time_features.apply(lambda row: ''.join(row.astype(str)), axis=1)
        #G_relabelled_time = nx.relabel_nodes(G, dict(zip(G.nodes, time_label)), copy=True)
        #self.graphs.append(G_relabelled_time)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
