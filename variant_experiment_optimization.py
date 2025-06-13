import argparse
import networkx as nx
import numpy as np
import OC4LGraph as DGI
import clutering_graph_utils as util
from community import community_louvain
import pandas as pd
import neptune
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
import random
from deap import base, creator, tools, algorithms
import traceback


from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Create experiment pipeline

def create_experiment(ocel_path, k, w, ocel_case_notion, config_input, occ_kernel, occ_gamma, occ_nu, r):
    name = os.path.splitext(os.path.basename(ocel_path))[0]
    metrics_dir = "datasets/experiments_ai4bpm"
    metrics_path = f"{metrics_dir}/{name}_{k}_{w}_{r}_metrics.csv"
    if not os.path.exists(metrics_path):    
        dataset = DGI.OC4LGraph(
            ocel_path=ocel_path,
            k=k,
            n_bins_time_features=w,
            ocel_case_notion=ocel_case_notion,
            config=config_input,
            plot=False,
        )

        G = dataset.graphs[0]

        partition = community_louvain.best_partition(G, resolution=r)
        bridge_nodes_stats = util.analyze_bridge_nodes(G, partition)
        bridge_nodes_stats.sort_values("ExternalEdges", ascending=False, inplace=True)

        bridge_nodes_stats["ExternalEdges"] = bridge_nodes_stats["ExternalEdges"].astype(int)
        bridge_nodes_stats["DistinctExternalCommunities"] = bridge_nodes_stats["DistinctExternalCommunities"].astype(int)
        meta_graph, _ = util.create_meta_graph(G, partition, bridge_nodes_stats)

        print("Computing metrics")
        df = calcolo_metriche(meta_graph, G, partition)
        df['label'] = df.index.astype(str).str.split('_').str[0]
        #df['label'] = df.index.astype(str)

        df_new = df[[
            'information_centrality', 'load_centrality', 'degree', 'degree_mean',
            'degree_min', 'degree_centrality', 'degree_std', 'betweenness_centrality',
            'eigenvector_centrality', 'closeness_centrality', 'subgraph_centrality_exp',
            'approximate_current_flow_betweenness', 'estrada_index', 'label'
        ]]

        os.makedirs(metrics_dir, exist_ok=True)
        df_new = df_new.round(4)
        df_new.to_csv(metrics_path, index=False)
    else:
        df_new = pd.read_csv(metrics_path)

    print("Running One-Class SVM")
    return modeling_occ_v2(df_new, occ_kernel, occ_gamma, occ_nu, name, k, w, r)

# Graph-based metrics computation

def calcolo_metriche(meta_graph, G, partition):
    partition_degrees = {part: [] for part in set(partition.values())}
    for node in G.nodes():
        part = partition[node]
        partition_degrees[part].append(G.degree(node))

    mean_degrees = {part: sum(degrees) / len(degrees) for part, degrees in partition_degrees.items()}
    node_degrees = {node: G.degree(node) for node in G.nodes()}
    partition_min_degrees = {part: float('inf') for part in set(partition.values())}
    for node, part in partition.items():
        partition_min_degrees[part] = min(partition_min_degrees[part], node_degrees[node])

    partition_std_degrees = {part: [] for part in set(partition.values())}
    for node, part in partition.items():
        partition_std_degrees[part].append(node_degrees[node])
    partition_std_degrees = {part: np.std(degrees) for part, degrees in partition_std_degrees.items()}

    df_meta_graph = nx.to_pandas_adjacency(meta_graph)
    df_meta_graph['partition'] = df_meta_graph.index.map(partition)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    closeness_centrality = nx.closeness_centrality(G)
    load_centrality = nx.load_centrality(G)

    information_centrality = {}
    subgraph_centrality_exp_dict = {}
    estrada_index_dict = {}
    approx_current_flow_betweenness_dict = {}

    for community_id in set(partition.values()):
        nodes_in_community = [node for node, comm in partition.items() if comm == community_id]
        subgraph = G.subgraph(nodes_in_community)

        if nx.is_connected(subgraph):
            info_centrality = nx.information_centrality(subgraph)
            subgraph_centrality_exp_dict.update(nx.subgraph_centrality_exp(subgraph))
            estrada_index_dict[community_id] = nx.estrada_index(subgraph)
            information_centrality.update(info_centrality)
            approx_current_flow_betweenness_dict.update(nx.approximate_current_flow_betweenness_centrality(subgraph))

    df_meta_graph['degree'] = df_meta_graph.index.map(lambda node: G.degree[node])
    df_meta_graph['degree_mean'] = df_meta_graph['partition'].map(mean_degrees)
    df_meta_graph['degree_min'] = df_meta_graph['partition'].map(partition_min_degrees)
    df_meta_graph['degree_std'] = df_meta_graph['partition'].map(partition_std_degrees)
    df_meta_graph['degree_centrality'] = df_meta_graph.index.map(degree_centrality)
    df_meta_graph['betweenness_centrality'] = df_meta_graph.index.map(betweenness_centrality)
    df_meta_graph['eigenvector_centrality'] = df_meta_graph.index.map(eigenvector_centrality)
    df_meta_graph['closeness_centrality'] = df_meta_graph.index.map(closeness_centrality)
    df_meta_graph['information_centrality'] = df_meta_graph.index.map(information_centrality)
    df_meta_graph['load_centrality'] = df_meta_graph.index.map(load_centrality)
    df_meta_graph['subgraph_centrality_exp'] = df_meta_graph.index.map(subgraph_centrality_exp_dict)
    df_meta_graph['approximate_current_flow_betweenness'] = df_meta_graph.index.map(approx_current_flow_betweenness_dict)
    df_meta_graph['estrada_index'] = df_meta_graph['partition'].map(estrada_index_dict)

    return df_meta_graph

# One-Class Classification

def modeling_occ_v2(df_input, kernel='rbf', gamma='scale', nu=0.5, name=None, k=None, w=None, r=None):
    df = df_input.copy()

    if name is None:
        raise ValueError("The 'name' parameter must be provided to load labels.")

    labels_path = f"labels_ai4bpm/{name}_anomalo.csv"
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file '{labels_path}' not found.")
    
    # Load external labels
    labels_df = pd.read_csv(labels_path)
    print('labels_df of nodes', labels_df.shape)
    print('labels_df with anomalies (R)', labels_df[labels_df['label anomalo'] == 'R'].shape)
    print('Nodes that represent a community', df.shape)                                            

    column_label = 'label anomalo'
    # Assumes one label per sample in same order as df_input
    if column_label not in labels_df.columns:
        raise ValueError("The label file must contain a 'label' column.")
    

    #print(df.columns)
    df = df.rename(columns={'label': "case_id"})
    df['case_id'] = df['case_id'].astype(str)
    labels_df['case_id'] = labels_df['case_id'].astype(str)
    df = pd.merge(df, labels_df, on="case_id", how="left")
    df['label anomalo'] = df['label anomalo'].map({"R": "anomalous"}).fillna("common")
    
    if "label time" in df.columns:
        df = df.drop(columns=["label time", "label flo"])
    df['is_common'] = df['label anomalo'].apply(lambda x: 1 if x == 'common' else -1)

    print("Total Samples distribution (*****):")
    print(df.shape)
    print(df['is_common'].value_counts())

    valori_unici = df['is_common'].unique()
    if len(valori_unici) == 1:
        print(f"⚠️ Solo una classe presente ({valori_unici[0]}). Performance = 0.")
        return 0

    X = df.drop(columns=['case_id', 'is_common', 'label anomalo'])
    y = df['is_common']
    X = X.select_dtypes(include=[np.number])

    print("\n--- One-Class SVM ---")
    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    clf.fit(X)
    ocsvm_preds = clf.predict(X)

    f1 = f1_score(y, ocsvm_preds, average='weighted')
    #os.makedirs("resultados_ai4bpm", exist_ok=True)
    #df = df.round(4)
    #df.to_csv(f"resultados_ai4bpm/{name}_{k}_{w}_{r}_{kernel}_{gamma}_{nu}_f1_{np.round(f1,2)}_occ.csv", index=False)
    
    metrics = {
        "accuracy": accuracy_score(y, ocsvm_preds),
        "precision": precision_score(y, ocsvm_preds, pos_label=1, zero_division=0),
        "recall": recall_score(y, ocsvm_preds, pos_label=1, zero_division=0),
        "f1_weighted": f1_score(y, ocsvm_preds, average='weighted', zero_division=0),
        "f1_macro": f1_score(y, ocsvm_preds, average='macro', zero_division=0)
    }
    conf_matrix = confusion_matrix(y, ocsvm_preds)

    print(conf_matrix)

    # Save full dataframe with data
    os.makedirs("resultados_ai4bpm", exist_ok=True)
    df = df.round(4)
    csv_filename = f"resultados_ai4bpm/{name}_{k}_{w}_{r}_{kernel}_{gamma}_{nu}_f1_{np.round(metrics['f1_weighted'], 2)}_occ.csv"
    df.to_csv(csv_filename, index=False)

    # Save metrics as a separate CSV line (append mode)
    metrics_out = {
        "experiment": name,
        "k": k,
        "w": w,
        "r": r,
        "kernel": kernel,
        "gamma": gamma,
        "nu": nu,
        **metrics,
        "confusion_matrix": conf_matrix.tolist()
    }

    metrics_df = pd.DataFrame([metrics_out])
    metrics_path = "resultados_ai4bpm/results_metrics_log.csv"
    if not os.path.isfile(metrics_path):
        metrics_df.to_csv(metrics_path, index=False)
    else:
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)

    print("Saved:")
    print(f"📁 Full data: {csv_filename}")
    print(f"📈 Metrics log: {metrics_path}")
    
    print('f1_macro', metrics["f1_macro"])
    print("************************** \n")
    return metrics["f1_macro"]

def modeling_occ(df_input, kernel='rbf', gamma='scale', nu=0.5):
    df = df_input.copy()

    valid_labels = ['G', 'I', 'R', 'S']
    df['label'] = df['label'].apply(lambda x: x if x in valid_labels else 'common')
    df['is_common'] = df['label'].apply(lambda x: 1 if x == 'common' else -1)

    print("Label distribution:")
    print(df.shape)
    print(df['is_common'].value_counts())

    X = df.drop(columns=['label', 'is_common'])
    y = df['is_common']
    X = X.select_dtypes(include=[np.number])

    print("\n--- One-Class SVM ---")
    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    clf.fit(X)
    ocsvm_preds = clf.predict(X)

    f1 =  f1_score(y, ocsvm_preds, average='weighted')
    print('F1-Score', f1)
    return f1

# DEAP Configuration

CASE_NOTION = "patients"
CONFIG_INPUT = {
    'encoding': 'onehot',
    'aggregation': 'average',
    'embed_from': 'nodes',
    'edge_operator': 'average'
}

K_VALUES = [2, 3, 5, 10, 15, 20, 24]
#W_VALUES = [4, 8, 16, 32, 64]
W_VALUES = [64]
#R_VALUES = [0.5, 1.0]
R_VALUES = [0.3, 0.5, 1.0, 1.5]
NU_VALUES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
#GAMMA_VALUES = ['scale', 'auto']
GAMMA_VALUES = [0.01]

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("k", random.choice, K_VALUES)
toolbox.register("w", random.choice, W_VALUES)
toolbox.register("r", random.choice, R_VALUES)
toolbox.register("nu", random.choice, NU_VALUES)
toolbox.register("gamma", random.choice, GAMMA_VALUES)

# Ensure order is preserved
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [toolbox.k(), toolbox.w(), toolbox.r(), toolbox.nu(), toolbox.gamma()])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Custom mutation operator that respects parameter positions
def mutate_individual(ind):
    i = random.randint(0, 4)
    if i == 0:
        ind[i] = random.choice(K_VALUES)
    elif i == 1:
        ind[i] = random.choice(W_VALUES)
    elif i == 2:
        ind[i] = random.choice(R_VALUES)
    elif i == 3:
        ind[i] = np.round(max(0.001, min(1.0, ind[i] + random.gauss(0, 0.01))), 2)
    elif i == 4:
        ind[i] = random.choice(GAMMA_VALUES)
    return ind,

# Optimization process

def run_optimization(use_neptune=False, ocel_path=None):
    if ocel_path is None:
        raise ValueError("ocel_path is empty.")

    if use_neptune:
        run = neptune.init_run(
            name=f"Optimization_{os.path.basename(ocel_path)}",
            project="MaleLab/VariantWavelets",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YTE5Zjg5NC1mMjk0LTRlN2UtYjgxMC03OTE1ZWJiYjliNTQifQ==",
        )
    else:
        run = None

    def local_evaluate(individual):
        k, w, r, nu, gamma = individual
        print(f"Evaluating: k={k}, w={w}, r={r}, nu={nu}, gamma={gamma}")
        try:
            results = create_experiment(
                ocel_path=ocel_path,
                k=k,
                w=w,
                r=r,
                ocel_case_notion=CASE_NOTION,
                config_input=CONFIG_INPUT,
                occ_kernel='rbf',
                occ_gamma=gamma,
                occ_nu=nu
            )
            return (results,)
        except Exception as e:
            print(f"⚠️ Error evaluating individual {individual}: {e}")
            stack_trace = traceback.format_exc()
            print(stack_trace)
            return (0.0,)

    # Registrando operadores
    toolbox.register("evaluate", local_evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)
    NGEN = 50

    for gen in range(NGEN):
        print(f"\n🔁 Generation {gen}")

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.5)

        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        elite = tools.selBest(population, k=1)
        selected = toolbox.select(offspring, k=len(population) - 1)
        population = selected + elite

        best = tools.selBest(population, k=1)[0]
        print(f"🏆 Best in Generation {gen}: {best}, F1: {best.fitness.values[0]}")
        if run:
            run["current best/f1"] = best.fitness.values[0]
            #run[f"generation/{gen}/k"] = best[0]
            #run[f"generation/{gen}/w"] = best[1]
            #run[f"generation/{gen}/r"] = best[2]
            #run[f"generation/{gen}/nu"] = best[3]
            #run[f"generation/{gen}/gamma"] = best[4]

    best_ind = tools.selBest(population, k=1)[0]
    best_f1 = best_ind.fitness.values[0]

    print("\n🎯 Best individual overall:", best_ind)
    print("🏅 F1-score:", best_f1)

    if run:
        run["best/k"] = best_ind[0]
        run["best/w"] = best_ind[1]
        run["best/r"] = best_ind[2]
        run["best/nu"] = best_ind[3]
        run["best/gamma"] = best_ind[4]
        run["best/f1"] = best_f1
        run.stop()

    return best_ind

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocel_path", type=str, default="datasets/experiments", help="Path to the OCEL file")
    parser.add_argument("--use_neptune", action="store_true", help="Enable Neptune logging")
    args = parser.parse_args()

    run_optimization(use_neptune=args.use_neptune, ocel_path=args.ocel_path)
