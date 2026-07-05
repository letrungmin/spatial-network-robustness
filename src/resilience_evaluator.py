import copy
import random
import networkx as nx

class NetworkResilienceSimulator:
    def __init__(self, baseline_graph):
        """
        Initialize the simulator with the intact baseline spatial graph.
        """
        self.baseline_graph = baseline_graph
        self.initial_nodes = baseline_graph.number_of_nodes()
        self.initial_edges = baseline_graph.number_of_edges()

    def calculate_global_efficiency(self, graph):
        """
        Calculate the topological efficiency of the network.
        Metric used to evaluate the resilience of the smart urban system.
        """
        if graph.number_of_nodes() == 0: 
            return 0.0
        return nx.global_efficiency(graph)

    def simulate_random_edge_failure(self, failure_rate):
        """
        Scenario 1: Random Edge Failure (e.g., random doors locked/jammed).
        Removes a percentage of random edges.
        """
        G_sim = copy.deepcopy(self.baseline_graph)
        edges = list(G_sim.edges())
        
        num_to_remove = int(len(edges) * failure_rate)
        edges_to_remove = random.sample(edges, num_to_remove)
        
        G_sim.remove_edges_from(edges_to_remove)
        return G_sim

    def simulate_targeted_node_attack(self, removal_rate):
        """
        Scenario 2: Targeted Node Attack based on Degree Centrality.
        (e.g., Fire breaking out in main hallways/hubs).
        Removes a percentage of the most highly-connected nodes.
        """
        G_sim = copy.deepcopy(self.baseline_graph)
        
        # Calculate Degree Centrality to identify structural hubs
        degree_dict = dict(G_sim.degree())
        # Sort nodes by degree (highest first)
        sorted_nodes = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
        
        num_to_remove = int(self.initial_nodes * removal_rate)
        nodes_to_remove = [node for node, degree in sorted_nodes[:num_to_remove]]
        
        G_sim.remove_nodes_from(nodes_to_remove)
        return G_sim

if __name__ == "__main__":
    from data_loader import S3DISGraphBuilder
    import os
    
    # 1. Load the Baseline Graph
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(base_dir, "data", "Stanford3dDataset_v1.2_Aligned_Version")
    builder = S3DISGraphBuilder(DATA_PATH)
    
    print("\n--- PHASE 1: GRAPH EXTRACTION ---")
    G = builder.build_spatial_network("Area_1")
    
    # 2. Initialize the Evaluator
    print("\n--- PHASE 2: RESILIENCE EVALUATION ---")
    simulator = NetworkResilienceSimulator(G)
    
    baseline_eff = simulator.calculate_global_efficiency(G)
    print(f"Baseline Global Efficiency (Intact Building): {baseline_eff:.4f}")
    
    # 3. Run Experiments (Ablation Study Matrix)
    print("\n[Experiment A] Random Door Failures (Edge Attack)")
    for rate in [0.1, 0.3, 0.5]: # 10%, 30%, 50% failure
        G_fail = simulator.simulate_random_edge_failure(failure_rate=rate)
        eff = simulator.calculate_global_efficiency(G_fail)
        print(f" -> Failure Rate {rate*100}% | Efficiency drop: {baseline_eff:.4f} -> {eff:.4f}")

    print("\n[Experiment B] Central Hub Fires (Targeted Node Attack)")
    for rate in [0.1, 0.2, 0.3]: # 10%, 20%, 30% hubs destroyed
        G_attack = simulator.simulate_targeted_node_attack(removal_rate=rate)
        eff = simulator.calculate_global_efficiency(G_attack)
        print(f" -> Hub Loss {rate*100}%   | Efficiency drop: {baseline_eff:.4f} -> {eff:.4f}")