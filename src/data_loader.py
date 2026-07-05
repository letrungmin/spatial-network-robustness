import os
import numpy as np
import networkx as nx

class S3DISGraphBuilder:
    def __init__(self, data_root):
        """
        Initialize the spatial network extractor from the S3DIS dataset.
        data_root: Path to the Stanford3dDataset_v1.2_Aligned_Version directory.
        """
        self.data_root = data_root

    def _get_bounding_box(self, file_path):
        """
        Quickly read the coordinate .txt file (X, Y, Z, R, G, B) 
        and return the Axis-Aligned Bounding Box (min, max).
        """
        try:
            # Only load the first 3 coordinate columns (X, Y, Z) to optimize RAM and speed
            points = np.loadtxt(file_path, usecols=(0, 1, 2))
            if points.ndim == 1:  # Handle edge case where the file has only 1 point
                points = np.expand_dims(points, axis=0)
            
            bbox_min = np.min(points, axis=0)
            bbox_max = np.max(points, axis=0)
            return bbox_min, bbox_max
        except Exception as e:
            return None, None

    def extract_room_geometry(self, area_name, room_name):
        """
        Extract all geometric entities (Rooms, Doors) into Bounding Boxes.
        """
        room_path = os.path.join(self.data_root, area_name, room_name)
        annotations_dir = os.path.join(room_path, "Annotations")
        
        room_data = {'doors': [], 'bbox': None, 'centroid': None}
        all_room_points_min = []
        all_room_points_max = []
        
        if not os.path.exists(annotations_dir):
            return None

        # Scan all objects inside the room
        for file in os.listdir(annotations_dir):
            if not file.endswith(".txt"): 
                continue
            
            file_path = os.path.join(annotations_dir, file)
            bbox_min, bbox_max = self._get_bounding_box(file_path)
            
            if bbox_min is None: 
                continue
            
            all_room_points_min.append(bbox_min)
            all_room_points_max.append(bbox_max)
            
            # If the entity is a door, save it as a spatial intersection zone
            if "door" in file:
                room_data['doors'].append((bbox_min, bbox_max))
                
        # Calculate the encompassing AABB for the entire room
        if all_room_points_min:
            r_min = np.min(all_room_points_min, axis=0)
            r_max = np.max(all_room_points_max, axis=0)
            room_data['bbox'] = (r_min, r_max)
            room_data['centroid'] = (r_min + r_max) / 2.0
            
        return room_data

    def _check_aabb_intersection(self, box1, box2, epsilon=0.15):
        """
        Mathematical algorithm to check intersection/contact between two AABBs.
        epsilon: Spatial tolerance threshold (meters) to handle fuzzy boundaries.
        """
        min1, max1 = box1
        min2, max2 = box2
        
        # Check if two boxes overlap or touch within the epsilon threshold
        intersect = (min1[0] - epsilon <= max2[0] and max1[0] + epsilon >= min2[0]) and \
                    (min1[1] - epsilon <= max2[1] and max1[1] + epsilon >= min2[1]) and \
                    (min1[2] - epsilon <= max2[2] and max1[2] + epsilon >= min2[2])
        return intersect

    def build_spatial_network(self, area_name):
        """
        Core contribution: Automate the extraction of the spatial topological graph.
        """
        G = nx.Graph()
        area_path = os.path.join(self.data_root, area_name)
        
        if not os.path.exists(area_path):
            print(f"Error: Data for {area_name} not found at {area_path}")
            return G

        print(f"=== ANALYZING TOPOLOGICAL NETWORK FOR: {area_name} ===")
        
        # Step 1: Scan and store the geometry of all rooms in RAM
        rooms_geometry = {}
        for room_name in os.listdir(area_path):
            if room_name.startswith(".") or os.path.isfile(os.path.join(area_path, room_name)):
                continue
            
            geo_data = self.extract_room_geometry(area_name, room_name)
            if geo_data and geo_data['bbox'] is not None:
                rooms_geometry[room_name] = geo_data
                # Initialize Node with spatial mathematical attributes
                G.add_node(room_name, 
                           type="room", 
                           centroid=geo_data['centroid'].tolist(),
                           bbox_min=geo_data['bbox'][0].tolist(),
                           bbox_max=geo_data['bbox'][1].tolist())

        # Step 2: Iterate through pairs to find network link relationships (Edge Generation)
        room_list = list(rooms_geometry.keys())
        for i in range(len(room_list)):
            for j in range(i + 1, len(room_list)):
                r1 = room_list[i]
                r2 = room_list[j]
                
                # Condition 1: Are the two spaces adjacent?
                if self._check_aabb_intersection(rooms_geometry[r1]['bbox'], rooms_geometry[r2]['bbox'], epsilon=0.2):
                    
                    # Condition 2: Is there a connecting door?
                    connected = False
                    
                    # Scan if room 1's door touches room 2's bounding box
                    for d1_box in rooms_geometry[r1]['doors']:
                        if self._check_aabb_intersection(d1_box, rooms_geometry[r2]['bbox'], epsilon=0.15):
                            connected = True
                            break
                    
                    if not connected:
                        # Conversely, scan if room 2's door touches room 1's bounding box
                        for d2_box in rooms_geometry[r2]['doors']:
                            if self._check_aabb_intersection(d2_box, rooms_geometry[r1]['bbox'], epsilon=0.15):
                                connected = True
                                break
                    
                    # If conditions are met, generate an Edge connecting the entities
                    if connected:
                        # Calculate the physical distance as the Edge Weight
                        c1 = rooms_geometry[r1]['centroid']
                        c2 = rooms_geometry[r2]['centroid']
                        weight = float(np.linalg.norm(c1 - c2))
                        
                        G.add_edge(r1, r2, weight=weight)

        print(f"-> Result: Successfully established Graph with {G.number_of_nodes()} Nodes (Rooms) and {G.number_of_edges()} Edges (Connections).")
        return G

if __name__ == "__main__":
    # Declare the relative path to the clean dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(base_dir, "data", "Stanford3dDataset_v1.2_Aligned_Version")
    
    builder = S3DISGraphBuilder(DATA_PATH)
    
    # Run system test on Area_1
    spatial_graph = builder.build_spatial_network("Area_1")