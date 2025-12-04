import geopandas as gpd
import leafmap.foliumap as leafmap
import streamlit as st
from shapely.geometry import MultiLineString, LineString, Point
import pickle
import math
import heapq


graph_nodes = gpd.read_file("Nodes.gpkg", layer="nodes")
graph_edges = gpd.read_file("GraphEdges.gpkg", layer="graphedges")


graph_nodes['node_id'] = graph_nodes.index
name_to_id = {}

for idx, row in graph_nodes.iterrows():
    if 'FULL_NAME' in row and row['FULL_NAME']:
        name_to_id[row['FULL_NAME']] = idx

with open("graph_adjacency.pkl", "rb") as f:
    graph = pickle.load(f)


def init_map():
    create_graph()

    building_nodes = gpd.read_file("BuildingNodes.gpkg", layer="buildings")
    building_nodes_wgs = building_nodes.to_crs(epsg=4326)
    building_nodes_wgs = building_nodes_wgs[~building_nodes_wgs.geometry.is_empty].reset_index(drop=True)

    building_nodes_filtered = building_nodes_wgs[building_nodes_wgs['FULL_NAME'].notnull()]

    building_names = building_nodes_filtered['FULL_NAME'].tolist()

    start_building = st.selectbox("Current Location", building_names)
    end_building = st.selectbox("Destination", building_names)


    center_lat = building_nodes_filtered.geometry.y.mean()
    center_lon = building_nodes_filtered.geometry.x.mean()

    st.title("WSU Campus Map")
    m = leafmap.Map(center=[center_lat, center_lon], zoom=15)

    for _, row in building_nodes_filtered.iterrows():
        m.add_marker(
            location=[row.geometry.y, row.geometry.x],
            popup=row["FULL_NAME"]  # Only the name is displayed
        )

    if st.button("Navigate"):
        path = dijkstra(start_building, end_building)
        print(path)
        display_path_on_map(path, m)


    m.to_streamlit()


def get_lines(line):
    if isinstance(line, LineString):
        return [line]
    elif isinstance(line, MultiLineString):
        return list(line.geoms)
    else:
        raise ValueError("Unsupported geometry type")

def nearest_node(point, nodes_gdf):
    distances = nodes_gdf.geometry.distance(point)
    nearest_idx = distances.idxmin()
    nearest_pos = nodes_gdf.index.get_loc(nearest_idx)
    return nodes_gdf.iloc[nearest_pos]['node_id']

#created with chatgpt 4.5
def create_graph(tol=0.01):
    global graph, edge_lookup
    graph = {node_id: {} for node_id in graph_nodes['node_id']}
    edge_lookup = {}



    sindex = graph_nodes.sindex

    for _, row in graph_edges.iterrows():
        lines = get_lines(row.geometry)

        for line in lines:
            # 1️⃣ Get candidate nodes via spatial index
            possible_idx = list(sindex.intersection(line.bounds))
            candidate_nodes = graph_nodes.iloc[possible_idx]

            # 2️⃣ Keep only nodes within tolerance of the line
            nodes_on_line = []
            for _, node_row in candidate_nodes.iterrows():
                if line.distance(node_row.geometry) <= tol:
                    nodes_on_line.append((node_row['node_id'], node_row.geometry))

            if len(nodes_on_line) < 2:
                continue  # skip lines that don't connect at least 2 nodes

            # 3️⃣ Sort nodes along the line
            nodes_on_line.sort(key=lambda x: line.project(x[1]))

            # 4️⃣ Create edges between consecutive nodes
            for i in range(len(nodes_on_line) - 1):
                n1_id, n1_geom = nodes_on_line[i]
                n2_id, n2_geom = nodes_on_line[i + 1]
                weight = n1_geom.distance(n2_geom)

                # Add to adjacency graph
                graph[n1_id][n2_id] = weight
                graph[n2_id][n1_id] = weight

    # Save both graph and edge_lookup
    with open("graph_adjacency.pkl", "wb") as f:
        pickle.dump(graph, f)

def display_path_on_map(path_nodes, m=None):
    if len(path_nodes) < 2:
        print("Path too short to draw.")
        return m

    # Build LineString using node coordinates
    coords = [ (graph_nodes.loc[node].geometry.x, graph_nodes.loc[node].geometry.y) for node in path_nodes ]
    path_line = LineString(coords)

    # Create GeoDataFrame
    gdf_path = gpd.GeoDataFrame(geometry=[path_line], crs=graph_nodes.crs)
    gdf_path_wgs = gdf_path.to_crs(epsg=4326)

    # Create map if none passed
    if m is None:
        center_lat = gdf_path_wgs.geometry.centroid.y.mean()
        center_lon = gdf_path_wgs.geometry.centroid.x.mean()
        m = leafmap.Map(center=[center_lat, center_lon], zoom=16)

    # Add path
    m.add_gdf(gdf_path_wgs, layer_name="Path", color="red", weight=5)

    return m




def dijkstra(start_name, end_name):
    start = name_to_id[start_name]
    end = name_to_id[end_name]

    f = {node: math.inf for node in graph}
    prev = {node: None for node in graph}

    f[start] = 0

    heap = [(0, start)]

    while heap:
        dist, x = heapq.heappop(heap)

        if dist > f[x]:
            continue  # outdated entry

        for z, weight in graph[x].items():
            if f[x] + weight < f[z]:
                f[z] = f[x] + weight
                prev[z] = x
                heapq.heappush(heap, (f[z], z))

        # Reconstruct path
    path = []
    cur = end
    if f[end] == math.inf:
        return []
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path








def main():
    init_map()










if __name__ == "__main__":
    main()

































