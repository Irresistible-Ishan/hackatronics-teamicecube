# pip install streamlit osmnx networkx folium streamlit-folium


import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
import random

# -------------------------------
# SafeRouteAI — Interactive Demo
# -------------------------------

st.set_page_config(page_title="SafeRouteAI", layout="wide")
st.title("🛣️ SafeRouteAI")
st.markdown("""
Find the **safest route** instead of just the fastest!  
Adjust the balance between **speed** and **safety** below.
""")

# -------------------------------
# User Inputs
# -------------------------------

place = st.text_input("Enter area name (city/locality):", "Koramangala, Bengaluru, India")
alpha = st.slider("Speed vs Safety ⚖️", 0.0, 1.0, 0.6, 0.1,
                  help="0 → prioritize safety, 1 → prioritize speed")

if st.button("Find Route"):
    with st.spinner("Fetching map and calculating safest path..."):
        try:
            # Load map graph
            G = ox.graph_from_place(place, network_type='walk', simplify=True)

            # Assign mock crime risk and combine with distance
            max_len = max(data.get('length', 1) for _, _, _, data in G.edges(keys=True, data=True))
            for u, v, key, data in G.edges(keys=True, data=True):
                data['length'] = data.get('length', 1)
                crime_risk = random.uniform(0, 1)  # simulate crime
                dist = data['length'] / max_len
                risk = crime_risk
                beta = 1 - alpha  # safety importance
                data['weight'] = alpha * dist + beta * risk
                data['crime_risk'] = crime_risk

            # Pick random start and end nodes for demo
            nodes = list(G.nodes())
            orig, dest = random.sample(nodes, 2)

            # Compute routes
            fastest_route = nx.shortest_path(G, orig, dest, weight='length')
            safest_route = nx.shortest_path(G, orig, dest, weight='weight')

            # Plot map
            m = folium.Map(location=[ox.graph_to_gdfs(G, nodes=True, edges=False).geometry.y.mean(),
                                     ox.graph_to_gdfs(G, nodes=True, edges=False).geometry.x.mean()],
                           zoom_start=15)

            # Add both routes
            folium.PolyLine(locations=ox.utils_graph.route_to_coordinates(G, fastest_route),
                            color="red", weight=4, opacity=0.6, tooltip="Fastest Route").add_to(m)
            folium.PolyLine(locations=ox.utils_graph.route_to_coordinates(G, safest_route),
                            color="green", weight=5, opacity=0.8, tooltip="Safest Route").add_to(m)

            # Add start and end markers
            start_coords = (G.nodes[orig]['y'], G.nodes[orig]['x'])
            end_coords = (G.nodes[dest]['y'], G.nodes[dest]['x'])
            folium.Marker(start_coords, tooltip="Start", icon=folium.Icon(color="blue")).add_to(m)
            folium.Marker(end_coords, tooltip="Destination", icon=folium.Icon(color="orange")).add_to(m)

            # Display map
            st.success("✅ Route generated successfully!")
            st_folium(m, width=700, height=500)

        except Exception as e:
            st.error(f"❌ Error: {e}")


