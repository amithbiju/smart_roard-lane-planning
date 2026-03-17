import pandas as pd
import subprocess
import sumolib
import random
import os

# --- CONFIG ---
CSV_FILE = 'traffic_history_v2.csv'
NET_FILE = 'map.net.xml'
RAW_TRIPS = 'raw_trips.trips.xml'   # Temporary file
VALID_ROUTE_FILE = 'simulation.rou.xml' # Final valid file
TOTAL_VEHICLES = 1000 

def generate_demand():
    print("1. Loading Congestion Data...")
    df = pd.read_csv(CSV_FILE)
    
    # Weight edges by congestion
    road_weights = df.groupby('osm_id')['congestion_ratio'].mean().to_dict()
    
    net = sumolib.net.readNet(NET_FILE)
    sumo_edges = [e.getID() for e in net.getEdges() if e.allows("passenger")]
    
    weighted_edges = []
    for edge in sumo_edges:
        clean_id = edge.lstrip('-').split('#')[0]
        try:
            osm_id = int(clean_id)
            weight = road_weights.get(osm_id, 0.1) 
            final_weight = 1 + (weight * 10) 
            weighted_edges.append((edge, final_weight))
        except:
            continue

    print("2. Generating Raw Trips...")
    with open(RAW_TRIPS, 'w') as f:
        f.write('<routes>\n')
        f.write(f'    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>\n')
        
        edges_only = [x[0] for x in weighted_edges]
        weights_only = [x[1] for x in weighted_edges]
        
        for i in range(TOTAL_VEHICLES):
            src, dst = random.choices(edges_only, weights=weights_only, k=2)
            if src != dst:
                f.write(f'    <trip id="veh{i}" type="car" depart="{i/2}" from="{src}" to="{dst}"/>\n')
        
        f.write('</routes>')

    print("3. Validating Routes with duarouter...")
    # This command calculates the actual path. 
    # --ignore-errors: Skips cars that can't find a path.
    # --remove-loops: Fixes paths that circle back unnecessarily.
    command = [
        "duarouter",
        "--net-file", NET_FILE,
        "--route-files", RAW_TRIPS,
        "--output-file", VALID_ROUTE_FILE,
        "--ignore-errors", "true",
        "--no-step-log", "true",
        "--no-warnings", "true"
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"SUCCESS: Valid routes saved to '{VALID_ROUTE_FILE}'")
        print("Invalid trips were automatically removed.")
    except subprocess.CalledProcessError as e:
        print(f"Error running duarouter: {e}")

if __name__ == "__main__":
    generate_demand()