import xml.etree.ElementTree as ET
import sys

def get_stats(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        durations = []
        wait_steps = []
        teleports = 0
        
        for trip in root.findall('tripinfo'):
            durations.append(float(trip.get('duration')))
            wait_steps.append(float(trip.get('waitingTime')))
            
        return {
            "avg_travel_time": sum(durations) / len(durations) if durations else 0,
            "total_waiting_time": sum(wait_steps),
            "total_cars": len(durations)
        }
    except FileNotFoundError:
        print(f"Error: Could not find {xml_file}. Did you run the simulation step?")
        sys.exit(1)

def main():
    print("--- Traffic Simulation Results ---")
    
    # 1. Get Old Stats
    old = get_stats("tripinfo_old.xml")
    print(f"\n[OLD MAP]")
    print(f"  Avg Travel Time: {old['avg_travel_time']:.2f} s")
    print(f"  Total Waiting:   {old['total_waiting_time']:.2f} s")
    print(f"  Completed Trips: {old['total_cars']}")

    # 2. Get New Stats
    new = get_stats("tripinfo_new.xml")
    print(f"\n[NEW MAP]")
    print(f"  Avg Travel Time: {new['avg_travel_time']:.2f} s")
    print(f"  Total Waiting:   {new['total_waiting_time']:.2f} s")
    print(f"  Completed Trips: {new['total_cars']}")

    # 3. Calculate Improvement
    print(f"\n[IMPROVEMENT]")
    
    if old['avg_travel_time'] > 0:
        time_imp = ((old['avg_travel_time'] - new['avg_travel_time']) / old['avg_travel_time']) * 100
        print(f"  Travel Time: {time_imp:+.2f}% {'(FASTER)' if time_imp > 0 else '(SLOWER)'}")
    
    if old['total_waiting_time'] > 0:
        wait_imp = ((old['total_waiting_time'] - new['total_waiting_time']) / old['total_waiting_time']) * 100
        print(f"  Congestion:  {wait_imp:+.2f}% {'(REDUCED)' if wait_imp > 0 else '(INCREASED)'}")

if __name__ == "__main__":
    main()