import gymnasium as gym
from gymnasium import spaces
import numpy as np
import xml.etree.ElementTree as ET
import subprocess
import traci
import sumolib
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import time
import csv

# --- CONFIGURATION ---
PLAIN_EDG_FILE = 'map_plain.edg.xml'
NET_FILE = 'map.net.xml'
ROUTE_FILE = 'simulation.rou.xml'
SUMO_BINARY = "sumo"  # Change to "sumo-gui" if you want to watch the training

# --- CIVIL ENGINEERING POLICY CONSTANTS ---
# Source: IRC:SP:84-2019 & Land Acquisition Manual
MAX_STANDARD_LANES = 4        # Maximum lanes before requiring expensive land acquisition
FLYOVER_SPEED_LIMIT = 27.78   # 100 km/h design speed
LAND_ACQ_COST_FACTOR = 4.5    # 4.5x Cost Penalty for land acquisition
FLYOVER_MIN_CONGESTION = 0.4  # Minimum density required for grade separator

def load_config():
    global MAX_STANDARD_LANES
    global FLYOVER_SPEED_LIMIT
    global LAND_ACQ_COST_FACTOR
    global FLYOVER_MIN_CONGESTION
    
    import json
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                
            if "maxStandardLanes" in config:
                MAX_STANDARD_LANES = int(config["maxStandardLanes"])
            if "flyoverSpeedLimit" in config:
                FLYOVER_SPEED_LIMIT = float(config["flyoverSpeedLimit"])
            if "landAcqCostFactor" in config:
                LAND_ACQ_COST_FACTOR = float(config["landAcqCostFactor"])
            if "flyoverMinCongestion" in config:
                FLYOVER_MIN_CONGESTION = float(config["flyoverMinCongestion"])
                
            print("Loaded dynamic configuration from config.json:")
            print(f"  MAX_STANDARD_LANES: {MAX_STANDARD_LANES}")
            print(f"  FLYOVER_SPEED_LIMIT: {FLYOVER_SPEED_LIMIT}")
            print(f"  LAND_ACQ_COST_FACTOR: {LAND_ACQ_COST_FACTOR}")
            print(f"  FLYOVER_MIN_CONGESTION: {FLYOVER_MIN_CONGESTION}")
        except Exception as e:
            print(f"Failed to load config.json: {e}")

# --- 1. THE ENVIRONMENT ---
class SumoNetworkEnv(gym.Env):
    def __init__(self):
        super(SumoNetworkEnv, self).__init__()
        
        # Load network to find valid edges
        try:
            self.net = sumolib.net.readNet(NET_FILE)
        except Exception as e:
            print(f"Error reading net file: {e}")
            sys.exit(1)

        self.edges = [e.getID() for e in self.net.getEdges() if e.allows("passenger")]
        
        # Action Space:
        # 0 to N-1: Add Lane (Corridor Mode)
        # N to 2N-1: Add Flyover (Spot Mode)
        self.n_actions = len(self.edges) * 2
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Observation Space: [Avg Speed, Occupancy] for each edge
        self.observation_space = spaces.Box(low=0, high=100, shape=(len(self.edges) * 2,), dtype=np.float32)
        
        self.current_step = 0
        self.last_step_metrics = {} 

    def reset(self, seed=None):
        self._reset_map_file() 
        initial_state = self._run_simulation()
        return initial_state, {}

    def _reset_map_file(self):
        # Restore map from the current BASELINE (.bak file)
        # Note: The .bak file updates whenever we find a new record!
        if os.path.exists(PLAIN_EDG_FILE + ".bak"):
            with open(PLAIN_EDG_FILE + ".bak", 'rb') as src, open(PLAIN_EDG_FILE, 'wb') as dst:
                dst.write(src.read())
        else:
            with open(PLAIN_EDG_FILE, 'rb') as src, open(PLAIN_EDG_FILE + ".bak", 'wb') as dst:
                dst.write(src.read())

    def step(self, action):
        target_edge_idx = action % len(self.edges)
        action_type = "LANE" if action < len(self.edges) else "FLYOVER"
        target_edge_id = self.edges[target_edge_idx]
        
        cost = 0
        build_success = False
        policy_penalty = 0

        # Get occupancy from last run for policy checks
        current_occupancy = self.last_step_metrics.get(target_edge_id, {}).get('occupancy', 0.5)

        action_info = None

        if action_type == "LANE":
            # Attempt Corridor Widening
            success, cost_val = self._add_lane_policy_aware(target_edge_id)
            build_success = success
            cost = cost_val
            if success:
                # Use base_id for lanes since it widens the whole corridor
                action_info = {"type": "Lane Addition", "target": target_edge_id.split('#')[0], "cost": cost_val}
            
        else: # FLYOVER
            # Policy Check: Justification
            if current_occupancy < FLYOVER_MIN_CONGESTION:
                policy_penalty = -20 # Penalty for proposing unnecessary structure
                build_success = False
            else:
                build_success = self._add_flyover_policy_aware(target_edge_id)
                cost = 50 # Base cost for structure
                if build_success:
                    action_info = {"type": "Flyover", "target": target_edge_id, "cost": cost}

        # Early return if failed
        if not build_success:
            return np.zeros(self.observation_space.shape, dtype=np.float32), -5 + policy_penalty, False, False, {}

        # Rebuild Map
        try:
            subprocess.run(['netconvert', 
                            '-e', PLAIN_EDG_FILE, 
                            '-n', 'map_plain.nod.xml', 
                            '-o', NET_FILE, 
                            '--no-warnings'], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("CRITICAL: Map build failed. Network is invalid.")
            return np.zeros(self.observation_space.shape, dtype=np.float32), -1000, True, False, {}

        # Run Simulation
        state = self._run_simulation()
        
        # INTEGRATED FROM UNIVERSAL OPTIMIZER: 
        # Heavy penalty if simulation fails/crashes so it isn't viewed as an improvement
        # (A crashed simulation returns 0 for everything, which without this check, looks like 0 congestion!)
        if np.sum(state) == 0:
            print("CRITICAL: Simulation crashed or routing became invalid.")
            return state, -1000, True, False, {}
            
        # Calculate Reward
        avg_speed = np.mean(state[::2])
        avg_occupancy = np.mean(state[1::2]) 
        
        # Reward Formula
        # We heavily penalize cost to force the AI to be "financially responsible"
        reward = (avg_speed * 10) - (avg_occupancy * 100) - (cost * 0.5)
        
        self.current_step += 1
        done = self.current_step >= 10 
        
        info = {"action": action_info} if action_info else {}
        return state, reward, done, False, info

    def _add_lane_policy_aware(self, edge_id):
        """
        Widening acts on the WHOLE CORRIDOR (all segments sharing the same Base ID)
        to ensure uniformity (IRC:SP:84-2019, 4.2).
        """
        try:
            tree = ET.parse(PLAIN_EDG_FILE)
            root = tree.getroot()
            
            # 1. Identify Corridor (e.g., 'edge#1' -> Base 'edge')
            base_id = edge_id.split('#')[0]
            corridor_edges = []
            
            for edge in root.findall('edge'):
                e_id = edge.get('id')
                if e_id == base_id or e_id.startswith(f"{base_id}#"):
                    corridor_edges.append(edge)
            
            if not corridor_edges:
                return False, 0

            # 2. Feasibility Check: Are we hitting the 4-Lane Cap?
            for edge in corridor_edges:
                curr_lanes = int(edge.get('numLanes', 1))
                if curr_lanes >= MAX_STANDARD_LANES:
                    # Reject entire project if any part is already maxed
                    return False, 0
            
            # 3. Apply Action to ALL segments
            print(f"Action: Upgrading Corridor {base_id} (Uniform Widening)...")
            total_cost_factor = 0
            
            for edge in corridor_edges:
                curr_lanes = int(edge.get('numLanes', 1))
                edge.set('numLanes', str(curr_lanes + 1))
                
                # Cost Calculation per segment
                # If going 3->4, it's expensive. 1->2 or 2->3 is standard.
                if curr_lanes == 3:
                    total_cost_factor += (10 * LAND_ACQ_COST_FACTOR) 
                else:
                    total_cost_factor += 10 # Base cost
            
            tree.write(PLAIN_EDG_FILE)
            return True, total_cost_factor

        except Exception as e:
            print(f"Error in corridor update: {e}")
            return False, 0

    def _add_flyover_policy_aware(self, edge_id):
        """
        Builds a flyover with specific design speed (100 km/h).
        """
        try:
            tree = ET.parse(PLAIN_EDG_FILE)
            root = tree.getroot()
            
            flyover_id = f"{edge_id}_flyover"
            
            # Check existence
            for edge in root.findall('edge'):
                if edge.get('id') == flyover_id: return False

            target = None
            for edge in root.findall('edge'):
                if edge.get('id') == edge_id:
                    target = edge
                    break
            
            if target is not None:
                new_edge = ET.Element('edge')
                new_edge.set('id', flyover_id)
                new_edge.set('from', target.get('from'))
                new_edge.set('to', target.get('to'))
                new_edge.set('numLanes', '1')
                
                # Policy: Design Speed 100km/h
                new_edge.set('speed', str(FLYOVER_SPEED_LIMIT)) 
                new_edge.set('priority', '100')
                
                root.append(new_edge)
                tree.write(PLAIN_EDG_FILE)
                print(f"Action: Built Flyover for {edge_id}")
                return True
            return False
        except Exception:
            return False

    def _run_simulation(self):
        # Robust simulation runner
        try: traci.close()
        except: pass
        
        port = sumolib.miscutils.getFreeSocketPort()
        
        # INTEGRATED FROM UNIVERSAL OPTIMIZER: 
        # Dynamic routing flags ensure vehicles adapt to new infrastructure correctly
        sumo_cmd = [
            SUMO_BINARY, 
            "-n", NET_FILE, 
            "-r", ROUTE_FILE, 
            "--no-step-log", "true", 
            "--time-to-teleport", "-1", 
            "--random", "true", 
            "--no-warnings", "true",
            "--device.rerouting.probability", "1",
            "--device.rerouting.adaptation-interval", "10"
        ]
        
        if SUMO_BINARY == "sumo-gui":
            sumo_cmd.extend(["--start", "--quit-on-end"])

        try:
            traci.start(sumo_cmd, port=port)
        except Exception:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        edge_stats = {e: {'speed': [], 'occupancy': []} for e in self.edges}
        step = 0
        
        try:
            while step < 500: # Simulation Duration
                traci.simulationStep()
                step += 1
                if step % 50 == 0:
                    for e in self.edges:
                        try:
                            s = traci.edge.getLastStepMeanSpeed(e)
                            o = traci.edge.getLastStepOccupancy(e)
                            edge_stats[e]['speed'].append(s)
                            edge_stats[e]['occupancy'].append(o)
                            self.last_step_metrics[e] = {'speed': s, 'occupancy': o}
                        except: pass
                if traci.simulation.getMinExpectedNumber() <= 0: break
        except Exception:
            # If TRACI disconnects or simulation fails mid-run
            try: traci.close()
            except: pass
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        finally:
            try: traci.close()
            except: pass
        
        obs = []
        for e in self.edges:
            speeds = edge_stats[e]['speed']
            occs = edge_stats[e]['occupancy']
            obs.extend([sum(speeds)/len(speeds) if speeds else 0, sum(occs)/len(occs) if occs else 0])
        return np.array(obs, dtype=np.float32)

# --- 2. THE DQN AGENT ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.fc(x)

def train():
    # Load dynamic config
    load_config()

    # Initial Backup
    if not os.path.exists(PLAIN_EDG_FILE + ".bak"):
        shutil.copy(PLAIN_EDG_FILE, PLAIN_EDG_FILE + ".bak")

    env = SumoNetworkEnv()
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    episodes = 50 
    gamma = 0.9
    epsilon = 1.0
    
    best_episode_speed = -1.0 
    best_actions_history = []

    print(f"Starting Training with Civil & Dynamic Routing Policies...")
    
    for ep in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        speed_measurements = []
        current_episode_actions = []
        
        done = False
        while not done:
            if random.random() < epsilon: action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = model(state)
                    action = torch.argmax(q_vals).item()
            
            next_state, reward, done, _, info = env.step(action)
            
            if "action" in info:
                act = info["action"]
                act["episode"] = ep + 1
                current_episode_actions.append(act)
            
            # Track Performance
            current_avg_speed = np.mean(next_state[::2])
            speed_measurements.append(current_avg_speed)
            
            next_state_t = torch.FloatTensor(next_state)
            target = reward + gamma * torch.max(model(next_state_t))
            prediction = model(state)[action]
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state_t
            total_reward += reward
            
        epsilon = max(0.1, epsilon * 0.95)
        avg_ep_speed = sum(speed_measurements) / len(speed_measurements) if speed_measurements else 0
        
        print(f"Episode {ep+1}: Reward: {total_reward:.2f} | Avg Speed: {avg_ep_speed:.2f} m/s")
        
        # --- ITERATIVE SAVE LOGIC ---
        # 1. Warmup Check: ep > 5
        # 2. Performance Check: Speed > Best Speed
        if ep > 5 and avg_ep_speed > best_episode_speed:
            best_episode_speed = avg_ep_speed
            
            # Save the Visual Result
            shutil.copy(NET_FILE, "best_map.net.xml")
            
            # CRITICAL: Update the Baseline (.bak)
            # This locks in the changes (e.g., widened corridors) so the AI
            # starts the NEXT episode with these improvements already in place.
            shutil.copy(PLAIN_EDG_FILE, PLAIN_EDG_FILE + ".bak")
            
            # Commit the structural changes made in this episode
            best_actions_history.extend(current_episode_actions)
            
            # Write to CSV
            with open("construction_log.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["episode", "type", "target", "cost"])
                writer.writeheader()
                writer.writerows(best_actions_history)
            
            print(f"  >>> New Record! Map Saved (Speed: {best_episode_speed:.2f} m/s)")
            print(f"  >>> Baseline Updated: Future episodes will start with this improved map.")

    # Restore original map.net.xml from temp folder at the end of training
    if os.path.exists("temp/map.net.xml"):
        shutil.copy("temp/map.net.xml", NET_FILE)
        print("Restored original map.net.xml from temp/")
    else:
        print("Warning: temp/map.net.xml not found - map.net.xml remains as the final optimized version")

if __name__ == "__main__":
    train()