import pickle
import sys

checkpoint_path = "experiments/2025-06-09_23-18-49_game_theoretic_moe_enriched_attempt_0/logs/0-run/stage_1_initial_implementation_1_preliminary/checkpoint.pkl"

with open(checkpoint_path, 'rb') as f:
    data = pickle.load(f)
    
print("Checkpoint keys:", list(data.keys()) if isinstance(data, dict) else type(data))
if isinstance(data, dict):
    for key, value in data.items():
        print(f"\n{key}:")
        if isinstance(value, (list, dict)):
            print(f"  Type: {type(value)}, Length/Size: {len(value)}")
        else:
            print(f"  Type: {type(value)}")
