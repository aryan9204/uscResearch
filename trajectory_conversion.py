import numpy as np
import re
from Code import value_iteration
from Code import maxent

def convert_raw_trajectories():
    processed_trajectories = []
    raw_trajectories = {}
    with open("outputHalfHomophily.txt", 'r') as f:
        for line in f:
            split_line = line.split(': ')
            traj = eval(split_line[1])
            raw_trajectories[split_line[0]] = traj

    for agent, traj in raw_trajectories.items():
        trajectory = []
        for s_str, a_str in traj:
            # Extract integers from strings like 's=1' and 'a=0'
            state = int(re.search(r"\d+", s_str).group())
            action = int(re.search(r"\d+", a_str).group())
            trajectory.append([state, action])
        processed_trajectories.append(trajectory)

    # Convert to NumPy array
    trajectories_np = np.array(processed_trajectories)
    return trajectories_np

def compute_transition_probabilities(trajectories, n_states, n_actions):
    transition_counts = np.zeros((n_states, n_actions, n_states))

    for traj in trajectories:
        for i in range(len(traj) - 1):
            s, a = traj[i]
            s_next, _ = traj[i + 1]
            transition_counts[s, a, s_next] += 1

    transition_probs = np.zeros_like(transition_counts, dtype=np.float64)

    for s in range(n_states):
        for a in range(n_actions):
            total = np.sum(transition_counts[s, a])
            if total > 0:
                transition_probs[s, a] = transition_counts[s, a] / total

    return transition_probs

def get_feature_matrix(n_states, n_actions):
    feature_matrix = np.eye(n_states * n_actions)
    return feature_matrix

trajectories_np = convert_raw_trajectories()
transition_probs = compute_transition_probabilities(trajectories_np, 5, 3)
#print(transition_probs)
feature_matrix = get_feature_matrix(5, 3)


print(maxent.irl(feature_matrix, 3, 0.5, transition_probs, trajectories_np, 1000, 0.01))

