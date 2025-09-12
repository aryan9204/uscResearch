import numpy as np
import re
from Code import value_iteration
from Code import maxent
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def convert_raw_trajectories():
    processed_trajectories = []
    # processed_trajectories_against = []
    # processed_trajectories_support = []
    # processed_trajectories_neutral = []
    with open("trajectories/tenRuns/oneHomophilyAllUsersAgainst.txt", 'r') as f:
        content = f.read()
        for i in range(1, 11):
            pattern = rf'RUN {i}\n(.*?)(?=RUN {i + 1}|$)'
            match = re.search(pattern, content, re.DOTALL)
            curr_trajectory = []
            raw_trajectories = {}
            # raw_trajectories_support = {}
            # raw_trajectories_against = {}
            # raw_trajectories_neutral = {}
            # curr_trajectory_against = []
            # curr_trajectory_support = []
            # curr_trajectory_neutral = []
            
            if match:
                text = match.group(1)
                lines = text.split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    split_line = line.split(': ')
                    traj = eval(split_line[1])
                    raw_trajectories[split_line[0]] = traj
                    # df = pd.read_csv('persona_5point.csv')
                    # splitName = split_line[0].split('_')
                    # if len(splitName) < 2:
                    #     name = splitName[0]
                    # else:
                    #     name = splitName[0] + ' ' + splitName[1]
                    # user_opinion = df.loc[df['Name'] == name, 'Opinion'].iloc[0]
                    # if user_opinion < 0:
                    #     raw_trajectories_against[split_line[0]] = traj
                    # elif user_opinion == 0:
                    #     raw_trajectories_neutral[split_line[0]] = traj
                    # else:
                    #     raw_trajectories_support[split_line[0]] = traj
                
                for agent, traj in raw_trajectories.items():
                    trajectory = []
                    for s_str, a_str in traj:
                        # Extract integers from strings like 's=1' and 'a=0'
                        state = int(re.search(r"\d+", s_str).group())
                        action = int(re.search(r"\d+", a_str).group())
                        trajectory.append([state, action])
                    curr_trajectory.append(trajectory)

                # for agent, traj in raw_trajectories_against.items():
                #     trajectory = []
                #     for s_str, a_str in traj:
                #         # Extract integers from strings like 's=1' and 'a=0'
                #         state = int(re.search(r"\d+", s_str).group())
                #         action = int(re.search(r"\d+", a_str).group())
                #         trajectory.append([state, action])
                #     curr_trajectory_against.append(trajectory)
                
                # for agent, traj in raw_trajectories_support.items():
                #     trajectory = []
                #     for s_str, a_str in traj:
                #         # Extract integers from strings like 's=1' and 'a=0'
                #         state = int(re.search(r"\d+", s_str).group())
                #         action = int(re.search(r"\d+", a_str).group())
                #         trajectory.append([state, action])
                #     curr_trajectory_support.append(trajectory)

                # for agent, traj in raw_trajectories_neutral.items():
                #     trajectory = []
                #     for s_str, a_str in traj:
                #         # Extract integers from strings like 's=1' and 'a=0'
                #         state = int(re.search(r"\d+", s_str).group())
                #         action = int(re.search(r"\d+", a_str).group())
                #         trajectory.append([state, action])
                #     curr_trajectory_neutral.append(trajectory)

        # Convert to NumPy array
                trajectories_np = np.array(curr_trajectory)
                processed_trajectories.append(trajectories_np)
                # trajectories_np_against = np.array(curr_trajectory_against)
                # processed_trajectories_against.append(trajectories_np_against)
                # trajectories_np_support = np.array(curr_trajectory_support)
                # processed_trajectories_support.append(trajectories_np_support)
                # trajectories_np_neutral = np.array(curr_trajectory_neutral)
                # processed_trajectories_neutral.append(trajectories_np_neutral)
    return processed_trajectories

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

#Heatmap code
trajectories_np = convert_raw_trajectories()
rewards = []
for trajectories in trajectories_np:
    transition_probs = compute_transition_probabilities(trajectories, 5, 3)
    feature_matrix = get_feature_matrix(5, 3)
    reward = maxent.irl(feature_matrix, 3, 0.5, transition_probs, trajectories, 1000, 0.01)
    rewards.append(reward)

average_reward = np.mean(np.stack(rewards), axis=0)

plt.figure(figsize=(8, 5))
ax = sns.heatmap(
    average_reward,
    annot=True, fmt=".2f", cmap="YlGnBu",
    xticklabels=["Decrease Willingness", "Stay the Same", "Increase Willingness"],
    yticklabels=["No Opposition", "Low Opposition", "Moderate Opposition", "High Opposition", "Extreme Opposition"],
    cbar_kws={"label": "Reward"}
)

plt.title("One Homophily (More Users Against)")
plt.xlabel("Actions")
plt.ylabel("States (Degree of Opposition)")
plt.tight_layout()
plt.savefig("heatmaps/one_homophily_more_users_against.png")


#Boxplot code for single group
# trajectories_np = convert_raw_trajectories()
# rewards = []
# for trajectories in trajectories_np:
#     transition_probs = compute_transition_probabilities(trajectories, 5, 3)
#     #print(transition_probs)
#     feature_matrix = get_feature_matrix(5, 3)
#     reward = maxent.irl(feature_matrix, 3, 0.5, transition_probs, trajectories, 1000, 0.01)
#     # with open("testBoxplot.txt", 'a') as f:
#     #     f.write(str(reward) + '\n')
#     rewards.append(reward)

# boxplot = np.stack(rewards)
# state_action_rewards = [boxplot[:, i, j] for i in range(5) for j in range(3)]

# # Labels
# labels = ['NO+DW', 'NO+SS', 'NO+RW', 'LO+DW', 'LO+SS', 'LO+RW', 'MO+DW', 'MO+SS', 'MO+RW', 'HO+DW', 'HO+SS', 'HO+RW', 'EO+DW', 'EO+SS', 'EO+RW']

# # Plot
# plt.figure(figsize=(12,6))
# plt.boxplot(state_action_rewards, labels=labels)
# plt.ylabel('Reward')
# plt.title('Reward distribution per state-action pair')
# plt.xticks(rotation=45)
# plt.show()
# plt.savefig('test_boxplot.png') # Save the figure

# average_reward = np.mean(np.stack(rewards), axis=0)
# print("Average Reward:", average_reward)

#Grouped boxplot code
# processed_trajectories_against, processed_trajectories_support, processed_trajectories_neutral = convert_raw_trajectories()
# againstRewards = []
# supportRewards = []
# neutralRewards = []
# for trajectories in processed_trajectories_against:
#     transition_probs = compute_transition_probabilities(trajectories, 5, 3)
#     #print(transition_probs)
#     feature_matrix = get_feature_matrix(5, 3)
#     reward = maxent.irl(feature_matrix, 3, 0.5, transition_probs, trajectories, 1000, 0.01)
#     # with open("testBoxplot.txt", 'a') as f:
#     #     f.write(str(reward) + '\n')
#     againstRewards.append(reward)

# for trajectories in processed_trajectories_support:
#     transition_probs = compute_transition_probabilities(trajectories, 5, 3)
#     #print(transition_probs)
#     feature_matrix = get_feature_matrix(5, 3)
#     reward = maxent.irl(feature_matrix, 3, 0.5, transition_probs, trajectories, 1000, 0.01)
#     # with open("testBoxplot.txt", 'a') as f:
#     #     f.write(str(reward) + '\n')
#     supportRewards.append(reward)

# for trajectories in processed_trajectories_neutral:
#     transition_probs = compute_transition_probabilities(trajectories, 5, 3)
#     #print(transition_probs)
#     feature_matrix = get_feature_matrix(5, 3)
#     reward = maxent.irl(feature_matrix, 3, 0.5, transition_probs, trajectories, 1000, 0.01)
#     # with open("testBoxplot.txt", 'a') as f:
#     #     f.write(str(reward) + '\n')
#     neutralRewards.append(reward)

# boxplotAgainst = np.stack(againstRewards)
# boxplotSupport = np.stack(supportRewards)
# boxplotNeutral = np.stack(neutralRewards)   
# final_against_rewards = [boxplotAgainst[:, i, j] for i in range(5) for j in range(3)]
# final_support_rewards = [boxplotSupport[:, i, j] for i in range(5) for j in range(3)]
# final_neutral_rewards = [boxplotNeutral[:, i, j] for i in range(5) for j in range(3)]
# boxplots = [final_against_rewards, final_support_rewards, final_neutral_rewards]

# # Labels
# labels = ['NO+DW', 'NO+SS', 'NO+RW', 'LO+DW', 'LO+SS', 'LO+RW', 'MO+DW', 'MO+SS', 'MO+RW', 'HO+DW', 'HO+SS', 'HO+RW', 'EO+DW', 'EO+SS', 'EO+RW']

# # def melt_rewards(boxplot, group_name):
# #     data = []
# #     for i in range(5):
# #         for j in range(3):
# #             for val in boxplot[:, i, j]:
# #                 data.append({
# #                     "StateAction": labels[i*3 + j],
# #                     "Reward": val,
# #                     "Group": group_name
# #                 })
# #     return pd.DataFrame(data)

# # df = pd.concat([
# #     melt_rewards(boxplotAgainst, "Against"),
# #     melt_rewards(boxplotSupport, "Support"),
# #     melt_rewards(boxplotNeutral, "Neutral"),
# # ])

# plt.figure(figsize=(12,6))
# # sns.boxplot(data=df, x="StateAction", y="Reward", hue="Group")
# plt.boxplot(final_neutral_rewards, labels=labels)
# plt.ylabel('Reward')
# plt.xticks(rotation=45)
# plt.title("Reward distribution per state-action pair for Neutral Agents")
# plt.savefig('test_boxplot_neutral.png')

# # average_reward = np.mean(boxplotAgainst, axis=0)
# # print("Average Reward (Against):", average_reward)

# # average_reward = np.mean(boxplotSupport, axis=0)
# # print("Average Reward (Support):", average_reward)

# # average_reward = np.mean(boxplotNeutral, axis=0)
# # print("Average Reward (Neutral):", average_reward)
