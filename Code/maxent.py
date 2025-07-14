"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product

import numpy as np
import numpy.random as rn

# import value_iteration
from Code import value_iteration

def sa_to_index(state, action, n_actions):
    return state * n_actions + action

def softmax_vec(x):
    x = x - np.max(x)  # for stability
    exp_x = np.exp(x)
    return np.sum(x * (exp_x / np.sum(exp_x)))  # expected value under softmax

def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    n_sa, d_features = feature_matrix.shape
    n_states = n_sa // n_actions
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Number of states: {n_states}, Number of actions: {n_actions}, Feature dimension: {d_features}")

    alpha = rn.uniform(size=(d_features,))
    feature_expectations = find_feature_expectations(feature_matrix, trajectories, n_actions)

    for i in range(epochs):
        r = feature_matrix.dot(alpha)  # shape (n_sa,)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)
        alpha += learning_rate * grad
    print("-----------------OUTPUT-------------------")
    return feature_matrix.dot(alpha).reshape((n_states, n_actions))

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)


    for trajectory in trajectories:
        #for state, _, _ in trajectory:
        for state, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories, n_actions):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = np.zeros(feature_matrix.shape[1])
    for trajectory in trajectories:
        for state, action in trajectory:
            idx = sa_to_index(state, action, n_actions)
            feature_expectations += feature_matrix[idx]
    feature_expectations /= len(trajectories)
    return feature_expectations


def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # Get policy over (s,a)
    #print(transition_probability.shape)
    print(r.shape)
    r = r.reshape((n_states, n_actions))
    policy = value_iteration.find_policy(n_states, n_actions, transition_probability, r, discount)

    # Compute distribution over start (state, action) pairs from trajectories
    start_svf_sa = np.zeros(n_states * n_actions)
    for trajectory in trajectories:
        s0, a0 = trajectory[0]
        idx0 = sa_to_index(s0, a0, n_actions)
        start_svf_sa[idx0] += 1
    p_start_svf_sa = start_svf_sa / n_trajectories  # empirical start distribution over (s,a)

    # Initialize expected visitation frequency matrix for all timesteps
    expected_svf_sa = np.zeros((n_states * n_actions, trajectory_length))
    expected_svf_sa[:, 0] = p_start_svf_sa

    # Iterate over timesteps
    for t in range(1, trajectory_length):
        expected_svf_sa[:, t] = 0
        for s in range(n_states):
            for a in range(n_actions):
                idx = sa_to_index(s, a, n_actions)
                for s_prime in range(n_states):
                    for a_prime in range(n_actions):
                        idx_prime = sa_to_index(s_prime, a_prime, n_actions)
                        expected_svf_sa[idx_prime, t] += (
                            expected_svf_sa[idx, t-1] *
                            transition_probability[s, a, s_prime] *
                            policy[s_prime, a_prime]
                        )

    # Sum over all timesteps to get total expected visitation frequency per (s,a)
    expected_svf_sa_total = expected_svf_sa.sum(axis=1)

    return expected_svf_sa_total

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.zeros(n_states)
    threshold = 1e-4
    while True:
        V_new = np.zeros_like(V)
        for s in range(n_states):
            V_new[s] = softmax_vec(
                np.array([
                    r[s, a] + discount * np.sum(
                        transition_probability[s, a, s_prime] * V[s_prime]
                        for s_prime in range(n_states)
                    ) for a in range(n_actions)
                ])
            )
        if np.max(np.abs(V - V_new)) < threshold:
            break
        V = V_new

    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + discount * np.sum(
                transition_probability[s, a, s_prime] * V[s_prime]
                for s_prime in range(n_states)
            )

    Q -= Q.max(axis=1, keepdims=True)
    policy = np.exp(Q)
    policy /= policy.sum(axis=1, keepdims=True)
    print("Policy: ", policy)
    return policy

def expected_value_difference(n_states, n_actions, transition_probability,
    reward, discount, p_start_state, optimal_value, true_reward):
    """
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    reward: Reward vector mapping state int to reward. Shape (N,).
    discount: Discount factor. float.
    p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    """

    policy = value_iteration.find_policy(n_states, n_actions,
        transition_probability, reward, discount)
    value = value_iteration.value(policy.argmax(axis=1), n_states,
        transition_probability, true_reward, discount)

    evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
    return evd
