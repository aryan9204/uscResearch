"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np

def sa_to_index(state, action, n_actions):
    return state * n_actions + action

def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)
    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            idx = sa_to_index(s, a, transition_probabilities.shape[1])
            v[s] = sum(transition_probabilities[s, a, k] * 
                       (reward[idx] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))
    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Compute optimal Q-values (value function over state-action pairs).

    transition_probabilities: shape (n_states, n_actions, n_states)
    reward: vector of shape (n_states * n_actions,)
    -> Returns Q-table of shape (n_states, n_actions)
    """
    Q = np.zeros((n_states, n_actions))
    diff = float("inf")

    while diff > threshold:
        diff = 0
        Q_new = np.zeros_like(Q)

        for s in range(n_states):
            for a in range(n_actions):
                #idx = sa_to_index(s, a, n_actions)
                r_sa = reward[s, a]  # scalar
                tp = transition_probabilities[s, a, :]  # shape: (n_states,)
                expected_v = np.max(Q, axis=1)  # shape: (n_states,)
                Q_sa = r_sa + discount * np.dot(tp, expected_v)
                #print("Q_sa: ", Q_sa)
                Q_new[s, a] = Q_sa
                diff = max(diff, abs(Q_new[s, a] - Q[s, a]))

        Q = Q_new

    return Q

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, Q=None, stochastic=True):
    """
    Extract policy from Q-table.

    transition_probabilities: shape (n_states, n_actions, n_states)
    reward: shape (n_states * n_actions,)
    Q: Optional precomputed Q-table, otherwise computed via optimal_value.
    -> Returns policy of shape (n_states, n_actions) if stochastic,
       or shape (n_states,) if deterministic.
    """
    if Q is None:
        Q = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Softmax over actions for each state
        Q_stable = Q - np.max(Q, axis=1, keepdims=True)
        exp_Q = np.exp(Q_stable)
        policy = exp_Q / exp_Q.sum(axis=1, keepdims=True)
        print("Policy: ", policy)
        return policy
    else:
        # Deterministic: take action with highest Q-value
        return np.argmax(Q, axis=1)

if __name__ == '__main__':
    # Quick unit test using gridworld.
    import mdp.gridworld as gridworld
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v = value([gw.optimal_policy_deterministic(s) for s in range(gw.n_states)],
              gw.n_states,
              gw.transition_probability,
              [gw.reward(s) for s in range(gw.n_states)],
              gw.discount)
    assert np.isclose(v,
                      [5.7194282, 6.46706692, 6.42589811,
                       6.46706692, 7.47058224, 7.96505174,
                       6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(gw.n_states,
                          gw.n_actions,
                          gw.transition_probability,
                          [gw.reward(s) for s in range(gw.n_states)],
                          gw.discount)
    assert np.isclose(v, opt_v).all()
