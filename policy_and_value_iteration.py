import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob

    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """
        Run value iteration (You probably need no more than 30 lines)
        Input Arguments
        ----------
            env:
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration
        ----------
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])

    ##### FINISH TODOS HERE #####
    V = np.zeros(num_spaces)
    R, P = get_rewards_and_transitions_from_env(env)

    # V_k+1(s) = max_a(sum_s_(R(s, s_, a) + gamma * P(s, s_, a) * V_k(s_)))
    for k in range(max_iterations):
        V_prev = V.copy()                 # V_prev = V_k, V = V_k+1
        for s in range(num_spaces):
            max_v, max_a = 0, 0
            for a in range(num_actions):
                val = sum(R[s][a]) + gamma * np.dot(P[s][a], V_prev)
                if val > max_v:
                    max_v, max_a = val, a
            V[s] = max_v
            policy[s] = max_a

        if max(abs(V - V_prev)) < eps:    # early stopping
            break

    #############################

    # Return optimal policy
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """
        Run policy iteration (You probably need no more than 30 lines)
        Input Arguments
        ----------
            env:
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation
        ----------
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])

    ##### FINISH TODOS HERE #####
    V = np.zeros(num_spaces)
    R, P = get_rewards_and_transitions_from_env(env)

    # 1. V_k(pi, s) = sum_a(pi(s, a) * (R(s, a) + gamma * sum(P(s, s_, a) * V(pi, k-1, s_)))) , pi(s, a) = 1 when a = policy[s], = 0 ohers
    # 2. Q_pi_k(s, a) = R(s, a) + gamma * sum(s_) * P_s_a(s_) * V_pi_k(s_)
    # 3. pi_k+1(s) = argmax_a Q_pi_k(s,a)
    while True:
        # policy evaluation
        # V_pi_0 -> V_pi_1 -> .... -> V_pi_k
        policy_prev = policy.copy()           # policy_prev = pi_k, policy = pi_k+1
        for k in range(max_iterations):
            V_prev = V.copy()                 # V_prev = V_k, V = V_k+1
            for s in range(num_spaces):
                V[s] = sum(R[s][policy[s]]) + gamma * np.dot(P[s][policy[s]], V_prev)

            if max(abs(V - V_prev)) < eps:    # early stopping
                break
        # policy improvement
        # Q with argmax a
        for s in range(num_spaces):
            max_qv, max_a = 0, 0
            for a in range(num_actions):
                q_val = sum(R[s][a]) + gamma * np.dot(P[s][a], V)
                if q_val > max_qv:
                    max_qv, max_a = q_val, a
            policy[s] = max_a

        if (policy == policy_prev).all():
            break
    #############################

    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """
        Enforce policy iteration and value iteration
    """
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)

    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])
    print('Discrepancy:', diff)