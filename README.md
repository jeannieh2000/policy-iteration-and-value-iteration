# policy-iteration-and-value-iteration
## Environment
OpenAI Gym ([Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/)).

## Policy iteration Derivation

We consider a Markov Decision Process (MDP) defined by:
- State space: S
- Action space: A
- Transition probability: P(s' | s, a)
- Reward function: R(s, a)
- Discount factor: γ ∈ (0, 1)

### Value Function Definition


For a given policy π(a | s), the state-value function is defined as:

    V^π(s) = E_π [ ∑_{t=0}^{∞} γ^t r_t | s_0 = s ]

The action-value function is:

    Q^π(s, a) = E_π [ ∑_{t=0}^{∞} γ^t r_t | s_0 = s, a_0 = a ]

### Bellman Expectation Equations

State-value function:

    V^π(s) = ∑_a π(a | s) [ R(s, a) + γ ∑_{s'} P(s' | s, a) V^π(s') ]

Action-value function:

    Q^π(s, a) = R(s, a) + γ ∑_{s'} P(s' | s, a) ∑_{a'} π(a' | s') Q^π(s', a')

### Policy Evaluation

Given a fixed policy π_k, we evaluate its value function by solving:

    V^{π_k}(s) = ∑_a π_k(a | s) [ R(s, a) + γ ∑_{s'} P(s' | s, a) V^{π_k}(s') ]

This can be computed by iterative updates:

    V_{k+1}(s) ← ∑_a π_k(a | s) [ R(s, a) + γ ∑_{s'} P(s' | s, a) V_k(s') ]

until convergence.

### Policy Improvement

Given V^{π_k}, we define a new policy π_{k+1} by acting greedily:

    π_{k+1}(s) = argmax_a Q^{π_k}(s, a)

where:

    Q^{π_k}(s, a) = R(s, a) + γ ∑_{s'} P(s' | s, a) V^{π_k}(s')

### Policy Improvement Theorem

For all s ∈ S:

    Q^{π_k}(s, π_{k+1}(s)) ≥ V^{π_k}(s)

This implies:

    V^{π_{k+1}}(s) ≥ V^{π_k}(s),  ∀ s ∈ S

Thus, π_{k+1} is guaranteed to be no worse than π_k.

### Policy Iteration Algorithm

Initialize policy π_0 arbitrarily.

Repeat until convergence:

(1) Policy Evaluation:
    Compute V^{π_k} by iterative Bellman updates:
        V(s) ← ∑_a π_k(a | s) [ R(s, a) + γ ∑_{s'} P(s' | s, a) V(s') ]

(2) Policy Improvement:
    Update policy greedily:
        π_{k+1}(s) = argmax_a Q^{π_k}(s, a)

If π_{k+1} = π_k, terminate.

Return optimal policy π*.


## Value iteration
### Value Iteration: Derivation

We start from the Bellman optimality equation for the optimal value function:

    V*(s) = max_a [ R(s, a) + γ ∑_{s'} P(s' | s, a) V*(s') ]

Define the Bellman optimality operator T as:

    (T V)(s) = max_a [ R(s, a) + γ ∑_{s'} P(s' | s, a) V(s') ]

The optimal value function V* is the unique fixed point of T:

    V* = T V*

Value Iteration applies this operator iteratively:

    V_{k+1} = T V_k

which expands to:

    V_{k+1}(s)
        = max_a [ R(s, a) + γ ∑_{s'} P(s' | s, a) V_k(s') ]

Under standard assumptions (0 ≤ γ < 1), the Bellman operator T is a contraction mapping, thus:

    lim_{k → ∞} V_k = V*

After convergence, the optimal policy can be extracted by:

    π*(s) = argmax_a [ R(s, a) + γ ∑_{s'} P(s' | s, a) V*(s') ]

---

### Value Iteration Algorithm

Initialize V(s) arbitrarily for all states s.

Repeat until convergence:

(1) For each state s:
        V_new(s) = max_a [ R(s, a) + γ ∑_{s'} P(s' | s, a) V(s') ]

(2) Check stopping condition:
        if max_s | V_new(s) - V(s) | < ε:
            break

(3) Update:
        V(s) ← V_new(s)

Return V* = V(s).

Extract optimal policy:

    π*(s) = argmax_a [ R(s, a) + γ ∑_{s'} P(s' | s, a) V*(s') ]
