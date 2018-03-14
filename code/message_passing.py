import numpy as np

# message passing : forward propagation
def forward_propagation(w_x, transitions):

    alpha = np.zeros((len(w_x), 26));
    for j in range(1, len(w_x)):
        for next_state in range(26):
            # log-sum-exp
            f_j = w_x[j-1] + transitions[:, next_state] + alpha[j-1]
            max_f = np.max(f_j)
            alpha[j][next_state] = max_f + np.log(np.sum(np.exp(f_j-max_f)))

    return alpha

def backward_propagation(w_x, transitions):
    beta = np.zeros((len(w_x), 26));
    for j in range(len(w_x) - 2, -1, -1):
        for current in range(26):
            f_j = transitions[current] + beta[j + 1] + w_x[j + 1]
            max_f = np.max(f_j)
            beta[j][current] = max_f + np.log(np.sum(np.exp(f_j - max_f)))

    return beta