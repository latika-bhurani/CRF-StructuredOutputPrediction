import numpy as np
import math
import re
from scipy.optimize import check_grad
import input as ip
import message_passing as mp
import time

class CRF:

    # sequence decoder : 1c
    def decode(self, transitions, weights, features, n):

        M = np.zeros((26, n))

        # for most probable sequence
        M_sequence = []

        # for every letter in position j
        for j in range(26):
            M[j][0] = np.inner(weights[j], features[0])

        M_sequence.append(np.argmax(M[:,0]))

        # decoder to fetch best possible sequence from position 1 to n
        for i in range(1, n):

            # for y_i
            for next_state in range(26):

                max_until_now = -999999999
                feature_weight_dot_product = np.inner(weights[next_state], features[i])

                # check for all possible values of y_(i-1)
                for prev_state in range(26):

                    value = M[prev_state][i - 1] + feature_weight_dot_product + transitions[prev_state][next_state]

                    if value > max_until_now:
                        max_until_now = value

                M[next_state][i] = max_until_now

            M_sequence.append(np.argmax(M[:,i]))

        return M, M_sequence


    # calculate log probability of given sequence
    def log_probability(self, y_seq, X, w, t):

        w_x_product = np.inner(X, w)
        numerator = self.numerator(w_x_product, t, y_seq)
        z = self.partition_fn(w_x_product, t, len(y_seq))

        return np.log(numerator/z)


    # calculate numerator for given sequence
    def numerator(self, w_x_product, t, y_seq):

        num_value = w_x_product[0][y_seq[0]]

        for i in range(1, len(y_seq)):
            num_value += w_x_product[i][y_seq[i]] + t[y_seq[i-1]][y_seq[i]]

        return np.exp(num_value)

    # calculate average log probability across training data
    def average_log_probability(self, params, X, y, data_count):

        w = ip.w_matrix(params)
        t = ip.t_matrix(params)

        probability = [self.log_probability(y[i], X[i], w, t) for i in range(data_count)]
        #print(probability)
        return np.sum(probability) / data_count

    def compute_average_gradient_wy(params, X, y):

        w = ip.w_matrix(params)
        t = ip.t_matrix(params)

        lenY = 5

        gradient = [gd.calculate_gradient_w_y(w, t, X[i], y[i]) for i in range(lenY)]

        return np.sum(gradient)/lenY

    # calculate normalizer
    def partition_fn(self, w_x, t, data_count):

        denominator_f = np.sum(np.exp(w_x[-1] + mp.forward_propagation(w_x, t)[-1]))
        return denominator_f

    # calculate gradient wrt w_y
    def calculate_gradient_w_y(self, w_x_product, t, X, y_seq, z, f, b):

        gradient = np.zeros((128, 26))

        for i in range(len(y_seq)) :
            prob_marginal = np.exp(f[i] + b[i] + (w_x_product[i]))

            for letter in range(26):
                if y_seq[i] == letter:



                    gradient[:, letter] += X[i]
                gradient[:, letter] -= prob_marginal[letter]/z * X[i]

        return gradient.transpose().flatten()

    # to change - copied Erik's code for this
    # calculate gradient wrt to t
    def calculate_gradient_t(self, w_x_product, t, X, y_seq, z, f, b):
        # #
        # gradient = np.zeros((26, 26))
        # for i in range(len(y_seq) - 1):
        #
        #     for j in range(26):
        #         gradient[j] -= np.exp(w_x_product[i] + t.transpose()[j] + w_x_product[i + 1] + b[i + 1] + f[i])
        #
        #     prob_marginal = np.exp(f[i] + b[i+1] + (w_x_product[i] + w_x_product[i+1] + t.transpose()))
        #     gradient -=  prob_marginal.transpose() / z
        #
        #     gradient[y_seq[i]][y_seq[i+1]] += 1
        #
        # return gradient.transpose().flatten()
        # return gradient.transpose().flatten()
        #
        # gradient = np.zeros(26 * 26)
        # grad_matrix = np.zeros((26, 26))
        # t = t.transpose()
        # for i in range(len(w_x_product) - 1):
        #     for j in range(26):
        #         gradient[j * 26: (j + 1) * 26] -= np.exp(
        #             w_x_product[i] + t[j] + w_x_product[i + 1][j] + b[i + 1][j] + f[i])
        #
        #         grad_matrix[j] -= np.exp(w_x_product[i] + t[j] + w_x_product[i + 1] + b[i + 1] + f[i])
        # gradient /= z
        #
        # for i in range(len(w_x_product) - 1):
        #     t_index = y_seq[i]
        #     t_index += 26 * y_seq[i + 1]
        #     gradient[t_index] += 1
        #
        # return gradient

        gradient = np.zeros((26 * 26))
        for i in range(len(w_x_product) - 1):
            for j in range(26):
                gradient[j * 26: (j + 1) * 26] -= np.exp(
                    w_x_product[i] + t.transpose()[j] + w_x_product[i + 1][j] + b[i + 1][j] + f[i])

        gradient /= z

        for i in range(len(w_x_product) - 1):
            t_index = y_seq[i]
            t_index += 26 * y_seq[i + 1]
            gradient[t_index] += 1

        return gradient

    # calculate gradient per word
    def gradient_per_sequence(self, w, t, X, y_seq):

        w_x_product = np.inner(X, w)

        #normalizer
        z = self.partition_fn(w_x_product, t, len(y_seq))

        #message passing
        fwd_message = mp.forward_propagation(w_x_product, t)
        bwd_message = mp.backward_propagation(w_x_product, t)

        # gradient wrt w_y
        grad_w_y = self.calculate_gradient_w_y(w_x_product, t, X, y_seq, z, fwd_message, bwd_message)

        # gradient wrt t
        grad_t = self.calculate_gradient_t(w_x_product, t, X, y_seq, z, fwd_message, bwd_message)
        return np.concatenate((grad_w_y, grad_t))

    # compute average gradient
    def compute_average_gradient(self, params, X, y, data_count):

        w = ip.w_matrix(params)
        t = ip.t_matrix(params)
        overall_gradient = np.zeros(128*26 + 26*26)
        for i in range(data_count):
            overall_gradient += self.gradient_per_sequence(w, t, X[i], y[i])
        return overall_gradient / (data_count)

    def brute_force(self, X, params):
        w = ip.w_matrix(params)
        t = ip.t_matrix(params)
        val = 0
        for i in range(26):
            for j in range(26):
                for k in range(26):
                    val += np.exp(np.dot(w[k], X[0]) + np.dot(w[j], X[1]) + np.dot(w[i], X[2]) + t[k][j] + t[j][i])
        print(val)

decoder = CRF()
y_tot, X_tot = ip.read_data_formatted()
#
params = ip.get_params()
# start = time.time()
# print(start)
# #
# print(check_grad(decoder.average_log_probability,
#             decoder.compute_average_gradient,
#                  params, X_tot, y_tot, 5))

#print(decoder.compute_average_gradient(params, X_tot, y_tot, 1))
print(decoder.average_log_probability(params, X_tot, y_tot, len(y_tot)))
#
# print(decoder.compute_average_gradient(params, X_tot, y_tot, len(y_tot)))
# print("Total time:" + str(time.time() - start))
# #decoder.brute_force(X_tot[0],  params)

# X, w, t = ip.get_weights_formatted()
# print(decoder.decode(t, w, X, 100))