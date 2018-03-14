import re
import numpy as np

def read_input(self):
    i = 0
    with open("../data/decode_input.txt") as file:
        for line in file:
            print(i)
            i = i + 1

def read_data():
    file = open('../data/train_struct.txt', 'r')
    y = []
    qids = []
    X = []

    for line in file:
        temp = line.split()
        # get label
        label_string = temp[0]
        # normalizes y to 0...25 instead of 1...26
        y.append(int(label_string) - 1)

        # get qid
        qid_string = re.split(':', temp[1])[1]
        qids.append(int(qid_string))

        # get x values
        x = np.zeros(128)

        # do we need a 1 vector?
        # x[128] = 1
        for elt in temp[2:]:
            index = re.split(':', elt)
            x[int(index[0]) - 1] = 1
        X.append(x)
    y = np.array(y)
    qids = np.array(qids)
    return y, qids, X

def read_data_formatted():
    # get initial output
    y, qids, X = read_data()
    y_tot = []
    X_tot = []
    current = 0;

    y_tot.append([])
    X_tot.append([])

    for i in range(len(y)):
        y_tot[current].append(y[i])
        X_tot[current].append(X[i])

        if (i + 1 < len(y) and qids[i] != qids[i + 1]):
            y_tot[current] = np.array(y_tot[current])
            X_tot[current] = np.array(X_tot[current])
            y_tot.append([])
            X_tot.append([])
            current = current + 1

    return y_tot, X_tot

def w_matrix(params):
    w = np.zeros((26, 128))
    for i in range(26):
        w[i] = params[128 * i: 128 * (i + 1)]
    return w

def t_matrix(params):
    t = np.zeros((26, 26))
    count = 0
    for i in range(26):
        for j in range(26):
            # want to be able to say t[0] and get all values of Taa, tba, tca...
            t[j][i] = params[128 * 26 + count]
            count += 1
    return t

def get_weights():
    file = open('../data/decode_input.txt', 'r')
    x_array = []
    w_array = []
    t_array = []
    for i, elt in enumerate(file):
        if(i < 100 * 128):
            x_array.append(elt)
        elif(i < 100 * 128 + 128 * 26):
            w_array.append(elt)
        else:
            t_array.append(elt)
    return x_array, w_array, t_array

def get_params():
    file = open('../data/model.txt', 'r')
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)

def parse_x(x):
    count = 0
    x_array = np.zeros((100, 128))
    for i in range(100):
        for j in range(128):
            x_array[i][j] = x[count]
            count += 1
    return x_array

def parse_w(w):
    w_array = np.zeros((26, 128))
    count = 0
    for i in range(26):
        for j in range(128):
            w_array[i][j] = w[count]
            count += 1
    return w_array

def parse_t(t):
    t_array = np.zeros((26, 26))
    count = 0
    for i in range(26):
        for j in range(26):
            #this is actuqlly right.  it goes T11, T21, T31...
            t_array[j][i] = t[count]
            count += 1
    return t_array

def get_weights_formatted():
    x, w, t = get_weights()
    x_array = parse_x(x)
    w_array = parse_w(w)
    t_array = parse_t(t)
    return x_array, w_array, t_array


