# -----------------------------------------------------------
# python code used to develop programmed versions of comp, dd
# and scomp group testing decoding schemes along with the 
# novel gtsvm decoding scheme that utilises machine learning
# -----------------------------------------------------------

import numpy as np
from sklearn import svm

# comp defective set estimate
# test_matrix - (T, n)
# test_outcomes - (T, 1)
def comp_decoder(test_matrix, test_outcomes):
    neg_tests = np.where(test_outcomes == 0)[0]
    dnd_indices = np.sort(np.unique(np.where(test_matrix[neg_tests] == 1)[1]))
    pd_indices = np.delete(np.arange(0, test_matrix.shape[1], 1), dnd_indices)
    return pd_indices + 1

# dd defective set estimate
# test_matrix - (T, n)
# test_outcomes - (T, 1)
def dd_decoder(test_matrix, test_outcomes):
    pd_indices = comp_decoder(test_matrix, test_outcomes) - 1
    pos_tests = np.delete(np.arange(0, test_matrix.shape[0], 1), np.where(test_outcomes == 0)[0])
    trunc_matrix = test_matrix.T[pd_indices].T[pos_tests]
    test_indices = np.where(np.sum(trunc_matrix, axis = 1) == 1)[0]
    dd_indices = pd_indices[np.unique(np.where(trunc_matrix[test_indices] == 1)[1])]
    return dd_indices + 1

# scomp defective set estimate
# test_matrix - (T, n)
# test_outcomes - (T, 1)
def scomp_decoder(test_matrix, test_outcomes):
    dd_indices = dd_decoder(test_matrix, test_outcomes) - 1
    scomp_indices = dd_indices
    pd_indices = comp_decoder(test_matrix, test_outcomes) - 1
    pos_tests = np.delete(np.arange(0, test_matrix.shape[0], 1), np.where(test_outcomes == 0)[0])
    unexplained_tests = pos_tests[np.where(np.sum(test_matrix[pos_tests].T[dd_indices].T, axis = 1) == 0)[0]]
    while(len(unexplained_tests) > 0):
        ignored_indices = np.nonzero(np.in1d(pd_indices, scomp_indices))[0]
        most_unexplained = np.argmax(np.sum(test_matrix[unexplained_tests].T[np.delete(pd_indices, ignored_indices)].T, axis = 0))
        scomp_indices = np.append(scomp_indices, np.delete(pd_indices, ignored_indices)[most_unexplained])
        unexplained_tests = np.delete(unexplained_tests, 
        np.where(test_matrix[unexplained_tests].T[np.delete(pd_indices, ignored_indices)][most_unexplained] == 1)[0])
    return np.sort(scomp_indices) + 1

# gtsvm defective set estimate
# classifier - svm.SVC()
# test_matrix - (T, n)
# test_outcomes - (T, 1)
def svc_decoder(classifier, test_matrix, test_outcomes):
    defects = np.array([], dtype=int)
    x = [np.hstack((test_matrix, test_outcomes)).flatten()]
    y = classifier.predict(x)
    binary = bin(int(y[0])).replace("0b", "")[::-1]
    for i in range(len(binary)):
        if (binary[i] == "1"):
            defects = np.append(defects, i + 1)
    return defects

# randomly generated group test
# num_items - n
# num_tests - T
# num_defect - k
# rng - np.random.default_rng()
def test_generator(num_items, num_tests, num_defect, rng):
    prob = 1.0 / num_defect if num_defect != 1 else 0.5
    test_defectives = np.sort(rng.choice(np.arange(1, num_items + 1), num_defect, replace=False))
    defectivity_vector = np.zeros((1, num_items), dtype=int)
    np.put(defectivity_vector, test_defectives - 1, 1)
    test_matrix = rng.choice([0, 1], (num_tests, num_items), p=[1 - prob, prob])
    test_outcomes = np.array([np.where(np.sum(test_matrix * defectivity_vector, axis=1) > 0, 1, 0)])
    return test_matrix, test_outcomes.T, test_defectives

# trains the gtsvm decoder with randomly generated group tests
# num_items - n
# num_tests - T
# num_defect - k
# size - training set size 
# rng - np.random.default_rng()
def train_svc(num_items, num_tests, num_defect, size, rng):
    x_train = np.empty((size, (num_items + 1) * num_tests))
    y_train = np.empty(size)
    for i in range(size):
        gen = test_generator(num_items, num_tests, num_defect, rng)
        x_train[i] = np.hstack((gen[0], gen[1])).flatten()
        y = 0
        for j in gen[2]:
            y += 2 ** (j-1)
        y_train[i] = y
    classifier = svm.SVC()
    classifier.fit(x_train, y_train)
    return classifier
    