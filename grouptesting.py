import math
import numpy as np
from matplotlib import pyplot as plt
rng = np.random.default_rng()

#input test_matrix of shape (T, n) and test_outcomes of shape (T, 1)
#output COMP defective set estimate
def comp_decoder(test_matrix, test_outcomes):
    neg_tests = np.where(test_outcomes == 0)[0]
    dnd_indices = np.sort(np.unique(np.where(test_matrix[neg_tests] == 1)[1]))
    pd_indices = np.delete(np.arange(0, test_matrix.shape[1], 1), dnd_indices)
    return pd_indices + 1

#input test_matrix of shape (T, n) and test_outcomes of shape (T, 1)
#output DD defective set estimate
def dd_decoder(test_matrix, test_outcomes):
    pd_indices = comp_decoder(test_matrix, test_outcomes) - 1
    pos_tests = np.delete(np.arange(0, test_matrix.shape[0], 1), np.where(test_outcomes == 0)[0])
    trunc_matrix = test_matrix.T[pd_indices].T[pos_tests]
    test_indices = np.where(np.sum(trunc_matrix, axis = 1) == 1)[0]
    dd_indices = pd_indices[np.unique(np.where(trunc_matrix[test_indices] == 1)[1])]
    return dd_indices + 1

#input test_matrix of shape (T, n) and test_outcomes of shape (T, 1)
#output SCOMP defective set estimate
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

#input n, T and k
#output test matrix, test outcomes and defective set according to Bernoulli design
def test_generator(num_items, num_tests, num_defect):
    prob = 1.0 / num_defect
    test_defectives = np.sort(rng.choice(np.arange(1, num_items + 1), num_defect, replace=False))
    defectivity_vector = np.zeros((1, num_items), dtype=int)
    np.put(defectivity_vector, test_defectives - 1, 1)
    test_matrix = rng.choice([0, 1], (num_tests, num_items), p=[1 - prob, prob])
    test_outcomes = np.array([np.where(np.sum(test_matrix * defectivity_vector, axis=1) > 0, 1, 0)])
    return test_matrix, test_outcomes.T, test_defectives

#input n, T and k along with number of samples
#output empirical probability of success over these samples
def empirical_rate(num_items, num_tests, num_defect, size):
    correct_tests = np.array([[0, 0, 0]])
    for i in range(size):
        sample = test_generator(num_items, num_tests, num_defect)
        correct_tests[0][0] += np.array_equal(comp_decoder(sample[0], sample[1]), sample[2])
        correct_tests[0][1] += np.array_equal(dd_decoder(sample[0], sample[1]), sample[2])
        correct_tests[0][2] += np.array_equal(scomp_decoder(sample[0], sample[1]), sample[2])
    return(correct_tests/float(size))

#input fixed n and k
#output list of empirical rates as T varies
def vary_tests(num_items, num_defect, size):
    results = np.zeros((num_items, 3))
    for i in range(1, num_items + 1):
        results[i - 1] = empirical_rate(num_items, i, num_defect, size)
    return results.T

#input fixed n and T
#output list of empirical rates as k varies
def vary_defects(num_items, num_tests, size):
    results = np.zeros((num_items, 3))
    for i in range(1, num_items + 1):
        results[i - 1] = empirical_rate(num_items, num_tests, i, size)
    return results.T

def vary_alpha(num_items, num_tests, size):
    results = np.zeros((100, 3))
    for i in range(0, 100):
        results[i] = empirical_rate(num_items, num_tests, math.ceil(math.pow(num_items, i/float(100))), size)
    return results.T

def plot_results_tests(results):
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper left')
    plt.xlabel("number of tests")
    plt.ylabel("success probability")
    plt.show()

def plot_results_defects(results):
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper right')
    plt.xlabel("number of defectives")
    plt.ylabel("success probability")
    plt.show()

def plot_results_alpha(results):
    plt.scatter(np.arange(0, 1, 0.01), results[0], c="b", marker="x", label="COMP")
    plt.scatter(np.arange(0, 1, 0.01), results[1], c="r", marker="x", label="DD")
    plt.scatter(np.arange(0, 1, 0.01), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper right')
    plt.xlabel("number of defectives")
    plt.ylabel("success probability")
    plt.show()

def main():
    test_matrix = np.array([[1, 1, 1, 1, 0, 0, 0, 0], 
                           [0, 0, 0, 1, 1, 1, 0, 1], 
                           [1, 0, 1, 0, 1, 0, 1, 0], 
                           [0, 1, 0, 0, 0, 1, 0, 1],
                           [0, 0, 1, 0, 1, 1, 0, 0]])
    test_outcomes = np.array([[1], [0], [1], [1], [0]])
    test_matrix2 = np.array([[1, 0, 1, 0, 0, 1, 0], 
                            [1, 1, 0, 1, 0, 0, 1], 
                            [1, 0, 0, 0, 1, 0, 0], 
                            [0, 1, 1, 0, 1, 1, 0],
                            [1, 0, 1, 1, 0, 1, 0]])
    test_outcomes2 = np.array([[0], [1], [0], [1], [1]])
    test_matrix3 = np.array([[1, 0, 0, 1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 1, 1, 1, 0],
                            [1, 0, 1, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 0, 0, 1, 0, 1]])
    test_outcomes3 = np.array([[1], [1], [0], [1], [0]])   
    #plot_results_tests(vary_tests(100, 5, 100))
    #plot_results_defects(vary_defects(100, 60, 100))
    #plot_results_alpha(vary_alpha(100, 60, 100))    
    
if __name__ == "__main__":
    main()