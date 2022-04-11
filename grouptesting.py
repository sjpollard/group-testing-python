import math
import scipy.stats as ss
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
rng1 = np.random.default_rng(2022)
rng2 = np.random.default_rng(0)

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
def test_generator(num_items, num_tests, num_defect, rng):
    prob = 1.0 / num_defect if num_defect != 1 else 0.5
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
        sample = test_generator(num_items, num_tests, num_defect, rng1)
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
    results = np.zeros((15, 3))
    for i in range(1, 14 + 1):
        results[i - 1] = empirical_rate(num_items, num_tests, i, size)
    return results.T

def vary_items(num_tests, num_defect, size):
    results = np.zeros((96, 3))
    for i in range(5, 101):
        results[i - 5] = empirical_rate(i, num_tests, num_defect, size)
    return results.T

def vary_alpha(num_items, num_tests, size):
    results = np.zeros((100, 3))
    for i in range(0, 100):
        results[i] = empirical_rate(num_items, num_tests, math.ceil(math.pow(num_items, i/float(100))), size)
    return results.T

def determine_num_tests(num_items, max_defects, size, error):
    threshold = 1 - error
    results = np.zeros((max_defects, 3))
    for i in range(1, max_defects + 1):
        result = vary_tests(num_items, i, size)
        results[i - 1][0] = np.argmax(result[0] > threshold)
        results[i - 1][1] = np.argmax(result[1] > threshold)
        results[i - 1][2] = np.argmax(result[2] > threshold)
    return results.T + 1

def closest_to(array, value):
    absolute_val =  np.abs(array - value)
    return np.argmin(absolute_val)

def train_svc(num_items, num_tests, num_defect, size):
    x_train = np.empty((size, (num_items + 1) * num_tests))
    y_train = np.empty(size)
    for i in range(size):
        gen = test_generator(num_items, num_tests, num_defect, rng2)
        x_train[i] = np.hstack((gen[0], gen[1])).flatten()
        y = 0
        for j in gen[2]:
            y += 2 ** (j-1)
        y_train[i] = y
    classifier = svm.SVC()
    classifier.fit(x_train, y_train)
    return classifier

def svc_decoder(classifier, test_matrix, test_outcomes):
    defects = np.array([], dtype=int)
    x = [np.hstack((test_matrix, test_outcomes)).flatten()]
    y = classifier.predict(x)
    binary = bin(int(y[0])).replace("0b", "")[::-1]
    for i in range(len(binary)):
        if (binary[i] == "1"):
            defects = np.append(defects, i + 1)
    return defects

def empirical_rate_svc(classifier, num_items, num_tests, num_defect, size, rng):
    correct_tests = np.array([[0, 0, 0, 0]])
    for i in range(size):
        sample = test_generator(num_items, num_tests, num_defect, rng)
        correct_tests[0][0] += np.array_equal(comp_decoder(sample[0], sample[1]), sample[2])
        correct_tests[0][1] += np.array_equal(dd_decoder(sample[0], sample[1]), sample[2])
        correct_tests[0][2] += np.array_equal(scomp_decoder(sample[0], sample[1]), sample[2])
        correct_tests[0][3] += np.array_equal(svc_decoder(classifier, sample[0], sample[1]), sample[2])
    return(correct_tests/float(size))

def vary_tests_svc(num_items, num_defect, size, rng):
    results = np.zeros((num_items, 4))
    for i in range(1, num_items + 1):
        classifier = train_svc(num_items, i, num_defect, 100000)
        results[i - 1] = empirical_rate_svc(classifier, num_items, i, num_defect, size, rng)
    return results.T

def empirical_rate_count(num_items, num_tests, num_defect, size, rng):
    correct_tests = np.array([[0, 0, 0]])
    for i in range(size):
        sample = test_generator(num_items, num_tests, num_defect, rng)
        correct_tests[0][0] += comp_decoder(sample[0], sample[1]).size == sample[2].size
        correct_tests[0][1] += dd_decoder(sample[0], sample[1]).size == sample[2].size
        correct_tests[0][2] += scomp_decoder(sample[0], sample[1]).size == sample[2].size
    return(correct_tests/float(size))

def vary_tests_count(num_items, num_defect, size, rng):
    results = np.zeros((num_items, 3))
    for i in range(1, num_items + 1):
        results[i - 1] = empirical_rate_count(num_items, i, num_defect, size, rng)
    return results.T

def vary_defects_count(num_items, num_tests, size, rng):
    results = np.zeros((15, 3))
    for i in range(1, 14 + 1):
        results[i - 1] = empirical_rate_count(num_items, num_tests, i, size, rng)
    return results.T


def plot_results_tests_svc(results):
    plt.title("Success probability vs Tests")
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.scatter(range(1, results.shape[1] + 1), results[3], c="m", marker="x", label="GTSVM")
    plt.legend(loc='upper left')
    plt.xlabel("Number of tests")
    plt.ylabel("Success probability")
    plt.show()

def plot_results_tests_count(results, results2):
    plt.title("Success probability vs Tests")
    plt.scatter(range(1, results2.shape[1] + 1), results2[0], c="grey", marker="x")
    plt.scatter(range(1, results2.shape[1] + 1), results2[1], c="grey", marker="x")
    plt.scatter(range(1, results2.shape[1] + 1), results2[2], c="grey", marker="x")
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper left')
    plt.xlabel("Number of tests")
    plt.ylabel("Success probability")
    plt.show()

def plot_results_defects_count(results, results2):
    plt.title("Success probability vs Defectives")
    plt.scatter(range(1, results2.shape[1] + 1), results2[0], c="grey", marker="x")
    plt.scatter(range(1, results2.shape[1] + 1), results2[1], c="grey", marker="x")
    plt.scatter(range(1, results2.shape[1] + 1), results2[2], c="grey", marker="x")
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper right')
    plt.xlabel("Number of defectives")
    plt.ylabel("Success probability")
    plt.show()

def plot_results_tests(results):
    plt.title("Success probability vs Tests")
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper left')
    plt.xlabel("Number of tests")
    plt.ylabel("Success probability")
    plt.show()

def plot_results_tests_gaussian(results):
    plt.title("Success probability vs Tests with Gaussian")
    g_x = np.linspace(1, 100, 5000)
    g_cdf = ss.norm.cdf(g_x, closest_to(results[2], 0.5) + 1, 10)
    plt.plot(g_x, g_cdf, c="black", label="Gaussian CDF")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper left')
    plt.xlabel("Number of tests")
    plt.ylabel("Success probability")
    plt.show()

def plot_results_defects(results):
    plt.title("Success probability vs Defectives")
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper right')
    plt.xlabel("Number of defectives")
    plt.ylabel("Success probability")
    plt.show()

def plot_results_items(results):
    plt.title("Success probability vs Items")
    plt.scatter(range(5, 100 + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(5, 100 + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(5, 100 + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='lower left')
    plt.xlim([0, 101])
    plt.xlabel("Number of items")
    plt.ylabel("Success probability")
    plt.show()

def plot_results_alpha(results):
    plt.scatter(np.arange(0, 1, 0.01), results[0], c="b", marker="x", label="COMP")
    plt.scatter(np.arange(0, 1, 0.01), results[1], c="r", marker="x", label="DD")
    plt.scatter(np.arange(0, 1, 0.01), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper right')
    plt.xlabel("number of defectives")
    plt.ylabel("success probability")
    plt.show()

def plot_determine_tests(results):
    plt.scatter(range(1, results.shape[1] + 1), results[0], c="b", marker="x", label="COMP")
    plt.scatter(range(1, results.shape[1] + 1), results[1], c="r", marker="x", label="DD")
    plt.scatter(range(1, results.shape[1] + 1), results[2], c="g", marker="x", label="SCOMP")
    plt.legend(loc='upper left')
    plt.xlabel("number of defectives")
    plt.ylabel("number of tests required")
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
    #plot_results_items(vary_items(60, 5, 1000))
    #plot_results_tests_gaussian(vary_tests(100, 5, 1000))
    #plot_results_tests(vary_tests(100, 5, 1000))
    #plot_results_defects(vary_defects(100, 60, 1000))
    #plot_results_alpha(vary_alpha(100, 60, 100))    
    #plot_determine_tests(determine_num_tests(100, 10, 100, 0.05)
    #classifier = train_svc(10, 10, 1, 10000)
    #print(empirical_rate_svc(classifier, 10, 10, 1, 1000, np.random.default_rng()))
    #plot_results_tests_svc(vary_tests_svc(10, 2, 1000, np.random.default_rng()))
    #plot_results_tests_count(vary_tests_count(100, 5, 1000, rng1), vary_tests(100, 5, 1000))
    plot_results_defects_count(vary_defects_count(100, 60, 1000, rng1), vary_defects(100, 60, 1000))

    
if __name__ == "__main__":
    main()