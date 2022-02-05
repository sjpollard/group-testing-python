import numpy as np

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
    dd_indices = pd_indices[np.where(trunc_matrix[test_indices] == 1)[1]]
    return dd_indices + 1

#input test_matrix of shape (T, n) and test_outcomes of shape (T, 1)
#output SCOMP defective set estimate
def scomp_decoder(test_matrix, test_outcomes):
    dd_indices = dd_decoder(test_matrix, test_outcomes) - 1
    scomp_indices = dd_indices
    pd_indices = comp_decoder(test_matrix, test_outcomes) - 1
    pos_tests = np.delete(np.arange(0, test_matrix.shape[0], 1), np.where(test_outcomes == 0)[0])
    unexplained_tests = pos_tests[np.where(np.sum(test_matrix[pos_tests].T[dd_indices].T, axis = 1) == 0)[0]]
    print(unexplained_tests)
    while(len(unexplained_tests) > 0):
        ignored_indices = np.nonzero(np.in1d(pd_indices, scomp_indices))[0]
        most_unexplained = np.argmax(np.sum(test_matrix[unexplained_tests].T[np.delete(pd_indices, ignored_indices)].T, axis = 0))
        scomp_indices = np.append(scomp_indices, np.delete(pd_indices, ignored_indices)[most_unexplained])
        unexplained_tests = np.delete(unexplained_tests, 
        np.where(test_matrix[unexplained_tests].T[pd_indices][most_unexplained] == 1)[0])
    return np.sort(scomp_indices) + 1

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
    
if __name__ == "__main__":
    main()