import numpy as np
def Barycenter(figure:np.ndarray, P = np.array([]), barycentric = np.array([]), direction = 'c2b'):
    '''
    Convert cartesian coordinates to barycentric & vice versa
    figure = simplex_points (points, dimension)
    P = point to convert (dimension)
    direction: c2b - cartesian to barycentric, b2c - barycentric to cartesian  
    '''
    if direction == 'b2c':
        assert(figure.shape[0] == barycentric.shape[0])
        return figure.T.dot(barycentric)
    else:
        assert(figure.shape[0] == (figure.shape[1] + 1))
        base_vector = figure[-1,:]
        vectors = np.tile(base_vector, [figure.shape[1],1]) - figure[:-1,:]
        target_vector = base_vector - P
        free_memb = vectors.dot(target_vector)
        koeffs = vectors.dot(vectors.T)
        solving_matrix = np.linalg.solve(koeffs, free_memb)
        return np.hstack((solving_matrix, 1 - np.sum(solving_matrix)))
