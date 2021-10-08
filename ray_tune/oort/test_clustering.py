from clustering import k_medoids

def distance(a, b):
    return abs(b - a)

points = [1, 2, 3, 4, 5, 6, 7]
diameter, medoids = k_medoids(points, k=2, distance=distance, spawn=2)
