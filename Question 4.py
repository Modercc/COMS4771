import scipy.io
import numpy as np
import collections
from queue import PriorityQueue
import operator
from matplotlib import pyplot as plt
from pprint import pformat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class MLE:
               
    def train(self, x, y: np.ndarray):
        n = x.shape[0]
        self.dim = x.shape[1]
        self.mean_list = []
        self.sigma_inv_list = []
        self.scalars = []
        for i in range(10):
            class_i = np.array([x[j] for j in range(n) if y[j] == i])
            self.mean_list.append(np.mean(class_i, axis = 0))
            
            sigma = np.cov(class_i, rowvar = False)
            if np.sum(np.linalg.eigvals(sigma) <= 0) != 0:
                print('Covarieance matrix is not positive definite!\n')
                sigma_inv = np.linalg.pinv(sigma, rcond = 1e-15, hermitian = True)
            else:
                sigma_inv = np.linalg.inv(sigma)
            self.sigma_inv_list.append(sigma_inv)
            
            scalar = 1 / np.sqrt(((2*np.pi)**self.dim)*np.linalg.det(sigma))
            self.scalars.append(scalar)
    
    def likelihood_by_class(self, x: np.ndarray, y: int):
        mean = self.mean_list[y]
        sigma_inv = self.sigma_inv_list[y]
        scalar = self.scalars[y]
        exp = (-0.5)*np.dot(np.matmul(x - mean, sigma_inv), x - mean)
        return scalar * (np.e**exp)
    
    def classify(self, x: np.ndarray):
        likelihoods = [self.likelihood_by_class(x, i) for i in range(10)]
        return np.argmax(likelihoods)
    
    def performance(self, x: np.ndarray, y: np.ndarray):
        n = x.shape[0]
        predicted_y = np.array([self.classify(x[i]) for i in range(n)])
        n_correct = 0
        for i in range(n):
            if predicted_y[i] == y[i]:
                n_correct += 1
        return n_correct / n    
    
class KDNode:
    
    def __init__(self, value, split, left, right):
        self.value = value
        self.split = split
        self.left = left
        self.right = right
        
class KDTree:
    
    def __init__(self, x):
        self.dim = len(x[0])
        self.tree = self.generate_kdTree(x)
    
    def generate_kdTree(self, x):
        if not x:
            return
        else:
           
            split, node = self.makeSplit(x)
            left = [point for point in x if point[split] > node[split]]
            right = [point for point in x if point[split] < node[split]]
            return KDNode(node, split, self.generate_kdTree(left), self.generate_kdTree(right))
    
    def makeSplit(self, x):
        std_list = []
        for i in range(self.dim):
            std_i = np.std([point[i] for point in x])
            std_list.append(std_i)
        split = std_list.index(max(std_list))
        x.sort(key=lambda y: y[split])
        index = int((len(x) + 2) / 2) - 1
        node = x[index]
        return split, node

class NodeDist:
    
    def __init__(self, node, dist):
        self.node = node
        self.dist = dist
        
    def __lt__(self, other):
        return self.dist > other.dist
    
def search_k_neighbor(kdtree, target, k, dist_type):
    k_queue = PriorityQueue()
    return search_path(kdtree, target, k, k_queue, dist_type)

def search_path(kdtree, target, k, k_queue, dist_type):
    if kdtree is None:
        return NodeDist([], np.inf)
    path = []
    while kdtree:
        if target[kdtree.split] <= kdtree.value[kdtree.split]:
            path.append((kdtree.value, kdtree.split, kdtree.right))
            kdtree = kdtree.left
        else:
            path.append((kdtree.value, kdtree.split, kdtree.left))
            kdtree = kdtree.right
    path.reverse()
    radius = np.inf
    for i in path:
        node = i[0]
        split = i[1]
        opposite_tree = i[2]
        distance_axis = abs(node[split] - target[split])
        if distance_axis > radius + 0.2 and False:
            print(distance_axis, radius)
            break
        else:
            if dist_type == 1:
                dist = Manhattan_dist(node, target)
            else:
                if dist_type == 2:
                    dist = Euclidian_dist(node, target)
                else:
                    dist = Chebyshev_dist(node, target)
            k_queue.put(NodeDist(node, dist))
            if k_queue.qsize() > k:
                k_queue.get()
                radius = k_queue.queue[0].dist
            #print([i.dist for i in k_queue.queue])
            search_path(opposite_tree, target, k, k_queue, dist_type)
            radius = k_queue.queue[0].dist
    return k_queue
    
def Euclidian_dist(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def Manhattan_dist(point1, point2):
    return np.sum(abs(np.array(point1) - np.array(point2)))

def Chebyshev_dist(point1, point2):
    return np.amax(abs(np.array(point1) - np.array(point2)))

class KKNclassifier:
    
    def __init__(self, k):
        self.k = k
    
    def findLabel(self, new_point, train_data, labels, dist_type):
        result = search_k_neighbor(self.kdtree.tree, new_point, self.k, dist_type)
        k_labels = [self.data_dict[tuple(data.node)] for data in result.queue]
        vote = [0 for i in range(10)]
        #print([i.dist for i in result.queue])
        #print(k_labels)
        for label in k_labels:
            vote[label[0]] += 1
        return vote.index(max(vote))
    
    def predict(self, x_train, y_train, x_test, y_test, dist_type):
        n = len(x_test)
        self.data_dict = {tuple(data):label for data,label in zip(x_train, y_train)}
        self.kdtree = KDTree(x_train)
        y_predictions = [self.findLabel(point, x_train, y_train, dist_type) for point in x_test]
        corr_count = 0
        for i in range(n):
            #print(y_predictions[i], " ", y_test[i])
            if y_predictions[i] == y_test[i][0]:
                corr_count += 1
        return corr_count / n
        
    
mat = scipy.io.loadmat('digits.mat')
x = mat['X']
y = mat['Y']
x = StandardScaler().fit_transform(x)
pca = PCA(0.43)
x_pca = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)
print(x_train.shape)
mle = MLE()
mle.train(x_train, y_train)
perf = mle.performance(x_test, y_test)
print("Performance ", perf)
"""
knn = KNearestNeighbors(x_train[:100], y_train[:100], n_neighbors=1)
print("Test size: ", x_test.shape[0])
print(knn.score(x_test, y_test))
"""

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train[:100], y_train[:100])
y_pred = classifier.predict(x_test[:100])
perf = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        perf += 1
perf /= len(y_pred)
print("SKLearn: ", perf)
distances, indices = classifier.kneighbors(x_test[7:8])
print(distances)
knn = KKNclassifier(5)
print(knn.predict(x_train.tolist()[:1000], y_train.tolist()[:1000], x_test.tolist()[0:100], y_test.tolist()[0:100], 2))



