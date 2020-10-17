import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np

colors = 10 * ["g", "r", "c", "b", "k"]


class K_Means:
    def __init__(self, kmeans=2, tolerance=0.001, maxIteration=300):
        self.kmeans = kmeans
        self.tolerance = tolerance
        self.maxIteration = maxIteration

    def fit(self, data):

        self.centroids = {}

        for i in range(self.kmeans):
            self.centroids[i] = data[i]

        for i in range(self.maxIteration):
            self.classifications = {}

            for i in range(self.kmeans):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def update(self, new_data, delta):
        for featureset in new_data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]

            if min(distances) < delta:
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            else:
                self.centroids[self.kmeans] = featureset
                self.classifications[self.kmeans] = []
                self.classifications[self.kmeans].append(featureset)
                self.kmeans = self.kmeans + 1

X = np.array([[1, 5],
              [2.7, 3.8],
              [5, 2],
              [8, 7],
              [9,4],
              [9,10],
              [3,6],
              [4,4],
              [1, 9],
              [8, 11]])

clf = K_Means()
clf.fit(X)

X1 = np.array([[9, 8],
               [8, 6],
               [6, 3],
               [6, 4],
               [3, 4]])

#Updating the model with X1 and threshold of 4 
clf.update(X1, 4)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()