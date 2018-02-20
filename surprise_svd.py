
from __future__ import unicode_literals # some movies have foreign chars
# install surprise with: pip install scikit-surprise
from surprise import SVD
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from scipy.linalg import svd

from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
# Load the movielens-100k dataset (download it if needed),
reader = Reader(line_format='user item rating', sep='\t')
trainset = Dataset.load_from_file('train.txt', reader).build_full_trainset()
raw_testset = Dataset.load_from_file('test.txt', reader).raw_ratings

# data does not have time stamps , so remove their placeholder None entry
testset = []
for i in range(len(raw_testset)):
    testset.append(raw_testset[i][0:3])

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

print ('U matrix shape', algo.pu.shape)
print ('V matrix shape', algo.qi.shape)

Uu, su, Vhu = svd(algo.pu, full_matrices=False)
Ui, si, Vhi = svd(algo.qi, full_matrices=False)
print (Ui.shape, si.shape, Vhi.shape)
V_reduced = np.dot(algo.qi,Ui[1:3,:].T)
print (V_reduced)

# visualizations
def plot_all(selection, title ="Visualization"):
    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(
        V_reduced[selection, 0], V_reduced[selection, 1], marker='o')

    for label, x, y in zip(movieData[selection,1], V_reduced[selection, 0], V_reduced[selection, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->'))
    plt.title(title)
    plt.show()

movieData = np.loadtxt('movies.txt', delimiter='\t', dtype='str')

# 10 movies of choice
selection = [0, 9, 27, 49, 88,94,98,150,164,165]

plot_all(selection, '10 chosen movies')

# 10 most popular movies
# Extract most popular movies and relevant rating data
ratingData = np.loadtxt('data.txt', dtype='int')
[ratings, counts] = np.unique(ratingData[:,2], return_counts=True)
[movieIDs, counts] = np.unique(ratingData[:,1], return_counts=True)
popularCountInd = np.argsort(counts)[-10:]
selection = movieIDs[popularCountInd] - 1 # get the indexes only
plot_all(selection, '10 most popular movies')

# 10 most higly rated:
movieIDs = np.unique(ratingData[:,1])
ratingArray = [[] for j in movieIDs]
for row in ratingData:
    ratingArray[row[1]-1].append(row[2])
averageRatings = [np.average(row) for row in ratingArray]
highlyRatedInd = np.argsort(averageRatings)[-10:]
selection = movieIDs[highlyRatedInd] - 1 # get the indexes only
plot_all(selection, '10 most highly rated movies')

# genres
def genre_plot(genreIndex, genreName):
    # Find movie IDs by genre and collect ratings
    genreMovieInd = (np.int64(movieData[:,genreIndex]) == 1)
    selection = np.random.choice(np.int64(movieData[genreMovieInd, 0]), size=10) - 1
    plot_all(selection, "10 " + genreName + " Movies Visualized")

genre_plot(3, "Action")
genre_plot(7, "Comedy")
genre_plot(14, "Musical")
