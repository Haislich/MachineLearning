import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.patches as patches


def load_data(csv_filepath):
    """
    Load data from a csv file.

    :param csv_filepath: path of the .csv file containing the data to load/extract

    :return X: array of values associated with the inputs of the data contained in the .csv file
    :return Y: array of values associated with the labels of the data contained in the .csv file or None if the file does not contain labels
    """
    data = pd.read_csv(csv_filepath)
    # Headers' list:
    headers = data.columns  # 'x' for inputs, 'y' for labels
    # Extract DataFrames based on the headers:
    x_data = data[headers[0]]
    # Convert an array-like string (e.g., '[0.02, 1.34\n, 2.12, 3.23\n]')
    # into an array of floats (e.g., [0.02, 1.34, 2.12, 3.23]):
    X_data = [
        [
            float(feature)
            for feature in feature_vec.replace("[", "").replace("]", "").split()
        ]
        for feature_vec in x_data
    ]
    # convert data into numpy arrays
    X = np.array(X_data)

    if len(headers) > 1:
        Y_data = data[headers[1]]
        Y = np.array(Y_data)
    else:
        Y = None
    return X, Y


X_1, y_1 = load_data("./HW1_datasets_public/dataset1.csv")
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(
    X_1, y_1, test_size=0.33, random_state=42
)
model = make_pipeline(
    StandardScaler(), PCA(n_components=9), SVC(kernel="rbf", probability=True)
)
model.fit(X_1_train, y_1_train)
y_1_pred = model.predict(X_1_test)
idx = np.logical_or(np.array(y_1) == 3, np.array(y_1) == 5)

X_1_35 = model["pca"].fit_transform(model["standardscaler"].fit_transform(X_1[idx]))
y_1_35 = y_1[idx]
l = []
for i in range(len(y_1_35)):
    if np.all(model.predict_proba(X_1[i].reshape(1, -1))[0][[3, 5]] >= 0.2):
        l.append(i)
X_1_tsne_35 = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=30
).fit_transform(X_1_35)
x_min, x_max = X_1_tsne_35[:, 0].min() - 1, X_1_tsne_35[:, 0].max() + 1
y_min, y_max = X_1_tsne_35[:, 1].min() - 1, X_1_tsne_35[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Step 6: Plot the decision boundary along with the t-SNE scatter plot
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_1_tsne_35[:, 0], X_1_tsne_35[:, 1], c=y_1_35, cmap=plt.cm.Paired)
plt.title("SVM Decision Boundary in t-SNE Space")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
