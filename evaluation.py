from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd


def run_evaluations(explanation_values, y, k_values_max=9):
    scaled_values = StandardScaler().fit_transform(explanation_values)
    pairwise_distances = squareform(pdist(scaled_values, "euclidean"))
    cluster_evaluations = []

    for k in range(2, k_values_max):
        kmeans = KMeans(n_clusters=k).fit(pairwise_distances)
        spectral_clusters = SpectralClustering(
            n_clusters=k, affinity="precomputed"
        ).fit(pairwise_distances)

        silhouette_kmeans = silhouette_score(pairwise_distances, kmeans.labels_)
        silhouette_spectral = silhouette_score(
            pairwise_distances, spectral_clusters.labels_
        )
        db_index_kmeans = davies_bouldin_score(pairwise_distances, kmeans.labels_)
        db_index_spectral = davies_bouldin_score(
            pairwise_distances, spectral_clusters.labels_
        )
        cluster_evaluations.append(
            [
                k,
                silhouette_kmeans,
                silhouette_spectral,
                db_index_kmeans,
                db_index_spectral,
            ]
        )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_features": [12, "sqrt", "log2"],
        "max_depth": [4, 6, 8, 10],
        "criterion": ["gini", "entropy"],
    }
    X_train, X_test, y_train, y_test = train_test_split(
        explanation_values, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    print("Best Hyperparameters:", best_params)

    best_rf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_features=best_params["max_features"],
        max_depth=best_params["max_depth"],
        criterion=best_params["criterion"],
        random_state=42,
    )

    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Compute the AUC
    roc_auc = auc(fpr, tpr)

    evaluation_df = pd.DataFrame(
        cluster_evaluations,
        columns=[
            "k",
            "Silhouette Kmeans",
            "Silhouette Spectral",
            "DBI Kmeans",
            "DBI Spectral",
        ],
    )

    return evaluation_df, fpr, tpr, roc_auc
