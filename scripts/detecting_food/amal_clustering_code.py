from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ACTION_SCHEMA_COLUMNS = [
    # "Pre-Grasp Target Offset X",
    # "Pre-Grasp Target Offset Y",
    # "Pre-Grasp Target Offset Z",
    "Pre-Grasp Initial Utensil Transform Translation X",
    "Pre-Grasp Initial Utensil Transform Translation Y",
    "Pre-Grasp Initial Utensil Transform Translation Z",
    "Pre-Grasp Initial Utensil Transform Rotation X",
    "Pre-Grasp Initial Utensil Transform Rotation Y",
    "Pre-Grasp Initial Utensil Transform Rotation Z",
    "Pre-Grasp Force Threshold",
    "Grasp In-Food Twist Linear X",
    "Grasp In-Food Twist Linear Y",
    "Grasp In-Food Twist Linear Z",
    "Grasp In-Food Twist Angular X",
    "Grasp In-Food Twist Angular Y",
    "Grasp In-Food Twist Angular Z",
    "Grasp Force Threshold",
    "Grasp Torque Threshold",
    "Grasp Duration",
    "Extraction Out-Of-Food Twist Linear X",
    "Extraction Out-Of-Food Twist Linear Y",
    "Extraction Out-Of-Food Twist Linear Z",
    "Extraction Out-Of-Food Twist Angular X",
    "Extraction Out-Of-Food Twist Angular Y",
    "Extraction Out-Of-Food Twist Angular Z",
    "Extraction Duration",
]

FOOD_NAME_FIX = {
    "bagels"        : "bagel",
    "chicken"       : "chickentenders",
    "doughnuthole"  : "donutholes",
    "doughnutholes" : "donutholes",
    "mashedpotato"  : "mashedpotatoes",
    "mashedpotatos" : "mashedpotatoes"
}

def run_pca(dataframe, visualize=False):
    scaled_data = StandardScaler().fit_transform(dataframe[ACTION_SCHEMA_COLUMNS])
    pca = PCA(n_components=scaled_data.shape[1])
    transformed_data = pca.fit_transform(scaled_data)
    # print(pca.explained_variance_ratio_)
    # print(np.cumsum(pca.explained_variance_ratio_))

    if visualize:
        # 2D, colored by food
        sns.scatterplot(x=transformed_data[:,0], y=transformed_data[:,1], hue=dataframe["Food"])
        plt.title("2D PCA By Food: Explained Variance %f" % sum(pca.explained_variance_ratio_[0:2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        plt.savefig(os.path.join(out_dir, "pca2_by_food.png"))
        plt.clf()

        # 2D, colored by participant
        sns.scatterplot(x=transformed_data[:,0], y=transformed_data[:,1], hue=dataframe["Participant"])
        plt.title("2D PCA By Participant: Explained Variance %f" % sum(pca.explained_variance_ratio_[0:2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        plt.savefig(os.path.join(out_dir, "pca2_by_participant.png"))
        plt.clf()

        # 3D, colored by food
        fig = plt.figure()
        fig.suptitle("3D PCA By Food: Explained Variance %f" % sum(pca.explained_variance_ratio_[0:3]))
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_zlabel("PCA3")
        for food in FOOD_NAMES:
            rows = dataframe["Food"] == food
            ax.plot(transformed_data[rows,0], transformed_data[rows,1], transformed_data[rows,2], marker='o', linestyle='', label=food)
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        plt.savefig(os.path.join(out_dir, "pca3_by_food.png"))
        plt.clf()

        # 3D, colored by food
        fig = plt.figure()
        fig.suptitle("3D PCA By Participant: Explained Variance %f" % sum(pca.explained_variance_ratio_[0:3]))
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_zlabel("PCA3")
        for pid in PIDS:
            rows = dataframe["Participant"] == pid
            ax.plot(transformed_data[rows,0], transformed_data[rows,1], transformed_data[rows,2], marker='o', linestyle='', label=pid)
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        plt.savefig(os.path.join(out_dir, "pca3_by_participant.png"))
        plt.clf()

def get_mean_std(dataframe):
    grouped_by_food = dataframe.groupby(['Food'])[ACTION_SCHEMA_COLUMNS]
    grouped_by_food.mean().to_csv(os.path.join(out_dir, "action_schema_per_food_mean.csv"))
    grouped_by_food.std().to_csv(os.path.join(out_dir, "action_schema_per_food_std.csv"))

def run_k_medoids(dataframe, k=None):
    # First, standardize the data
    scaler = StandardScaler().fit(dataframe[ACTION_SCHEMA_COLUMNS])
    data = scaler.transform(dataframe[ACTION_SCHEMA_COLUMNS])

    # # Without scaling the data
    # data = dataframe[ACTION_SCHEMA_COLUMNS].values

    if k is None:
        k_results = []
        k_vals = range(4, 61, 1)
        for k in k_vals:
            # NOTE: In retrospect, method='pam' would have been a better option,
            # but we already ran the experiments using the default (method='alternate')
            kmedoids = KMedoids(n_clusters=k).fit(data)
            k_results.append([k, kmedoids.inertia_, "K-Medoids"])
            # kmeans = KMeans(n_clusters=k).fit(data)
            # k_results.append([k, kmeans.inertia_, "K-Means"])
        k_results = pd.DataFrame(k_results, columns=["K", "Inertia", "Method"])

        # Option A: Elbow Method
        # kl = KneeLocator(k_results["K"].values, k_results["Inertia"].values, curve="convex", direction="decreasing", interp_method="polynomial", S=1, polynomial_degree=50)
        kl = KneeLocator(k_results["K"].values, k_results["Inertia"].values, curve="convex", direction="decreasing", interp_method="interp1d", S=1)
       # kl.plot_knee()
        # plt.show()
        elbow = kl.elbow
        print("Automatically-Calculated Elbow: ", elbow)

        # # Option B: Elbow Method Defined using Max Curvature
        # intertia_start = k_results[k_results["K"] == 1]["Inertia"].values[0]
        # print(intertia_start)
        # intertia_end = k_results[k_results["K"] == 59]["Inertia"].values[0]
        # print(intertia_end)
        # max_curvature, max_curvature_k = None, None
        # for k in range(2, 59):
        #     intertia_k = k_results[k_results["K"] == k]["Inertia"].values[0]
        #     print(intertia_k)
        #     slope_1 = (intertia_k - intertia_start)/(k - 1)
        #     slope_2 = (intertia_end - intertia_k)/(59 - k)
        #     curvature = (slope_2 - slope_1) / (59 - 1)
        #     if max_curvature is None or curvature > max_curvature:
        #         max_curvature = curvature
        #         max_curvature_k = k
        # elbow = k
        # print("Automatically-Calculated Elbow: ", elbow)

        # # Option C: Silhouette Method
        # sil = []
        # for k in k_vals:
        #     kmedoids = KMedoids(n_clusters=k).fit(data)
        #     labels = kmedoids.labels_
        #     sil.append(silhouette_score(data, labels, metric = 'euclidean'))
        # print("silhouette scores: ", sil)
        # elbow = k_vals[np.argmax(sil)]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=k_results, x="K", y="Inertia", hue="Method", ax=ax)
        ax.scatter([elbow], k_results[k_results["K"] == elbow]["Inertia"].values)
        ax.set_xticks(k_vals, minor=True)
        ax.grid(which='both')
        plt.savefig(os.path.join(out_dir, "kmedoids_by_k.png"))
        plt.clf()
    else:
        kmedoids = KMedoids(n_clusters=k, method='alternate').fit(data)
        centers = pd.DataFrame(np.c_[np.arange(k),scaler.inverse_transform(kmedoids.cluster_centers_)], columns=["Cluster ID"] + ACTION_SCHEMA_COLUMNS)
        centers.to_csv(os.path.join(out_dir, "kmedoids_centers_k_%d.csv" % k))

def run_k_medoids_and_get_confusion_matrix(dataframe, k1, k2, cosine=False):
    """
    Runs k-medoids on the dataframr, separately for k1 and k2. Then outputs a
    confusion matrix of the distance between each medoid in k1 and each medoid
    in k2.
    """
    # First, standardize the data
    scaler = StandardScaler().fit(dataframe[ACTION_SCHEMA_COLUMNS])
    data = scaler.transform(dataframe[ACTION_SCHEMA_COLUMNS])

    # Run k-medoids
    kmedoids1 = KMedoids(n_clusters=k1).fit(data)
    kmedoids2 = KMedoids(n_clusters=k2).fit(data)

    # Get num datapoints in each cluster
    print("k", k2, "cluster sizes", np.unique(kmedoids2.labels_, return_counts=True)[1])

    # Get the num unique participants in each k2 cluster
    _, cluster_assignment = np.unique(kmedoids2.labels_, return_inverse=True)
    cluster_assignment_key = "k %d cluster assignment" % k2
    dataframe[cluster_assignment_key] = cluster_assignment
    num_unique_participants = []
    for i in range(k2):
        num_unique_participants.append(len(dataframe[dataframe[cluster_assignment_key] == i]["Participant"].unique()))
    print("k", k2, "num_unique_participants", num_unique_participants)

    # Get cluster centers, and unscale them
    centers1 = scaler.inverse_transform(kmedoids1.cluster_centers_)
    centers2 = scaler.inverse_transform(kmedoids2.cluster_centers_)

    # Get confusion matrix
    confusion_matrix = np.zeros((k1, k2))
    for i in range(k1):
        for j in range(k2):
            if cosine: # Cosine Distance
                confusion_matrix[i,j] = 1 - np.dot(centers1[i], centers2[j])/(np.linalg.norm(centers1[i])*np.linalg.norm(centers2[j]))
            else: # Euclidean Distance
                confusion_matrix[i,j] = np.linalg.norm(centers1[i] - centers2[j])

    # pprint.pprint(confusion_matrix)

    # Graph the confusion matrix as a heatmap
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(confusion_matrix, annot=True, ax=ax, fmt=".3f")
    ax.set_xlabel("kmedoids2")
    ax.set_ylabel("kmedoids1")
    plt.savefig(os.path.join(out_dir, "kmedoids_confusion_matrix_k1_%d_k2_%d.png" % (k1, k2)))
    plt.clf()

if __name__ == "__main__":
    # Load the data
    base_dir = "/Users/amalnanavati/Documents/PRL/feeding_study_cleanup/data"
    out_dir = "/Users/amalnanavati/Documents/PRL/feeding_study_cleanup/data/output"
    file_name = "action_schema_data.csv"
    dataframe = pd.read_csv(os.path.join(base_dir, file_name))

    # Make food names unique
    dataframe['Food'] = dataframe['Food'].apply(lambda food: FOOD_NAME_FIX[food] if food in FOOD_NAME_FIX else food)
    FOOD_NAMES = sorted(dataframe["Food"].unique())
    # Make Participant a String
    dataframe['Participant'] = dataframe['Participant'].apply(lambda pid: str(pid))
    PIDS = sorted(dataframe["Participant"].unique())

    # Drop nan and inf
    print("DROPPING THE FOLLOWING ROWS")
    print(dataframe[dataframe.isin([np.nan, np.inf, -np.inf]).any(axis=1)]["Bag File Name"])
    dataframe = dataframe[~dataframe.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    # # Run PCA
    # run_pca(dataframe, visualize=True)

    # Get the mean and std per food item
    get_mean_std(dataframe)

    # # Run k-medoids on the full action schema
    run_k_medoids(dataframe)
    # run_k_medoids(dataframe, k=11)

    # # Run k-medoids on two k's and get a confusion matrix
    # # ks = [3, 7, 11, 15]
    # ks = range(1,16)
    # for i in range(len(ks)-1):
    #     k1 = ks[i]
    #     k2 = ks[i+1]
    #     run_k_medoids_and_get_confusion_matrix(dataframe, k1, k2, cosine=False)
