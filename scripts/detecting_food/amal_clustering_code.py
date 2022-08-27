from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, AgglomerativeClustering

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
    "bagels" : "bagel",
    "chicken" : "chickentenders",
    "doughnuthole" : "donutholes",
    "doughnutholes" : "donutholes",
    "mashedpotato" : "mashedpotatoes",
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

    if k is None:
        k_results = []
        k_vals = range(1, 60)
        for k in k_vals:
            kmedoids = KMedoids(n_clusters=k).fit(data)
            k_results.append([k, kmedoids.inertia_, "K-Medoids"])
            # kmeans = KMeans(n_clusters=k).fit(data)
            # k_results.append([k, kmeans.inertia_, "K-Means"])
        k_results = pd.DataFrame(k_results, columns=["K", "Inertia", "Method"])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=k_results, x="K", y="Inertia", hue="Method", ax=ax)
        ax.set_xticks(k_vals, minor=True)
        ax.grid(which='both')
        plt.savefig(os.path.join(out_dir, "kmedoids_by_k.png"))
        plt.clf()

        kl = KneeLocator(k_results["K"].values, k_results["Inertia"].values, curve="convex", direction="decreasing", interp_method="polynomial")
        print("Automatically-Calculated Elbow: ", kl.elbow)
    else:
        kmedoids = KMedoids(n_clusters=k).fit(data)
        centers = pd.DataFrame(np.c_[np.arange(k),scaler.inverse_transform(kmedoids.cluster_centers_)], columns=["Cluster ID"] + ACTION_SCHEMA_COLUMNS)
        centers.to_csv(os.path.join(out_dir, "kmedoids_centers_k_%d.csv" % k))



if __name__ == "__main__":
    # Load the data
    base_dir = "/Users/amaln/Documents/PRL/feeding_study_cleanup/data"
    out_dir = "/Users/amaln/Documents/PRL/feeding_study_cleanup/data/output"
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
    print(dataframe[dataframe.isin([np.nan, np.inf, -np.inf]).any(1)]["Bag File Name"])
    dataframe = dataframe[~dataframe.isin([np.nan, np.inf, -np.inf]).any(1)]

    # Run PCA
    run_pca(dataframe, visualize=True)

    # Get the mean and std per food item
    get_mean_std(dataframe)

    # # Run k-medoids on the full action schema
    # run_k_medoids(dataframe)
    run_k_medoids(dataframe, k=6)
