# Standard Imports
import argparse
import os
import pprint

# Third-Party Imports
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ACTION_SCHEMA_COLUMNS = [
    # "Pre_Grasp_Target_Offset_X",
    # "Pre_Grasp_Target_Offset_Y",
    # "Pre_Grasp_Target_Offset_Z",
    "Pre_Grasp_Initial_Utensil_Transform_Translation_X",
    "Pre_Grasp_Initial_Utensil_Transform_Translation_Y",
    "Pre_Grasp_Initial_Utensil_Transform_Translation_Z",
    "Pre_Grasp_Initial_Utensil_Transform_Rotation_X",
    "Pre_Grasp_Initial_Utensil_Transform_Rotation_Y",
    "Pre_Grasp_Initial_Utensil_Transform_Rotation_Z",
    "Pre_Grasp_Force_Threshold",
    "Grasp_In_Food_Twist_Linear_X",
    "Grasp_In_Food_Twist_Linear_Y",
    "Grasp_In_Food_Twist_Linear_Z",
    "Grasp_In_Food_Twist_Angular_X",
    "Grasp_In_Food_Twist_Angular_Y",
    "Grasp_In_Food_Twist_Angular_Z",
    "Grasp_Force_Threshold",
    "Grasp_Torque_Threshold",
    "Grasp_Duration",
    "Extraction_Out_Of_Food_Twist_Linear_X",
    "Extraction_Out_Of_Food_Twist_Linear_Y",
    "Extraction_Out_Of_Food_Twist_Linear_Z",
    "Extraction_Out_Of_Food_Twist_Angular_X",
    "Extraction_Out_Of_Food_Twist_Angular_Y",
    "Extraction_Out_Of_Food_Twist_Angular_Z",
    "Extraction_Duration",
]

FOOD_NAME_FIX = {
    "bagels"        : "bagel",
    "chicken"       : "chickentenders",
    "doughnuthole"  : "donutholes",
    "doughnutholes" : "donutholes",
    "mashedpotato"  : "mashedpotatoes",
    "mashedpotatos" : "mashedpotatoes"
}

def run_k_medoids(dataframe, out_dir, k=None):
    # First, standardize the data
    scaler = StandardScaler().fit(dataframe[ACTION_SCHEMA_COLUMNS])
    data = scaler.transform(dataframe[ACTION_SCHEMA_COLUMNS])

    if k is None:
        k_results = []
        k_vals = range(4, 61, 1)
        for k in k_vals:
            # NOTE: In retrospect, method='pam' would have been a better option,
            # than the default (method='alternate')
            kmedoids = KMedoids(n_clusters=k).fit(data)
            k_results.append([k, kmedoids.inertia_, "K-Medoids"])
        k_results = pd.DataFrame(k_results, columns=["K", "Inertia", "Method"])

        # Option A: Elbow Method
        kl = KneeLocator(k_results["K"].values, k_results["Inertia"].values, curve="convex", direction="decreasing", interp_method="interp1d", S=1)
        elbow = kl.elbow
        print("Automatically-Calculated Elbow: ", elbow)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=k_results, x="K", y="Inertia", hue="Method", ax=ax)
        ax.scatter([elbow], k_results[k_results["K"] == elbow]["Inertia"].values)
        ax.set_xticks(k_vals, minor=True)
        ax.grid(which='both')
        plt.savefig(os.path.join(out_dir, "kmedoids_by_k.png"))
        plt.clf()
    else:
        kmedoids = KMedoids(n_clusters=k, method='alternate').fit(data)

        print("Medoid Indices", kmedoids.medoid_indices_)

        centers = pd.DataFrame(np.c_[np.arange(k),scaler.inverse_transform(kmedoids.cluster_centers_)], columns=["Cluster ID"] + ACTION_SCHEMA_COLUMNS)
        centers.to_csv(os.path.join(out_dir, "kmedoids_centers_k_%d.csv" % k))

if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the file `action_schema_data.csv`, relative to the repository root")
    parser.add_argument("--k", help="How many clusters to use. If not set, it does a sweep to find the elbow point.")
    args = parser.parse_args()

    # Get the default data path
    if args.data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data/action_schema_data.csv")
    else:
        data_path = os.path.join(os.path.dirname(__file__), "..", args.data_path)

    # Read the data
    dataframe = pd.read_csv(data_path)

    # Make food names unique
    dataframe['Food'] = dataframe['Food'].apply(lambda food: FOOD_NAME_FIX[food] if food in FOOD_NAME_FIX else food)
    FOOD_NAMES = sorted(dataframe["Food"].unique())

    # Make Participant a String
    dataframe['Participant'] = dataframe['Participant'].apply(lambda pid: str(pid))
    PIDS = sorted(dataframe["Participant"].unique())

    # Drop nan and inf
    print("DROPPING THE FOLLOWING ROWS")
    print(dataframe[dataframe.isin([np.nan, np.inf, -np.inf]).any(axis=1)][["Participant","Food","Trial"]])
    dataframe = dataframe[~dataframe.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    if args.k is None:
        run_k_medoids(dataframe, os.path.dirname(data_path))
    else:
        try:
            k = int(args.k)
        except ValueError:
            raise Exception("k must be an integer")
        run_k_medoids(dataframe, os.path.dirname(data_path), k=k)