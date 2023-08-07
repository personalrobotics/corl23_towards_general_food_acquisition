import os
import pandas as pd

if __name__ == "__main__":
    base_dir = "/Volumes/HCRLAB/2021_11_Bite_Acquisition_Study/processed"

    total_time = 0.0
    for subfolder_name in os.listdir(base_dir):

        subfolder_path = os.path.join(base_dir, subfolder_name)

        if os.path.isfile(subfolder_path):
            continue

        for subsubfolder_name in os.listdir(subfolder_path):
            if "wrenches_poses" not in subsubfolder_name or subsubfolder_name[0] == ".":
                continue

            subsubfolder_path = os.path.join(subfolder_path, subsubfolder_name)
            print(subsubfolder_path)
            df = pd.read_csv(subsubfolder_path)
            # print(df)

            duration = df["Time (sec)"].max() - df["Time (sec)"].min()
            total_time += duration

    print("Total Time", total_time, total_time/60.0)
