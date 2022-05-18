import pandas as pd

if __name__ == "__main__":
    # Load the data
    filepath = "/home/amalnanavati/workspaces/amal_noetic_ws/src/feeding_study_cleanup/scripts/detecting_food/data/action_schema_data.csv"
    data = pd.read_csv(filepath)
    print(data["Bag File Name"])

    # Drop the rows aren't skewered items. Using substrings due to naming
    # inconsistencies (e.g., doughnutholes vs. donutholes)
    skewered_item_substrings = [
        "chicken",
        "broc",
        "fri",
        "fry",
        "bagel",
        "nut",
        "lettuce",
        "spinach",
        "sand",
        "pizza",
    ]
    rows_to_drop = []
    for i, row in data.iterrows():
        is_skewered = False
        for substr in skewered_item_substrings:
            if substr in row["Bag File Name"]:
                is_skewered = True
                break
        if not is_skewered:
            rows_to_drop.append(i)
    data  = data.drop(rows_to_drop, axis=0)
    print(data["Bag File Name"])

    # Drop the columns that aren't part of the action schema
    data  = data.drop([
        "Save Timestamp",
        "Bag File Name",
        "Action Start Time",
        "Action Contact Time",
        "Action Extraction Time",
        "Action End Time",
        "Bag Duration",
        "Food Reference Frame Translation X",
        "Food Reference Frame Translation Y",
        "Food Reference Frame Translation Z",
        "Food Reference Frame Rotation X",
        "Food Reference Frame Rotation Y",
        "Food Reference Frame Rotation Z",
    ], axis=1)

    print(data)
