import os
import shutil
import tarfile
import traceback

if __name__ == "__main__":
    IN_DIR = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study/processed/"
    OUT_DIR = "/media/amalnanavati/HCRLAB/2022_11_Bite_Acquisition_Study/compressed/"

    processed_files = -1
    total_files = len(os.listdir(IN_DIR))
    for base_filename in os.listdir(IN_DIR):
        processed_files += 1
        print(processed_files, total_files, base_filename)
        try:
            filepath = os.path.join(IN_DIR, base_filename)
            # For all files in the main directory, directly copy them over
            if os.path.isfile(filepath):
                shutil.copyfile(filepath, os.path.join(OUT_DIR, base_filename))
            # For all dirs, zip each type of data (rgb images, depth images, CSVs) separately
            elif os.path.isdir(filepath):
                # Open the tarfiles
                rgb_files = tarfile.open(os.path.join(OUT_DIR, "%s_rgb_images.tar.gz" % (base_filename,)), "w|gz")
                depth_files = tarfile.open(os.path.join(OUT_DIR, "%s_depth_images.tar.gz" % (base_filename,)), "w|gz")
                csv_files = tarfile.open(os.path.join(OUT_DIR, "%s_wrenches_poses_transforms.tar.gz" % (base_filename,)), "w|gz")

                # Get all the trial numbers for this participant and food type
                trial_nums = set()
                for filename in os.listdir(filepath):
                    if os.path.isdir(os.path.join(filepath, filename)):
                        trial_nums.add(int(filename))

                for trial_num in trial_nums:
                    # Add the images
                    for filename in os.listdir(os.path.join(filepath, str(trial_num))):
                        if "rgb" in filename:
                            rgb_files.add(os.path.join(filepath, "%d/%s" % (trial_num, filename)), arcname="%s/%d/%s" % (base_filename, trial_num, filename))
                        elif "depth" in filename:
                            depth_files.add(os.path.join(filepath, "%d/%s" % (trial_num, filename)), arcname="%s/%d/%s" % (base_filename, trial_num, filename))
                        else:
                            print("WARNING: Unaccounted for filename %s/%d/%s" % (filepath, trial_num, filename))
                    # Add the CSV files
                    csv_files.add(os.path.join(filepath, "%d_static_transforms.csv" % (trial_num,)), arcname="%s/%d_static_transforms.csv" % (base_filename, trial_num))
                    csv_files.add(os.path.join(filepath, "%d_wrenches_poses.csv" % (trial_num,)), arcname="%s/%d_wrenches_poses.csv" % (base_filename, trial_num))

                # Close the tarfiles
                rgb_files.close()
                depth_files.close()
                csv_files.close()
            else:
                print("WARNING: %s is neither a file nor a directory" % os.path.join(IN_DIR, filename))
        except Exception as e:
            traceback.print_exc()
            print(e)
