import os
import shutil


def clear_directory(directory):
    if os.path.exists(directory):
        # Remove all files and subdirectories
        shutil.rmtree(directory)
        os.makedirs(directory)  # Recreate the directory
    else:
        os.makedirs(directory)  # Create the directory if it doesn't exist

