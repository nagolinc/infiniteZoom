#remove images in static/samples

import os
import sys
import shutil

def main(image_dir="static/samples"):
    for file in os.listdir(image_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(image_dir, file))
            print("Removed {}".format(file))

if __name__ == "__main__":
    main()