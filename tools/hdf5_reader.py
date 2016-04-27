'''
Usage:
    python read_hdf5.py -p $PATH_OF_YOUR_HDF5_FILE
'''
from __future__ import print_function
import h5py
import argparse


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
            for key, value in f.attrs.items():
                print("  {}: {}".format(key, value))

            if len(f.items()) == 0:
                return

            for layer, g in f.items():
                print("  {}".format(layer))
                print("    Attributes:")
                for key, value in g.attrs.items():
                    print("      {}: {}".format(key, value))

                print("    Dataset:")
                for p_name in g.keys():
                    param = g[p_name]
                    print("      {}: {}".format(p_name, param.shape))

        elif len(f.keys()):
            print("{} contains: ".format(weight_file_path))
            print("Root keys:")
            print("{KEY}: {SHAPE}")
            for key in f.keys():
                print("  {}: {}".format(key, f[key].shape))

    finally:
        f.close()


def main():
    parser = argparse.ArgumentParser(
        description="Simple tool to read hdf5 format."
        )
    parser.add_argument(
        "-p", "--path", type=str, default=None, required=True,
        help="Path of the hdf5 file."
        )
    args = parser.parse_args()
    print_structure(args.path)

if __name__ == '__main__':
    main()
