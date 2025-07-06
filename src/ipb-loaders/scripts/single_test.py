import subprocess

command = " export dataloader={} && python3 -m unittest test/test_{}.py"

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Script to test only one dataloader at a time.")
    arg_parser.add_argument(
        "--type",
        "-t",
        dest="type",
        required=True,
        help="dataset type muste be rgb, rgbd or lidar",
    )
    arg_parser.add_argument(
        "--dataloader",
        "-d",
        dest="dataloader",
        required=True,
        help="dataloader class name",
    )

    args = arg_parser.parse_args()
    cmd = command.format(args.dataloader, args.type)
    subprocess.call(cmd, shell=True)
