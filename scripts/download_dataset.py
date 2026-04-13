from libero.libero import get_libero_path, benchmark
from termcolor import colored

import libero.libero.utils.download_utils as download_utils
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="LIBERO Download script")
    LIBERO_DATASETS = ["libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90"]
    parser.add_argument("dataset", type=str, help=f"Options available: {LIBERO_DATASETS}")
    args = parser.parse_args()

    # Get download path
    download_dir = get_libero_path("datasets")

    # Get benchmarks
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # Get the list of dataset to download
    datasets = LIBERO_DATASETS if args.dataset  == "all" else [args.dataset]

    for dataset in datasets:
        benchmark_instance = benchmark_dict[dataset]()
        num_tasks = benchmark_instance.get_num_tasks()

        libero_datasets_exist = download_utils.check_libero_dataset(download_dir=download_dir)

        if not libero_datasets_exist:
            # download_utils.libero_dataset_download(download_dir=download_dir, datasets=datasets)
            download_utils.download_from_huggingface(dataset_name=dataset, download_dir=download_dir)

        # Check if the demo files exist
        demo_files = [os.path.join(download_dir, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]
        for demo_file in demo_files:
            if not os.path.exists(demo_file):
                print(colored(f"[error] demo file {demo_file} cannot be found. Check your paths", "red"))


if __name__ == "__main__":
    main()