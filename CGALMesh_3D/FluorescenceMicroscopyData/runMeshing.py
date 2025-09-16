import os
import glob
import subprocess

from multiprocessing import Pool

# Define how many jobs you want to run concurrently
MAX_CONCURRENT_JOBS = 10  # Change this number to adjust the number of concurrent jobs


def process_file(mesh_file, output_folder, log_folder):
    try:
        # Run the process_mesh_script.py as a separate process
        result = subprocess.run(
            ["python3", "process_mesh_script.py", mesh_file, output_folder, log_folder],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600,  # Timeout in seconds for each job
        )

        if result.returncode != 0:
            print(
                f"Job for {mesh_file} failed with return code {result.returncode}. Error: {result.stderr.decode()}"
            )
        else:
            print(f"Successfully processed {mesh_file}")

    except subprocess.TimeoutExpired:
        print(f"Job for {mesh_file} exceeded the timeout and was canceled.")
    except Exception as e:
        print(f"An unexpected error occurred for {mesh_file}: {e}")


def main():
    input_folder = "/home/bo8482/elegans_meshes_bottom/elegans-netgen"
    mesh_files = glob.glob(os.path.join(input_folder, "*.vol.gz"))
    output_folder = "processed_meshes"
    log_folder = "logs"

    # Create a Pool of workers to process the files concurrently
    with Pool(processes=MAX_CONCURRENT_JOBS) as pool:
        # Each job corresponds to processing a file
        pool.starmap(
            process_file,
            [(mesh_file, output_folder, log_folder) for mesh_file in mesh_files],
        )


if __name__ == "__main__":
    main()
