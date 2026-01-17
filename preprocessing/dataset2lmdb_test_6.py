import os
import lmdb
import yaml
import numpy as np
import argparse
from tqdm import tqdm

#THIS WORKS FOR CELEB-DF-V1 and stores as relative path names
def file_to_binary(file_path):
    """Convert images or numpy arrays to binary data"""
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        file_binary = data.astype(np.float32).tobytes()
    else:
        with open(file_path, 'rb') as f:
            file_binary = f.read()
    return file_binary

def create_lmdb_dataset(source_folder, lmdb_path, dataset_name, map_size):
    """Creates a full LMDB dataset and prints every file name added"""
    
    db = lmdb.open(
        lmdb_path, 
        map_size=map_size,
        metasync=False, 
        sync=False, 
        map_async=True,
        writemap=True
    )

    print("Step 1: Counting total files (please wait)...")
    all_files = []
    for root, _, files in os.walk(source_folder, followlinks=True):
        for file in files:
            if not file.startswith('.'):
                all_files.append(os.path.join(root, file))
    
    total_files = len(all_files)
    print(f"Step 2: Starting LMDB creation. Total: {total_files} files.")
    
    commit_every = 500
    count = 0
    txn = db.begin(write=True)
    
    # Progress bar setup
    pbar = tqdm(total=total_files, desc="Processing", unit="file")

    for image_path in all_files:
        # 1. Create the key
        rel_p = os.path.relpath(image_path, source_folder)
        relative_path = os.path.join(dataset_name, rel_p)
        key_string = relative_path.replace('\\', '/')
        key = key_string.encode('utf-8')

        # 2. PRINT THE FILE NAME (using tqdm.write to keep the bar at the bottom)
        tqdm.write(f"Adding: {key_string}") 
        
        try:
            # 3. Read and write
            value = file_to_binary(image_path)
            txn.put(key, value)
            
            count += 1
            pbar.update(1)

            # 4. Commit in batches for safety/speed
            if count % commit_every == 0:
                txn.commit()
                txn = db.begin(write=True)
                
        except Exception as e:
            tqdm.write(f"!!! ERROR at {image_path}: {e}")

    txn.commit()
    pbar.close()
    db.close()
    print(f"\nSUCCESS: Added {count} files to {lmdb_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', type=int, required=True, help='Size in GB')
    args = parser.parse_args()

    # Load config and set paths
    #yaml_path = '/app/preprocessing/config.yaml'
    yaml_path = '/home/asabbat/DEV/2026_01_08_CVDFD/CVDeepfakeBench/preprocessing/config.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)['to_lmdb']

    dataset_name = config['dataset_name']['default']
    dataset_root_path = config['dataset_root_path']['default']
    output_lmdb_dir = config['output_lmdb_dir']['default']
    
    os.makedirs(output_lmdb_dir, exist_ok=True)
    dataset_dir_path = os.path.join(dataset_root_path, dataset_name)
    lmdb_path = os.path.join(output_lmdb_dir, f"{dataset_name}_lmdb")
    
    map_size_bytes = int(args.dataset_size) * 1024 * 1024 * 1024
    create_lmdb_dataset(dataset_dir_path, lmdb_path, dataset_name, map_size=map_size_bytes)