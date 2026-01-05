import os
import json
import cv2
import lmdb
import yaml
from PIL import Image
import io
import numpy as np
import argparse
from tqdm import tqdm # ADD THIS

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
    """Creates a full LMDB dataset with a progress bar and file tracking"""
    
    db = lmdb.open(
        lmdb_path, 
        map_size=map_size,
        metasync=False, 
        sync=False, 
        map_async=True,
        writemap=True
    )

    # First, quickly count files so we can have a % progress bar
    print("Counting files...")
    all_files = []
    for root, _, files in os.walk(source_folder, followlinks=True):
        for file in files:
            if not file.startswith('.'):
                all_files.append(os.path.join(root, file))
    
    total_files = len(all_files)
    print(f"Total files to add: {total_files}")
    print(f"Starting LMDB creation at: {lmdb_path}")
    
    commit_every = 500
    count = 0
    txn = db.begin(write=True)
    
    # Use tqdm for the progress bar
    # 'desc' is the label, 'unit' is the item type
    pbar = tqdm(total=total_files, desc="Building LMDB", unit="file")

    for image_path in all_files:
        # Update the progress bar label with the current file name (shortened)
        short_name = os.path.basename(image_path)
        pbar.set_postfix(file=short_name[:20]) 

        # 1. Create the relative path key
        rel_p = os.path.relpath(image_path, source_folder)
        relative_path = os.path.join(dataset_name, rel_p)
        
        # 2. Standardize slashes
        key_string = relative_path.replace('\\', '/')
        key = key_string.encode('utf-8')
        
        try:
            # 3. Read and write data
            value = file_to_binary(image_path)
            txn.put(key, value)
            
            count += 1
            pbar.update(1) # Advance the bar

            if count % commit_every == 0:
                txn.commit()
                txn = db.begin(write=True)
                
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")

    # Final commit
    txn.commit()
    pbar.close()
    db.close()
    print(f"\nFinished! Total items added: {count}")

if __name__ == '__main__':
    # ... (Keep the same argparse and YAML logic as your previous script) ...
    parser = argparse.ArgumentParser(description='Create LMDB for DeepfakeBench')
    parser.add_argument('--dataset_size', type=int, default=100, required=True,
                        help='LMDB size in GB')

    args = parser.parse_args()

    yaml_path = '/app/preprocessing/config.yaml'
    
    try:
        with open(yaml_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config['to_lmdb']
    except Exception as e:
        print(f"YAML error: {e}")
        exit(1)

    dataset_name = config['dataset_name']['default']
    dataset_root_path = config['dataset_root_path']['default']
    output_lmdb_dir = config['output_lmdb_dir']['default']
    
    os.makedirs(output_lmdb_dir, exist_ok=True)
    
    dataset_dir_path = os.path.join(dataset_root_path, dataset_name)
    lmdb_path = os.path.join(output_lmdb_dir, f"{dataset_name}_lmdb")
    
    map_size_bytes = int(args.dataset_size) * 1024 * 1024 * 1024
    
    create_lmdb_dataset(dataset_dir_path, lmdb_path, dataset_name, map_size=map_size_bytes)