import lmdb

# Path to your LMDB folder on the Marcus drive
lmdb_path = "/media/alex/Marcus/00_DATA/10B_LMDB/Celeb-DF-v1_lmdb"

# Open the database in read-only mode
env = lmdb.open(lmdb_path, readonly=True, lock=False)

print("--- First 5 Keys in LMDB ---")
with env.begin() as txn:
    cursor = txn.cursor()
    for i, (key, value) in enumerate(cursor):
        # Decode the byte-key to a string
        print(f"Key {i}: {key.decode('utf-8')}")
        if i >= 200: break

env.close()