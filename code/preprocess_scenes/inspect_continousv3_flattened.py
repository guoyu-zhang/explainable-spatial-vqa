import h5py
import json

def print_h5_overview_and_example(h5_filename):
    with h5py.File(h5_filename, 'r') as hf:
        keys = list(hf.keys())
        print(f"Opened HDF5 file: {h5_filename}")
        print(f"Number of top-level datasets (questions): {len(keys)}")
        # print("Dataset keys:", keys)
        
        if keys:
            example_key = keys[0]
            print(f"\nExample dataset: {example_key}")
            ds = hf[example_key]
            data = ds[()]
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            # Print a snippet of the raw data
            snippet = data[:200] + "..." if len(data) > 200 else data
            print("Raw data snippet:", snippet)
            
            # Attempt to parse the JSON and pretty-print it
            try:
                question = json.loads(data)
                print("\nParsed JSON for the example dataset:")
                print(json.dumps(question, indent=4))
            except Exception as e:
                print("Error parsing JSON:", e)
                print("Raw data:", data)

if __name__ == "__main__":
    h5_filename = "annotated_questions.h5"  # Update the path if needed
    print_h5_overview_and_example(h5_filename)
