import json
import argparse

def merge_ds(datasets,out_file,dedup=True,verbose=False):
    common_objects = []
    idx = 1
    for dataset in datasets:
        with open(dataset, "r") as file:
            ds = json.load(file)
            for data in ds:
                if (dedup):
                    if data not in common_objects:
                        common_objects.append(data)
                    else:
                        print(f"Removed duplicate -> {idx}")
                        idx += 1
                else:
                    common_objects.append(data)
                
    print(f"Combined Dataset Size: {len(common_objects)}")
    # Write common objects to output JSON
    with open(out_file, "w", encoding='utf8') as outfile:
        json.dump(common_objects, outfile, indent=4, ensure_ascii=True)
    
def main():
    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='Merge Datasets')
    
    # Define the arguments
    parser.add_argument('--datasets', required=True, nargs="+", help='List of datasets to merge')
    parser.add_argument('--output_file', type=str, default="combined_dataset.json", help="output filename")
    parser.add_argument('--dedup', type=bool, default=True, help='Deduplicate entries')
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose output")
    
    # Parse the arguments
    args = parser.parse_args()
    merge_ds(args.datasets, args.output_file, args.dedup, args.verbose)
    
if __name__ == "__main__":
    main()