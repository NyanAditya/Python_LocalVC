import json

def print_json_as_table(file_path="metadata.json"):
    with open(file_path, "r") as f:
        data = json.load(f)

    print("FILENAME\tLABEL\tSPLIT\tORIGINAL")
    
    for filename, info in data.items():
        label = info.get("label", "")
        split = info.get("split", "")
        original = info.get("original", "")
        print(f"{filename}\t{label}\t{split}\t{original}")
        
print_json_as_table("G:\\dfdc_train_part_05\\dfdc_train_part_5\\metadata.json")
