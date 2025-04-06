import json
import os
from tqdm import tqdm

def extract_captions(annotations_file, output_file):
    """
    Extract image captions from COCO annotations file and save them
    to a text file with one caption per line.
    
    Args:
        annotations_file: Path to COCO captions annotations JSON file
        output_file: Path where to save the extracted captions
    """
    print(f"Reading annotations from {annotations_file}")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a mapping from image ID to image filename
    image_id_to_filename = {}
    for image in data['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # Extract captions
    captions = []
    print("Extracting captions...")
    for annotation in tqdm(data['annotations']):
        image_id = annotation['image_id']
        if image_id in image_id_to_filename:
            image_filename = image_id_to_filename[image_id]
            caption = annotation['caption']
            captions.append((image_filename, caption))
    
    # Save captions to file
    print(f"Writing {len(captions)} captions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_filename, caption in captions:
            f.write(f"{image_filename}|{caption}\n")
    
    print("Caption extraction completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract captions from COCO annotations')
    parser.add_argument('--annotations_file', type=str, required=True, 
                        help='Path to COCO captions annotations JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path where to save the extracted captions')
    
    args = parser.parse_args()
    
    extract_captions(args.annotations_file, args.output_file) 