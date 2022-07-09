import unicodedata
import os
import json

def read_captions_txt(path):
    """ 
    Reads captions from a text file.
    """
    with open(path, 'r') as f:
        captions = f.readlines()
    
    entries = {}
    for entry in captions:
        entry_name, caption = entry.split('\t')
        caption = caption.strip().rstrip().replace('\n','')
        entries[entry_name] = caption
    return entries

def create_keywords_dict(path):
    with open(os.path.join(path), 'r') as f:
        keywords = f.readlines()

    keyword_dict = {}
    for keyword in keywords:
        filename, _, *keywords = keyword.split('\t')
        if keywords != []:
            keywords[-1] = keywords[-1].replace('\n','')
            keyword_dict[filename] = keywords
        else:
            keyword_dict[filename] = []
    
    return keyword_dict



def create_json_files(root):
    """
    Input: Root str of Roco dataset
    Output: Creates a json file for each of the train, val, and test sets
    """
    for mode in ['train', 'validation', 'test']:
        print(f"Creating json file for {mode}")
        entries = []
        # Roco has both Radiology and Non-Radiology datasets
        # Non Radiology 
        nonrad_path = os.path.join(root, mode, 'non-radiology')
        nonrad_path_images_path = os.path.join(nonrad_path, 'images')
        nonrad_path_captions_path = os.path.join(nonrad_path, 'captions.txt')
        nonrad_path_keywords_path = os.path.join(nonrad_path, 'keywords.txt')
        
        # Reads captions as dictionary and keys are file name and values are captions
        non_rad_captions= read_captions_txt(nonrad_path_captions_path)
        non_rad_keywords = create_keywords_dict(nonrad_path_keywords_path)
        for entry  in os.listdir(nonrad_path_images_path):
            path_image = os.path.join(nonrad_path_images_path, entry)
            caption_img = non_rad_captions[entry.replace('.jpg','')]
            caption_img = unicodedata.normalize("NFKC", caption_img)
            keywords = non_rad_keywords[entry.replace('.jpg','')]
            entries.append({'image_path': path_image, 'caption': caption_img, 'keywords': keywords})


        # Radiology
        radiology_path = os.path.join(root, mode, 'radiology')
        rad_images_path = os.path.join(radiology_path, 'images')
        rad_captions_path = os.path.join(radiology_path, 'captions.txt')
        rad_keywords_path = os.path.join(radiology_path, 'keywords.txt')

        # Reads captions as dictionary and keys are file name and values are captions
        rad_captions= read_captions_txt(rad_captions_path)
        rad_keywords = create_keywords_dict(rad_keywords_path)
        for entry  in os.listdir(rad_images_path):
            path_image = os.path.join(rad_images_path, entry)
            caption_img = rad_captions[entry.replace('.jpg','')]
            caption_img = unicodedata.normalize("NFKC", caption_img)

            keywords = rad_keywords[entry.replace('.jpg','')]
            entries.append({'image_path': path_image, 'caption': caption_img}) 

        # Write to json file
        with open(os.path.join(root, mode, mode + '.json'), 'w',) as f:
            json.dump(entries, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":

    if os.getcwd().startswith('/home/mlmi-matthias'):
        root_path = '/home/mlmi-matthias/roco-dataset/data'

    elif os.getcwd().startswith('/Users/caghankoksal'):
        root_path = '/Users/caghankoksal/Desktop/development/roco-dataset/data'

        
    create_json_files(root_path)