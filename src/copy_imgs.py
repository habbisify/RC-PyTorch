import csv
import shutil
import glob
import os
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset_dir')
    p.add_argument('out_dir')
    flags = p.parse_args()

    dataset_dir = flags.dataset_dir
    out_dir = flags.out_dir
       
    q_path = os.path.join(dataset_dir, "optimal_q.txt")
    
    with open(q_path, newline='') as q_file:
        reader = csv.reader(q_file, delimiter=',')
        next(reader, None) # First row out

        for [img_name, q] in reader:
            wild_path = dataset_dir + "_bpg_q" + q + "/" + img_name + "*.png"
            
            for complete_path in glob.glob(wild_path):                                
                shutil.copyfile(complete_path, out_dir + "/" + img_name + ".png")       


if __name__ == '__main__':
    main()
