import os
import csv

ALPHA = 40.0
R_MIN = 1.0
R_MAX = 5.0

def confidence(rating):
    return 1.0 + ALPHA * (rating - R_MIN) / (R_MAX - R_MIN)

def transform_dataset(dataset_path, rating_threshold=4.0):
    csv_file = os.path.join(dataset_path, 'ratings.csv')
    dat_file = os.path.join(dataset_path, 'ratings.dat')
    
    if os.path.exists(csv_file):
        transform_csv(csv_file, rating_threshold)
    elif os.path.exists(dat_file):
        transform_dat(dat_file, rating_threshold)
    else:
        print(f"nao tem o arquivo")

def transform_csv(input_file, rating_threshold=4.0):
    output_file = input_file.replace('.csv', '_implicit.csv')
    
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        writer.writerow(header + ['preference', 'confidence'])
    
        for row in reader:
            if len(row) >= 3:
                try:
                    rating = float(row[2])
                    row.append('1')
                    row.append(f'{confidence(rating):.4f}')
                    writer.writerow(row)
                except ValueError:
                    pass

def transform_dat(input_file, rating_threshold=4.0):
    #usando separador :: pq o dat tem isso
    output_file = input_file.replace('.dat', '_implicit.dat')
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                try:
                    rating = float(parts[2])
                    parts.append('1')                          
                    parts.append(f'{confidence(rating):.4f}') 
                    outfile.write('::'.join(parts) + '\n')
                except ValueError:
                    pass

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_folders = ['ml-100k', 'ml-1m', 'ml-10m', 'ml-latest-small']
    
    for folder in dataset_folders:
        dataset_path = os.path.join(current_dir, folder)
        transform_dataset(dataset_path)