import pandas as pd

# Read the first table from CSV file with header in the first row
ensemble = pd.read_csv('/home/enrico/PycharmProjects/variant-classifier/datasets/exclude-chr3/data_ensemble.csv')

# Read the second table from CSV file with header 'snp_id'
dir = '/home/enrico/PycharmProjects/variant-classifier/datasets/exclude-chr3/regions/'
region_file = dir + 'narrow.csv'
region = pd.read_csv(region_file)
region.columns = ['snp_id']

# Query the first table with elements from the second table
result = ensemble[ensemble['snp_id'].isin(region['snp_id'])]
print(f"Ensemble size: {len(result)}")

# Compute the counts of 'funct'
counts = result.groupby(['funct']).size().reset_index(name='count')
print(counts)

# Compute the counts of 'n_tissue'
counts = result.groupby(['n_tissue']).size().reset_index(name='count')
print(counts)
