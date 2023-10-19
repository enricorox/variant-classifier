#!/bin/env python
import numpy as np
import pandas as pd

"""
Input file format (vcf):
    __________________________________
    ...variant_info... ...individuals...
    .
    .
    .
    variants
    .
    .
    .
    __________________________________

This file has doubled labels that need to be corrected

Input file format (bin):
    __________________________________
    ? individuals phenotypes
    ...
    __________________________________
    
Output file (main csv):
    __________________________________
    ...variants..., phenotype
    .
    .
    .
    individuals
    .
    .
    .
    __________________________________
    
Output file (add csv):
    __________________________________
    variant_info
    ...
    __________________________________
"""

# --- parameters
# input
vcf_file_name = "/home/enrico/PNRR/imputed_VNN_WGS_ALL.vcf"
imputed_file_name = "/home/enrico/PNRR/bin_file.txt"
# output
main_csv_name = "main.csv"
add_csv_name = "add.csv"
# ---

# --- read bin file
print(f"Reading {imputed_file_name}...")
imputed_df = pd.read_csv(imputed_file_name, sep=" ", names=["?", "individual", "phenotype"])
print("\tSorting...")
imputed_df.sort_values(by=["individual"])  # make sure it's sorted
print("Done.")
print(imputed_df.head())
# ---

# --- read vcf file
print(f"Reading {vcf_file_name}...")

print("\tCorrecting header...")
vcf_header_df = pd.read_csv(vcf_file_name, sep="\t", skiprows=29, nrows=1, dtype=str, header=None)
individuals = []
for x in vcf_header_df.loc[0][9:]:
    u = len(x) // 2
    assert x[u] == "_"
    individuals.append(x[:u])

print("\tReading whole file...", flush=True)
vcf_df = pd.read_csv(vcf_file_name, sep="\t", skiprows=30, nrows=10,
                     # memory_map=True,
                     low_memory=False,
                     header=None,
                     true_values=["0/1", "1/1"],  # mutations on one or both chromosomes
                     false_values=["0/0"]  # no mutation at all
                     )
print(vcf_df.head())
# drop mutation metadata
vcf_df = vcf_df.iloc[:, 9:]
# attach corrected header
vcf_df.columns = individuals
print(vcf_df.head())
# sort individuals
vcf_df = vcf_df.sort_index(axis=1)
print(vcf_df.head())
# transpose
print(vcf_df.T.head())
# ---
