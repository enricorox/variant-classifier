#!/bin/env python
import pandas as pd

"""
Convert vcf and bin files into csv.

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
vcf_file_name = "imputed_VNN_WGS_ALL-nochr3.vcf"
phenotypes_file_name = "bin_file.txt"
# output
main_csv_name = "main-nochr3.csv"
add_csv_name = "add.csv"
# ---


def convert(v):
    if v == "2":
        return True
    else:
        return False


# --- read bin file
print(f"Reading {phenotypes_file_name}...", flush=True)

phenotypes_df = pd.read_csv(phenotypes_file_name, delim_whitespace=True, names=["?", "individual", "phenotype"],
                            true_values=["2"],
                            false_values=["1"],
                            converters={"phenotype": convert},
                            )

print("\tSorting...", flush=True)
phenotypes_df.sort_values(by=["individual"])  # make sure it's sorted

print("\tStats:")
print(f"\t\tNumber of elements: {len(phenotypes_df['individual'].values)}")
print(f"\t\tPercentage of resistant: "
      f"{(phenotypes_df['phenotype'].values == False).sum() / len(phenotypes_df['individual'].values) * 100 : .2f} %")
print(f"\t\tPercentage of susceptible: "
      f"{(phenotypes_df['phenotype'].values == True).sum() / len(phenotypes_df['individual'].values) * 100 : .2f} %")
print("Done.", flush=True)
# ---

print()

# --- read vcf file
print(f"Reading {vcf_file_name}...")

print("\tCorrecting header...")
vcf_header_df = pd.read_csv(vcf_file_name, delim_whitespace=True, skiprows=29, nrows=1, dtype=str, header=None)
header = [""]  # empty string for id column
for x in vcf_header_df.loc[0][9:]:
    u = len(x) // 2
    assert x[u] == "_"
    header.append(x[:u])

print("\tReading whole file...", flush=True)
vcf_df = pd.read_csv(vcf_file_name, sep="\t", skiprows=30,
                     # memory_map=True,
                     low_memory=False,
                     header=None,
                     true_values=["0/1", "1/1"],  # mutations on one or both chromosomes
                     false_values=["0/0"]  # no mutation at all
                     )
print("Done.", flush=True)

print()

print(f"Preparing {main_csv_name}...")
# drop mutation metadata
vcf_df = vcf_df.drop(columns=vcf_df.columns[[0, 1] + list(range(3, 9))], axis=1)

# attach corrected header
vcf_df.columns = header

# sort individuals
vcf_df = vcf_df.sort_index(axis=1)

# transpose
vcf_df = vcf_df.T

# set first row as header
vcf_df.columns = vcf_df.iloc[0]
vcf_df = vcf_df[1:]

# append phenotypes
vcf_df["phenotype"] = phenotypes_df["phenotype"].values

print("Writing...", flush=True)
vcf_df.to_csv(main_csv_name)
print("Done.")
# ---
