#!/bin/env python
import pandas as pd
import pyarrow
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
LABEL_ID = "ID"
LABEL_PHENOTYPE = "phenotype"
LABEL_CLUSTER = "cluster"

CLUSTER_0 = "1"
CLUSTER_1 = "2"


def main():
    # --- read bin file
    print(f"Reading {phenotypes_file_name}...", flush=True)

    phenotypes_df = pd.read_csv(phenotypes_file_name, delim_whitespace=True, names=[LABEL_ID, LABEL_PHENOTYPE, LABEL_CLUSTER],
                                skiprows=1, skipinitialspace=True,
                                # true_values=[CLUSTER_1],
                                # false_values=[CLUSTER_0],
                                converters={LABEL_PHENOTYPE: lambda v: v == "2", LABEL_CLUSTER: lambda v: v == CLUSTER_1}
                                )

    print("\tSorting...", flush=True)
    phenotypes_df.sort_values(by=[LABEL_ID])  # make sure it's sorted

    print("\tStats:")
    print(f"\t\tNumber of elements: {len(phenotypes_df[LABEL_ID].values)}")
    print(f"\t\tPercentage of resistant: "
          f"{(phenotypes_df[LABEL_PHENOTYPE].values == False).sum() / len(phenotypes_df[LABEL_ID].values) * 100 : .2f} %")
    print(f"\t\tPercentage of susceptible: "
          f"{(phenotypes_df[LABEL_PHENOTYPE].values == True).sum() / len(phenotypes_df[LABEL_PHENOTYPE].values) * 100 : .2f} %")
    print("Done.", flush=True)
    # ---

    print()

    # --- read vcf file
    print(f"Reading {vcf_file_name}...")

    print("\tCorrecting header...")
    vcf_header_df = pd.read_csv(vcf_file_name, delim_whitespace=True, skiprows=29, nrows=1, dtype=str, header=None)
    header = [LABEL_ID]  # empty string for id column
    for x in vcf_header_df.loc[0][9:]:
        u = len(x) // 2
        assert x[u] == "_"
        header.append(x[:u])

    print(f"\t\tHeader length: {len(header)}")
    print("\tReading whole file...", flush=True)
    vcf_df = pd.read_csv(vcf_file_name, sep="\t", skiprows=30, skipinitialspace=True,
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
    to_remove = vcf_df.columns[[0, 1] + list(range(3, 9))]
    vcf_df = vcf_df.drop(columns=to_remove, axis=1)

    print()

    # attach corrected header
    vcf_df.columns = header

    # sort individuals
    vcf_df = vcf_df.sort_index(axis=1)

    # transpose
    vcf_df = vcf_df.T

    # set first row as header
    vcf_df.columns = vcf_df.iloc[0]
    vcf_df = vcf_df.iloc[1:]

    # append phenotypes
    vcf_df[LABEL_PHENOTYPE] = phenotypes_df[LABEL_PHENOTYPE].values
    vcf_df[LABEL_CLUSTER] = phenotypes_df[LABEL_CLUSTER].values

    columns = []
    for col in vcf_df.columns:
        col: str
        columns.append(col.replace(".", "_"))
    vcf_df.columns = columns

    print("Writing...", flush=True)
    # vcf_df.to_csv(main_csv_name)
    vcf_df.to_parquet(main_parquet_name)
    print("Done.")
    # ---


if __name__ == "__main__":
    # --- parameters
    # input
    vcf_file_name = "imputed_VNN_WGS_ALL.vcf"
    phenotypes_file_name = "pheno_cluster.csv"
    # output
    main_csv_name = "main.csv"
    main_parquet_name = "main.parquet"
    add_csv_name = "add.csv"
    # ---

    main()
