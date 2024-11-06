
# Variational Autoencoders to Integrate Gene expression and Histone marks Dynamics across Cardiac differentiation 

This repository contains a computational pipeline designed for the analysis of RNA-seq and ChIP-seq data through various stages of cardiac differentiation in mice. The pipeline leverages multiple tools for data processing, normalization, dimensionality reduction, and clustering to explore gene regulatory dynamics throughout differentiation.

## Usage
1. **Data Mapping**: Use scripts in `01_Mapping` to download and align raw data.
2. **Differential Expression Analysis**: `02_DESeq/DESeq2.Rmd` runs differential expression on the processed RNA-seq data.
3. **Signal Recovery**: Run notebooks and scripts in `03_RecoverSignal` to focus on promoter region signals.
4. **Modeling**: Use notebooks in `04_Models/jupyter_notebooks` to preprocess data, train models, and perform dimensionality reduction and clustering analyses.

## Dependencies
Ensure the required Python libraries and R packages are installed. Use `DL.yml` for setting up the python environment in conda.

## Program versions

| Program  | Version |
| -------- | ------- |
| NCBI-SRA | 3.0.10  |
| BOWTIE   | 2.5.3   |
| TOPHAT   | 2.0.14  |
| SAMTOOLS | 1.19.2  |

# Contents description

### `01_Mapping/`
This directory contains scripts and Jupyter notebooks for downloading, parsing, and mapping ChIP-seq and RNA-seq data. The main steps are as follows:
- **ChIP-seq and RNA-seq Parsing and Downloading**: Scripts such as `01_1_ChIP_ENA_table_parse.ipynb` and `02_1_RNA_ENA_table_parse.ipynb` process tables of data from public repositories.
- **Mapping Scripts**: `MapChIPseq2.pl` and `ProcessRNAseq.pl` handle sequence mapping to the mouse genome.
  
Files:
- `ChIP_ENA_table.tsv` and `RNA_ENA_table.tsv`: Metadata tables for sequencing datasets.
---
### `02_DESeq/`
This directory includes the `DESeq2.Rmd` R markdown file for differential expression analysis of RNA-seq data, enabling identification of differentially expressed genes across stages of cardiac differentiation. A helper script, `GenerateSymLinks.sh`, organizes files for DESeq2 processing.

---
### `03_RecoverSignal/`
Scripts for generating BED files and counting reads in specific promoter regions are included here:
- **Signal Recovery Notebooks and Scripts**: `01_GenerateBed.ipynb` and `02_CountReadsInPromoterRegions.sh` focus on recovering signal data in promoter regions to support downstream analysis.



---
### `04_Models/jupyter_notebooks/`

This directory contains Jupyter notebooks and scripts for data preprocessing, dimensionality reduction, deep learning models training, and clustering to identify patterns in gene expression and epigenetic modifications across cardiac differentiation stages. 

1. **Data Preparation**:
   - `01_DataPreprocessing.ipynb`: Cleans and prepares RNA-seq and ChIP-seq data, standardizing inputs for modeling.

2. **Dimensionality Reduction**:
   - **Autoencoders**: `03_AE_tuning.ipynb` and `04_AE_training.ipynb` perform hyperparameter tuning and training for the autoencoder model.
   - **Variational Autoencoder (VAE)**: `03_VAE_tuning.ipynb` and `04_VAE_training.ipynb` handle VAE tuning and training, with VAE used for compressed representations of gene features.
   - **Classical Approaches**: `04_PCA_training.ipynb` and `04_UMAP_training.ipynb` apply PCA and UMAP, respectively, to reduce data dimensions as alternative methods for comparison.

3. **Latent Space and Clustering Analysis**:
   - `05_LatentSpaceAnalysis.ipynb`: Analyzes the low-dimensional latent space produced by dimensionality reduction models, examining the structure and distribution of gene features.
   - `06_GMMClustering.ipynb`: Uses Gaussian Mixture Modeling (GMM) for clustering within the latent space, grouping genes with potentially shared regulatory patterns.
   - `07_ClustersAnalysis.ipynb`: Further explores cluster characteristics to identify functional groups among the clustered genes.

4. **Functional Enrichment and Visualization**:
   - **Over-Representation Analysis (ORA)**: `01_ORA_RNA_CV.ipynb` and `08_ORA_gmm.ipynb` perform functional enrichment analysis to determine if gene clusters are associated with specific biological pathways or functions.
   - **Visualization**: `09_TSSPlots.ipynb` and `TSSplots.sh` generate transcription start site (TSS) meta-plots

#### Additional Files
- `CustomObjects.py`: Contains custom Python objects and functions imported in the ipynb files

