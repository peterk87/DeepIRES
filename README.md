# DeepIRES

<!---
TODO: on merge into main repo, replace badge URL with 
[![CI](https://github.com/SongLab-at-NUAA/DeepIRES/actions/workflows/ci.yml/badge.svg)](https://github.com/SongLab-at-NUAA/DeepIRES/actions/workflows/ci.yml)
TODO: Add Bioconda and PyPI badges
--->
[![CI](https://github.com/peterk87/DeepIRES/actions/workflows/ci.yml/badge.svg)](https://github.com/peterk87/DeepIRES/actions/workflows/ci.yml)
[![Cite Paper](http://img.shields.io/badge/DOI-10.1093/bib/bbae439-1073c8)](https://doi.org/10.1093/bib/bbae439)

DeepIRES: a hybird deep learning model for indentifying internal ribosome entry site in mRNA

## Installation

### Conda

Download the repository and create corresponding environment.

```bash
git clone https://github.com/SongLab-at-NUAA/DeepIRES.git
cd DeepIRES
conda env create -n deepires -y python=3.11 pip
conda activate deepires
pip install .
deepires --help
```

## USAGE

Show help info with `$ deepires --help`

```
 Usage: deepires [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --input               -i      PATH  FASTA file containing the sequences to predict [default: None] [required]                                             │
│    --output              -o      PATH  Output CSV file containing the predictions [default: deepires-results.csv]                                            │
│    --model               -m      PATH  Model weights file [default: /path/to/weights/first]                                                                  │
│    --output-seqs         -s            Output the sequences with the predictions                                                                             │
│    --verbose             -v            Increase verbosity                                                                                                    │
│    --version             -V            Show version                                                                                                          │
│    --install-completion                Install completion for the current shell.                                                                             │
│    --show-completion                   Show completion for the current shell, to copy it or customize the installation.                                      │
│    --help                              Show this message and exit.                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 deepires version 0.0.1; Python 3.11.10
```

Predict IRES in sequences in an input FASTA file:

```bash
deepires -i input.fasta
```

### Running on the example data

```bash
deepires -i data/main_independent.fa -o main.csv
deepires -i data/core.fa -o core.csv
deepires -i data/5utr_independent.fa -o 5utr.csv
```

### Output

There are four columns in the prediction results tabular CSV output file：

1. sequence name
2. IRES score
3. the start locations of region may contain IRES
4. the termination locations of region may contain IRES

## Citation

Please cite the [DeepIRES publication](https://academic.oup.com/bib/article/25/5/bbae439/7749489):

```bibtex
@article{10.1093/bib/bbae439,
    author = {Zhao, Jian and Chen, Zhewei and Zhang, Meng and Zou, Lingxiao and He, Shan and Liu, Jingjing and Wang, Quan and Song, Xiaofeng and Wu, Jing},
    title = "{DeepIRES: a hybrid deep learning model for accurate identification of internal ribosome entry sites in cellular and viral mRNAs}",
    journal = {Briefings in Bioinformatics},
    volume = {25},
    number = {5},
    pages = {bbae439},
    year = {2024},
    month = {09},
    abstract = "{The internal ribosome entry site (IRES) is a cis-regulatory element that can initiate translation in a cap-independent manner. It is often related to cellular processes and many diseases. Thus, identifying the IRES is important for understanding its mechanism and finding potential therapeutic strategies for relevant diseases since identifying IRES elements by experimental method is time-consuming and laborious. Many bioinformatics tools have been developed to predict IRES, but all these tools are based on structure similarity or machine learning algorithms. Here, we introduced a deep learning model named DeepIRES for precisely identifying IRES elements in messenger RNA (mRNA) sequences. DeepIRES is a hybrid model incorporating dilated 1D convolutional neural network blocks, bidirectional gated recurrent units, and self-attention module. Tenfold cross-validation results suggest that DeepIRES can capture deeper relationships between sequence features and prediction results than other baseline models. Further comparison on independent test sets illustrates that DeepIRES has superior and robust prediction capability than other existing methods. Moreover, DeepIRES achieves high accuracy in predicting experimental validated IRESs that are collected in recent studies. With the application of a deep learning interpretable analysis, we discover some potential consensus motifs that are related to IRES activities. In summary, DeepIRES is a reliable tool for IRES prediction and gives insights into the mechanism of IRES elements.}",
    issn = {1477-4054},
    doi = {10.1093/bib/bbae439},
    url = {https://doi.org/10.1093/bib/bbae439},
    eprint = {https://academic.oup.com/bib/article-pdf/25/5/bbae439/59021567/bbae439.pdf},
}
```

## Development

In addition to the source code the CLI and Tensorflow/Keras neual net model under `src/`, this repository contains four other folders: `dataset`, `weights`, `data`, and `result`.

### Dataset

This folder contains orginal data, traing dataet and testing dataset we constructed.

### Weights

This folder saves the model weights we trained.

### Data

`data/` contains the test input FASTA files for IRES prediction.

### Result

This folder contains the expected prediction outputs for each input FASTA file under `data/`
