Debiasing Vandalism Detection Models at Wikidata: Classification and Evaluation
===============================================================================

The Wikidata Vandalism Detectors FAIR-E and FAIR-S are machine learning models for automatic vandalism detection in Wikidata without discriminating against anonymous editors. They were developed as a joint project between Paderborn University and Leipzig University.

This is the classification and evaluation component for FAIR-E, FAIR-S and the baselines WDVD, ORES, and FILTER. The feature extraction can be done with the corresponding [feature extraction component](https://github.com/heindorf/www19-fair-feature-extraction).

Paper
-----

This source code forms the basis for our WWW 2019 paper [Debiasing Vandalism Detection Models at Wikidata](https://doi.org/10.1145/3308558.3313507). When using the code, please make sure to refer to it as follows:

```TeX
@inproceedings{heindorf2019debiasing,
  author    = {Stefan Heindorf and
               Yan Scholten and
               Gregor Engels and
               Martin Potthast},
  title     = {Debiasing Vandalism Detection Models at Wikidata},
  booktitle = {{WWW}},
  publisher = {{ACM}},
  year      = {2019}
}
```

Classification and Evaluation Component
---------------------------------------

### Requirements

The code was tested with Python 3.5.2, 64 Bit under Windows 10 with 16 cores and 256 GB RAM.

### Installation

We recommend [Miniconda](http://conda.pydata.org/miniconda.html) for easy installation on many platforms.

1. Create new environment:  
   `conda create --name www19-fair python=3.5.2 --file requirements.txt`
2. Activate environment:  
   `activate www19-fair`
3. Install Kernel:  
   `python -m ipykernel install --user --name www19-fair --display-name www19-fair`
3. Start Jupyter:  
   `jupyter notebook`

### Execute Notebooks

Run the Jupyter notebooks in this order:

```
01-dataset-analysis.ipynb
02-truth-biases.ipynb
03-baselines.ipynb
04-FAIR-E.ipynb
05-FAIR-S.ipynb
06-evaluation.ipynb
```

### Required Data

We assume the following project structure:

```
www19-fair/
├── data/
│   ├── classification/
│   ├── corpus-validity/
│   ├── external/
│   │   └─── wdvc-2016/
│   ├── features/
│   │   ├── test/
│   │   │   ├── embeddings/
│   │   │   └── features.csv.bz2
│   │   ├── training/
│   │   │   ├── embeddings/
│   │   │   └── features.csv.bz2
│   │   ├── validation/
│   │   │   ├── embeddings/
│   │   │   └── features.csv.bz2
│   │   └── wdvd_features.csv.bz2
│   ├── item-properties/
│   └── property-domains/
└── www19-fair-classification/
```

**classification:** This folder will contain the output of the classification component: plots, tables, and vandalism scores. Initially, it can be empty.

**corpus-validity:** Manually reviewed Wikidata revisions. You can download the folder [corpus-validity](https://groups.uni-paderborn.de/wdqa/www19-fair/data/corpus-validity/).

**external:** Contains the [Wikidata Vandalism Corpus 2016](https://www.wsdm-cup-2017.org/vandalism-detection.html).

**features:** Contains the features for our models. The feature extraction can be done with the [feature extraction component](https://github.com/heindorf/www19-fair-feature-extraction). Alternatively, you can download the [features](https://groups.uni-paderborn.de/wdqa/www19-fair/data/features/) directly.

**item-properties:** The list of Wikidata item properties at the end of the training set. The file can be created with the [feature extraction component](https://github.com/heindorf/www19-fair-feature-extraction). Alternatively, you can download the [item-properties](https://groups.uni-paderborn.de/wdqa/www19-fair/data/item-properties/) directly.

**property-domains:** The domain each Wikidata property belongs to. You can download the folder [property-domains](https://groups.uni-paderborn.de/wdqa/www19-fair/data/property-domains/).

**www19-fair-feature-classification:** This git repository.


Known Issues
------------

The dataset contains some revisions that change references of subject-predicate-object triples instead of subject-predicate-object triples themselves. In order to filter all references, in the notebook `01-dataset-analysis.ipynb`, the condition `df['revisionAction'].isin(revisionActions)` must be changed to `(df['revisionAction'].isin(revisionActions) & df['param4'].isna())`. This change has little effect on our evaluation results. For consistency to the paper, we use the original version in this repository.


Contact
-------

For questions and feedback please contact:

Stefan Heindorf, Paderborn University  
Yan Scholten, Paderborn University  
Gregor Engels, Paderborn University  
Martin Potthast, Leipzig University  

License
-------

The code by Stefan Heindorf, Yan Scholten, Gregor Engels, Martin Potthast is licensed under a MIT license.
