## A snapshot neural ensemble method for cancer type prediction based on CNVs
Code and supplementary materials for the following two papers:

* "Cancer Risk and Type Prediction Based on Copy Number Variations with LSTM and Deep Belief Networks" submitted to Artificial Intelligence International Conference – A2IC 2018, Barcelona, Spain, November 2018. 
* "A snapshot neural ensemble method for cancer type prediction based on copy number variations", submitted to Journal of Neural Computing and Applications. 

## Initial preprocessing
TCGA provide samples from each patient in a single .txt file. CNV data is extracted from these sample files and merged all together in a sinle .csv file. These data processing is done in raw_cnv_data_processing module. This module has four functions

1. merge_cancer_files:
This function merge each cancer sample data into a single .csv file. While merging the samples, it does some inital preprocessing such as numbering of samples, adding gender (male 1, female 0), cancer type and renaming of columns for our convenience. Here, cancer types 1-14 are 'COAD', 'GBM', 'KIRC', 'LGG', 'LUAD', 'LUSC', 'OV', 'UCEC', 'BRCA', 'HNSC', 'THCA', 'PRAD', 'STAD', 'BLCA' respectively.

2. merge_normal_files:
This function does the similar task as 'merge_cancer_files' but for healthy samples. Also, all healthy samples are considered as type 15 initially.

3. data_preprocessing:
In previous two functions, we mainly extracted data from txt files to a single csv file. In this function, we prepare CNV data for machine learning models. Raw CNV data contains chromosome 'X' and 'Y' as string. We replace them by 23 and 24 respectively. Also, we determine CNV gain or loss and CNV length within this function. Finally, we remove all copy numbers that has segmentation mean less than 0.301 and probe number less than 11. 

4. generate_data_summary:
Here we generate a simple statistical sumaary on the data. For example, total no of samples, total CNVs, no of CNVs per sample, average CNVs per male, female samples etc. 

## Gene data processing
For final data processing we need human gene list along with their locations. TCGA provides gene details as json format. Gene processing is done in gene_data_processing module.

1. extract_gene_names:
Gene details are extracted from json files to a .csv file.

2. Rscript to convert cytobands to gene coordinates:
TCGA provides gene locations as cytobands. To convert cytobands to DNA coordinates we used a Rscript which uses ensemble package of biomaRt library. It takes chromosome name, cytobands as input and return start and end positions of that chromosome.

3. preprocess_gene_list:
Now that we have extracted gene details, we need to process gene data to remove noise. For example, chromosome names 'X', 'Y' are represented as 23, 24 respectively. Remove duplicate gene names etc.

### Final CNVs with oncogenes:
common_moduls->prepare_input_data: here only the oncogenes are taken as features. In TCGA data, is_cancer_gene_census = True are the oncogenes. 

### Final CNVs with protein-coding genes
Recursive feature elimination: common_moduls->feature_ranking: We used Recursive Feature Elimination(RFE) from sklearn.feature_selection. 

## A quick example
A quick example on a small dataset can be performed as follows: 
* $ cd GradCAM
* $ python3 load_data.py (make sure that the data in CSV format in the 'data' folder)
* $ python3 model.py
* $ python3 grad_cam.py

## Citation request
If you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{karim2018recurrent,
        title={Cancer Risk and Type Prediction Based on Copy Number Variations with LSTM and Deep Belief Networks},
        author={Karim, Md Rezaul and Rahman, Md Ashiqur and Decker, Stefan and Beyan Deniz},
        booktitle={Artificial Intelligence International Conference (A2IC2018)},
        year={2018}
    }
    @article{karim2019cae,
        title={A snapshot neural ensemble method for cancer type prediction based on copy number variations},
        author={Karim, Md Rezaul and Rahman, Md Ashiqur and Decker, Stefan and Beyan Deniz},
        journal={Neural Computing and Applications},
        year={2019}
    }

## Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de
