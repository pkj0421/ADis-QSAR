# ADis-QSAR
Activity Differences - Quantatative Structure Activity Relationship

## Installation
This code was tested in Python 3.11.

A yaml file containing all requirements is provided. 

This can be readily setup using conda.

```sh
conda env create -f ADis-QSAR-env.yaml
conda activate ADis-QSAR-env
```

## Getting Started
* Prepare 

Check the collected compounds from ChEMBL and classified them into active and inactive

Compounds are automatically classified using the criteria below:

active (IC50, Ki, Kd <= 100nM), inactive (IC50, Ki, Kd >= 1000nM, %Inhibition <= 20%)

The data format is based on ChEMBL ('-chembl' option)

If add %Inhibition assay results ('-i' option)

This code can only be applied to raw data of ChEMBL

Outputs : active, inactive and total compounds

```sh
python Prepare.py -d raw_data_path -o output_path -i -chembl
```

For example:

```sh
python Prepare.py -d Dataset/ChEMBL/ALK/ALK_raw.tsv -o Dataset/ChEMBL/ALK -i -chembl
```

* Preprocess

Selecting central structures (50 compounds) and generating descriptors using a pair system

Fingerprint types : radius_size (2: ECFP4, 3: ECFP6), number_of_bits (256, 512)

The scaler can be chosen from three options: Standard, MinMax and Robust

Afterwards, the compounds are divided into training (train), validation (valid) and test (test) sets

If you want to generate test set with other data ('-t' option)

The default value for the validation set size is 0.2, but it can be changed ('-v' option)

The number of active and inactive compounds is automatically adjusted to a ratio of 1:1.5 each set

Outputs : g1 (50 compounds), train, valid and test sets

```sh
python Preprocessing.py -a active_path -i inactive_path -o output_path -v valid_size -r radius_size -b number_of_bits -s scaler_type -core num_cores -t
```

For example:

```sh
python Preprocessing.py -a Dataset/ChEMBL/ALK/ALK_prepare/ALK_active.tsv -i Dataset/ChEMBL/ALK/ALK_prepare/ALK_inactive.tsv -o Dataset/ChEMBL/ALK -v 0.2 -r 2 -b 256 -s Standard -core 12
```

* ADis_QSAR

Start model training 

Use model type such as SVM, MLP, RF and XGB ('-m' option) 

If test set is available ('-test' option)

The test set does not participate in training/validation

You can obtain directly prediction results from the generated model

Outputs : model, log files

```sh
python ADis_QSAR.py -train train_path -valid valid_path -test test_path -m model_type -o output_path -core num_cores
```

For example:

```sh
python ADis_QSAR.py -train Dataset/ChEMBL/ALK/ALK_preprocessing/ALK_train_vector.tsv -valid Dataset/ChEMBL/ALK/ALK_preprocessing/ALK_valid_vector.tsv -test Dataset/ChEMBL/ALK/ALK_preprocessing/ALK_test_vector.tsv -m SVM -o Dataset/ChEMBL/ALK/ALK_preprocessing -core 12
```

* Predict

Predicting external dataset from the generated model

If you would like to apply an external dataset to the trained model, use the following code

Outputs : predict log file

```sh
python Predict.py -m model_path -e external_path -n external_name -o output_path -ev 
```

For example:

```sh
python Predict.py -m Dataset/ChEMBL/ALK/ALK_preprocessing/ALK_model/SVM/ALK_SVM_model.pkl -e Dataset/ChEMBL/ALK/ALK_preprocessing/ALK_test_vector.tsv -n ext -o Dataset/ChEMBL/ALK -core 12 -ev
```

## Baseline run

This code for generating a baseline model for comparing the performance of ADis-QSAR

Training a model using raw fingerprints (binary data) with the ADis-QSAR approach

The entire process for performing ADis-QSAR is executed automatically

```sh
python Baseline_run.py
```

## Vary params run

This is the execution code for comparing the performance of ADis-QSAR based on various parameter changes

The parameters that are being modified are as follows:

a. number of center structures (g1) : [20, 50, 80]

b. vary the radius size : [ECFP4, ECFP6]

c. vary the number of bits : [256, 512]

d. vary the scaler : [ECFP4, ECFP6]

The entire process of performing ADis-QSAR, including parameter switching, is performed automatically

```sh
python Vary_params_run.py
```

## Contact (Questions/Bugs/Requests)
Please submit a GitHub issue or contact me [rudwls2717@naver.com](rudwls2717@naver.com)

## Acknowledgements
Thank you for our [Laboratory](https://homepage.cnu.ac.kr/cim/index.do).

If you find this code useful, please consider citing my work.
