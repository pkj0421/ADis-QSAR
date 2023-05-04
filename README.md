# ADis-QSAR
Activity Differences Quantatative Structure Activity Relationship

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

Check the collected compounds and separate them into active and inactive compounds.

The data format is based on ChEMBL. ('-chembl' option)

If add %Inhibition assay results. ('-i' option)


```sh
python Prepare.py -d raw_data_path -o output_path -i -chembl
```

* Preprocess

Selecting central structures and generating descriptors using a pair system.

It is recommended to use a ratio of 1:~1.5 for active and inactive compounds.

```sh
python Preprocessing.py -a active_path -i inactive_path -t test_size -r radius_size -b number_of_bits -o output_path -core num_cores
```

* ADis_QSAR

Start training the model using model type ('RF', 'XGB', 'SVM') options. 

Add '-ext external_path' option if you have external set beforehand.

```sh
python ADis_QSAR.py -train train_path -test test_path -m model_type -o output_path -core num_cores
```

* Predict

Predicting new dataset (raw) from the generated model.

```sh
python Predict.py -m model_path -e external_path -n external_name -o output_path -core num_cores -g1 g1_path -s scaler_path -r radius_size -b number_of_bits 
```

If using a preprocessed dataset, use the following code.

```sh
python Predict.py -m model_path -e external_path -n external_name -o output_path -core num_cores -ev
```

## Contact (Questions/Bugs/Requests)
Please submit a GitHub issue or contact me [rudwls2717@naver.com](rudwls2717@naver.com)

## Acknowledgements
Thank you for our [Laboratory](https://homepage.cnu.ac.kr/cim/index.do).

If you find this code useful, please consider citing my work.
