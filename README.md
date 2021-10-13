# Featurization and dimension reduction of a dataset of images using Spark and AWS EMR

This project aims at creating a spark application to:
1) vectorize a dataset of images using transfer learning from a pretrained neural network
2) reduce the dimensions of the vectorized images using PCA
3) be run locally and then be run on a cluster of three nodes (AWS EMR)

The dataset comes from the Kaggle dataset [fruits](https://www.kaggle.com/moltean/fruits)

[link to the app](https://share.streamlit.io/vpvinc/p7_loan_scoring_model/main/Credit_granting_CS_Streamlit/main.py)

## Table of content

1. Structure of the project
2. Installation and set up of spark on Windows 10 with anaconda
3. Featurization using a pre-trained NN ResNet50
4. Dimension reduction using PCA
5. Set-up of the AWS EMR cluster and execution of the app on the cluster
6. Limits and perspectives

## 1. Structure of the projet

**This project articulates around 1 folder and 7 files:**

- training_data: folder containing a sample data for local test (3 images for each of three fruit categories)
- P8_local.ipynb: notebook containing the app run locally, main steps are listed below:
  - start of the spark session set-up to interact with aws s3 (NOTE: even if files are loaded and exported 
locally)
  - loading of the images
  - featurization 
  - pca
- cloud_app.ipynb: notebook containing the app run on the cluster, the main difference with P8_local is that it interacts
with s3 instead of the local drive
- environment.yml: file to set up dependencies with conda (local app)
- requirements.txt: file to set up dependencies with pip (local app)
- export_cli_emr.txt: command to create the EMR cluster with AWS CLI
- emr_p8_bootstrap.sh: bootstrap file to set up dependencies on the nodes of the cluster
- MyConfig.json: json file containing the configuration of spark and livy to be passed when creating the EMR cluster

## 2. Installation and set up of spark on Windows 10 with anaconda

<p align="center">
  <img width="511" src=?raw=true" />
</p>

## 3. Featurization using a pre-trained NN ResNet50

<p align="center">
  <img width="511" src=?raw=true" />
</p>

## 4. Dimension reduction using PCA

<p align="center">
  <img width="511" src=?raw=true" />
</p>

## 5. Set-up of the AWS EMR cluster and execution of the app on the cluster

<p align="center">
  <img width="511" src=?raw=true" />
</p>

## 6. Limits and perspectives

