# Featurization and dimension reduction of a dataset of images using Spark and AWS EMR

This project aims at creating a spark application to:
1) vectorize a dataset of images using transfer learning from a pretrained neural network
2) reduce the dimensions of the vectorized images using PCA
3) be run locally and then be run on a cluster of three nodes (AWS EMR)

The dataset comes from the Kaggle dataset [fruits](https://www.kaggle.com/moltean/fruits)

## Table of content

1. Structure of the project
2. Installation and set up of spark on Windows 10 with anaconda
3. Featurization using a pre-trained NN ResNet50
4. Dimension reduction using PCA
5. Set-up of the AWS EMR cluster and execution of the app on the cluster
6. Limits and perspectives

## 1. Structure of the projet

**This project articulates around 1 folder and 7 files:**

- **training_data**: folder containing a sample data for local test (3 images for each of three fruit categories)
- **P8_local.ipynb**: notebook containing the app run locally, main steps are listed below:
  - start of the spark session set-up to interact with aws s3 (NOTE: even if files are loaded and exported 
locally)
  - loading of the images
  - featurization 
  - pca
- **cloud_app.ipynb**: notebook containing the app run on the cluster, the main difference with P8_local is that it interacts
with s3 instead of the local drive
- **environment.yml**: file to set up dependencies with conda (local app)
- **requirements.txt**: file to set up dependencies with pip (local app)
- **export_cli_emr.txt**: command to create the EMR cluster with AWS CLI
- **emr_p8_bootstrap.sh**: bootstrap file to set up dependencies on the nodes of the cluster (to be passed when creating 
the EMR cluster)
- **MyConfig.json**: json file containing the configuration of spark and livy (to be passed when creating the EMR cluster)

## 2. Installation and set up of spark on Windows 10 with anaconda

sources : 

•	https://tech.supertran.net/2020/06/pyspark-anaconda-jupyter-windows.html

•	https://phoenixnap.com/kb/install-spark-on-windows-10

•	https://stackoverflow.com/questions/51728177/can-pyspark-work-without-spark

1) install java: as these instruction are given, spark supports no later than version 1.8:
Install version: jdk1.8.0_301
https://www.techspot.com/downloads/5198-java-jre.html


2) In anaconda, download pyspark 3.1.2 . This module includes a full installation of spark (it comes with Hadoop jars of Hadoop 3.2.0 located in C:\Users\VP\anaconda3\envs\env_p8_3_7\Lib\site-packages\pyspark\jars)


3) Set environment variable for java and spark

    SPARK_HOME => C:\Users\VP\anaconda3\envs\env_p8_3_7\Lib\site-packages\pyspark

    JAVA_HOME => C:\Program Files\Java\jdk1.8.0_301


2) In order to run some operations of Spark on windows such as saving to a paquet file, you would need Hadoop winutils.exe file as windows don’t support HDFS and winutils provides a wrapper
   * create the following folder “Hadoop” and subfolder “bin”
‘C:\Hadoop\bin’
   * Download the corresponding winutils.exe and Hadoop.ddl in the folder of the right Hadoop version from the following repo
https://github.com/cdarlint/winutils
   
     https://github.com/cdarlint/winutils/tree/master/hadoop-3.2.0/bin

     and place it in C:\Hadoop\bin
   
   * Set the environment variable HADOOP_HOME as C:\Hadoop 
   * Add C:\Hadoop\bin to PATH

     N.B: According to some posts, deleting haddop.ddl from Hadoop/bin may resolve some issues
https://stackoverflow.com/a/51681347/14668029


3) Pyspark in anaconda does not come with all the jars that spark might need. Especially those needed to connect with S3:
These two missing jars for our spark and Hadoop versions are:
   * aws-java-sdk-bundle-1.11.375.jar that assures the integration between java and AWS 
   * hadoop-aws-3.2.0.jar that assures the integration between Hadoop and AWS

    these two jars can be found respectively at:

   https://mvnrepository.com/artifact/com.amazonaws/aws-java-sdk

   https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-aws

   download these two jars and place them in C:\Users\VP\anaconda3\envs\<your_env>\Lib\site-packages\pyspark\jars


4) In your activated command prompt (jupyterlab in our case), run the following command to launch a session and connect to s3 with spark:

```
import findspark
findspark.init()
findspark.find()
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local') \
                            .appName('p8') \
                            .config('spark.hadoop.fs.s3a.access.key', 'your_access_key_for_this_aws_account') \
                            .config('spark.hadoop.fs.s3a.secret.key', 'your_secret_key_for_this_aws_account'') \
                            .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') \
                            .getOrCreate()
                            
sc = spark.sparkContext
sc.setSystemProperty('com.amazonaws.services.s3.enableV4', 'true')
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com") # be sure this is the same region as your S3 bucket

# setting up S3
client = boto3.client('s3', region_name='us-east-2') # region_name parameter should be optional

# test connection between spark and s3
s3_url = "s3a://s3-p8/input_data/*"
image_df = spark.read.format("binaryfile").load(s3_url)
image_df.printSchema()
```

## 3. Featurization using a pre-trained NN ResNet50

source: https://docs.databricks.com/applications/machine-learning/preprocess-data/transfer-learning-tensorflow.html

The approach is very well explained in the source:

>1) Start with a pre-trained deep learning model, in this case an image classification model from tensorflow.keras.applications.
>2) Truncate the last layer(s) of the model. The modified model produces a tensor of features as output, rather than a prediction.
>3) Apply that model to a new image dataset from a different problem domain, computing features for the images.
>4) Use these features to train a new model. The following notebook omits this final step

Process detailed for an image:
1) the image is loaded as bytearray: `bytearray(b'\xff\xd8\xff\xe0\x00..` , length = 57284
2) the image is preprocessed to be fed to RestNet50: 
 `array([[[151.061  , 138.22101, 131.32...` , shape = (224, 224, 3) (RGB format)
3) the image is processed by RestNet50 and outputs a vector as a list: `[0.0, 0.0, 2.884536027908, 0.0,...`, length 100352

<p align="center">
  <img width="511" src=?raw=true" />
</p>

## 4. Dimension reduction using PCA

This step is quite straightforward, the code is identical as the scikit-learn API. The only extra step is 
to convert the list we extracted from the transfer learning to Vectors.dense format that spark PCA needs as an input.
For a sample of nine images, after a few tries for the optimal K value, we end up with a reduced vector as list of 
length 7 that explains more than 99% of variations.

## 5. Set-up of the AWS EMR cluster and execution of the app on the cluster

### a. Set up of the cluster

Once the local app has been tested, we need to configure a cluster on AWS EMR to run the app on a distributed way 
on a much larger sample.

First a quick reminder about what [AWS EMR](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html) is:
>Amazon EMR (previously called Amazon Elastic MapReduce) is a managed cluster platform that simplifies running big data
> frameworks, such as Apache Hadoop and Apache Spark, on AWS to process and analyze vast amounts of data. Using
> these frameworks and related open-source projects, you can process data for analytics purposes and business 
> intelligence workloads. Amazon EMR also lets you transform and move large amounts of data into and out of
> other AWS data stores and databases, such as Amazon Simple Storage Service (Amazon S3)

Note: the distributed file system used is S3 File system, not HDFS.

**Configuration of the cluster through the AWS console:**

Consider that all elements not mentioned are left with default values
1) **Software and steps:**
- Software Configuration:

EMR release: 6.4.0

unselect Hue, Hive and PIG

select Spark, JupyterEnterpriseGateway and Livy (necessary to use EMR notebooks)

- Edit software settings:

select "Load JSON from S3" and write the path to MyConfig.json in an accessible s3 repo.
This file contains values to modify some default properties of the software installed in the previous step.
The properties modified are the following:

[maximizeResourceAllocation](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-configure.html#emr-spark-maximizeresourceallocation) -> True: let AWS optimize spark properties to maximise resource allocation

[spark.sql.files.maxRecordsPerFile](https://stackoverflow.com/questions/61837678/job-65-cancelled-because-sparkcontext-was-shut-down) -> "10000": limit the maximum number of records to write out to a single file. May prevent spark
from crashing during the .write step.

[livy.server.session.timeout](https://stackoverflow.com/questions/54220381/how-to-set-livy-server-session-timeout-on-emr-cluster-boostrap) -> "5H": prevents the session of shutting down after 1H (default value) of execution for a task 

3) **General cluster settings:**
- bootstrap actions

add a custom action, write the path the bootstrap file on an accessible s3 repo

The boostrap file will execute on each node and install the packages listed inside

content of emr_p8_bootstrap.sh:
```
#!/bin/bash -xe
sudo pip3 install numpy --upgrade --ignore-installed
sudo pip3 install pandas==0.25.1
sudo pip3 install Pillow
sudo pip3 install pyarrow
sudo pip3 install boto3
sudo pip3 install s3fs
```
**Note**: as this README is writen, it is impossible to update numpy on the EMR nodes, which forces us to use an older version of 
pandas for compatibility. Besides, bootstrap files generally include `sudo yum update` to update packages of the linux distribution
on the node. In our case, this instruction would not complete, so the instruction was removed and the bootstrap action completed.

4) **Security**

Select the EC2 key pair that you will use in your spark application

### b. Execution of the spark application on the cluster

input data: 100 images (size: 451 KB)

Completion time: 18 minutes

## 6. Limits and perspectives

 #### a. Size of featurized image

The PCA takes 15 minutes to complete. One approach to make it faster could be reducing the size of the input vectors (100352).
More precisely, we could find a pre-trained CNN that featurizes images in shorter vectors, while loosing marginal information
compared with RestNet50.

### b. Tuning of the cluster

The set-up of spark parameters influencing performance were automatically adjusted using the AWS property 
`maximizeResourceAllocation`. Perhaps, this configuration is not the best one for our case, and it would be worthwhile handtuning
spark properties.
