# covid-19

It is an attempt to predict the covid-19 pandemic outbreak for all states in US and the worldwide. 

## Prepare

#### 1) Clone the repository:

    git clone https://github.com/lhuang-pvamu/covid-19.git


#### 2) Create a conda environment:

If you don't have Anaconda, you need to download Python 3.x from https://www.anaconda.com/distribution/#download-section.

In the main repository directory, use the following commands to create the enviroment.: 

    conda env create -f environment.yml
    conda activate covid-19


#### 3) Get the data:

We use the Johns Hopkins github dataset for the work. Go to the Data directory and clone the dataset:

    cd Data
    git clone https://github.com/CSSEGISandData/COVID-19.git

      
#### 4) Run the prediction:

##### 4.1 go to the Analysis folder:

    cd Analysis
    
##### 4.2 Run the prediction for worldwide:

    python process.py -w
    
##### 4.2 Run the prediction for US:

    python process.py
    
#### 5) Check the prediction:

All prediction figures are saved into the Results folder. Rename it after each run so that they won't be overwritten. 


      
