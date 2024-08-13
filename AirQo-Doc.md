# How to set up folders and where each file is saved 

create a folder named data
at the same file level of the notebook, name the training data file "Train.csv" name the testing data file "Test.csv"
model predictions' file will be created at the same file level of the notebook named "submission.csv".

data is available at : https://zindi.africa/competitions/airqo-african-air-quality-prediction-challenge/data

so the file tree will look as follows:
    main.ipynb (file)
    data(folder)--|
                  |-Train.csv(file)
                  |-Test.csv(file)


# Order in which to run code 
install the required libraries, and from top to buttom  

# Explanations of features used 
we used time related features, and post processing in the form of taking the mean of the target and grouping by the day of the year.
# Environment for the code to be run (conda environment.yml file or an environment.txt file) 
enivorment.txt file can be found here <https://drive.google.com/file/d/1Fk_GDzvkpzaLNt3vb-qaDFC52v7TBtQv/view?usp=sharing>
# Hardware used (e.g. Google Colab or the specifications of your local machine) 
Python 3.10.7
OS: Ubuntu 22.10 x86_64
Kernel: 5.19.0-46-generic 
CPU: 11th Gen Intel i5-11400H (12) @ 4.500GHz
Memory: 4706MiB / 7667MiB

# Expected run time for each notebook. This will be useful to the review team for time and resource allocation. 
- the run time for the notebok is as following: 
Run Time for the Notebook = 13.68726921081543 second
Run TIme for the model = 2.93 s ± 457 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
