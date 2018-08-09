# kaggle-dsb2-keras
Transforming How We Diagnose Heart Disease
Kaggle 2nd Annual Data Science Bowl submission that took 39th spot among 773 teams on Mar 2016:
Contest details: https://www.kaggle.com/c/second-annual-data-science-bowl
Data: https://www.kaggle.com/c/second-annual-data-science-bowl/data

Utilized a laptop with Intel i7-4710HQ CPU @ 2.50GHz(16GB) with Nvidia GTX 860M running Fedora 23
Since GPU heats up and also to utilize cross validation, run it as multiple times as shown in runMe.sh
Dependencies:
Python 2.7.10
numpy 
scipy
pydicom 0.9.9
scikit-image 0.11.3
Theano 0.8.0rc1
Keras 0.3.2


Download/Put Data Set under these files under directory called data as:
 ---- data (Download from Kaggle link above)
 |
 ---- sample_submission_validate.csv
 |
 ---- train.csv
 |
 ---- train
 |    |
 |    ---- 0
 |    |
 |    ---- …
 |
 ---- validate
      |
      ---- 501
      |
      ---- …


How to Run it all:
First process Images:
python mydata2.py

Then follow the training as described in runMe.sh
Finally create the prediction:
python submission.py

