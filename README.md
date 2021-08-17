# Micro-clusters-based Outlier Explanations for Data Streams

Outlier or anomaly detection has been used in many applications including data streams applications; however, it is up to the users to decide what to do with the detected anomalous objects. Outlier explanation is an important part of the outlier analysis as it gives clues that can help user to understand the outliers better and thus can speed up the decision process. One type of explanations that can be provided to the users is which attributes or features that responsible for the abnormality of the data point. It is also known as the outlying attributes. 

In this project , we present a micro-clusters-based outlier explanation framework for data streams (MICOES). It finds outlying attributes for outliers detected by an arbitrary anomaly detection algorithm. Data streams bring extra challenges because they have an unbounded volume of data that keep coming such that the processing is limited to time and memory constraints. Therefore, for real-time processing, it is not possible to have multiple passes on data. MICOES addresses this nature of data streams by utilizing the statistical synopsis of the data maintained in the micro-clusters to find outlying attributes. 

The MICOES paper is available in [the 1st Workshop on Anomaly and Novelty Detection, Explanation and Accommodation (ANDEA) in Conjunction with KDD 2021](https://sites.google.com/view/andea2021/accepted-papers)

### Package Installation - Development Mode
- Download and extract the the ZIP file
- On the terminal cd to extracted micoes folder and run the command: `python3 setup.py develop`

### Citation
```
@misc{MICOES2021,
title= {Micro-clusters-based Outlier Explanations for Data Streams},
author = {Egawati Panjei and Le Gruenwald and Eleazar Leal and Christopher Nguyen},
year = {2021},
booktitle= {The 1st Workshop on Anomaly and Novelty Detection, Explanation and Accommodation (ANDEA) in Conjunction with KDD 2021},
URL= {https://sites.google.com/view/andea2021/accepted-papers},
}
```
