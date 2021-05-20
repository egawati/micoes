# Micro-clusters-based Outlier Explanations for Data Streams

Outlier or anomaly detection has been used in many applications, including data streams applications; however, it is up to the users to interpret the detected anomalous objects. Outlier explanation is an essential part of the outlier analysis as it gives clues that can help users understand the outliers better and thus can speed up the decision process. One type of explanations that can be provided to the users is identifying the attributes or features responsible for the abnormality of the data point, which are known as the outlying attributes. In this project, we propose a micro-clusters-based outlier explanation framework for data streams (MICOES).

MICOES finds outlying attributes for outliers detected by an arbitrary anomaly detection algorithm. Data streams bring extra challenges because they have an unbounded volume of data that keep coming such that the processing is limited to time and memory constraints. Therefore, for real-time processing, it is not possible to have multiple passes on data. MICOES addresses this nature of data streams by utilizing the statistical synopsis of the data maintained in the micro-clusters to find outlying attributes.

### Package Installation - Development Mode

`python3 setup.py develop`

