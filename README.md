# Pycaret-Unsupervised-Anomaly-Detection

PyCaret is an open-source, machine learning library in Python that helps you from data preparation to model deployment. It is easy to use and you can do almost every data science project task with just one line of code.

This project is an experimentation with the Anomaly Detection module of the PyCaret library. It is useful for unsupervised machine learning that is used for identifying rare items or events.
Some examples of Anomaly detection are bank fraud, structural defect, medical problems etc.

The dataset I am using is rather odd in the sense that it does not have specified columns. It consists of mostly categorical fields.

### Steps followed to normalize dataset : 
1. Rename columns into something understandable
2. Supposedly sensor data, we are dropping some fields that do not contribute to the process.
3. Using *Cardinal Encoding* instead of One Hot Encoding
   - Here the values have many levels, hence One Hot Encoding would lead to creation of large number of features. Hence, we are using another encoding technique, built into PyCaret called, "Feature Encoding" More details [https://pycaret.org/cardinal-encoding/](here).

4. Handling Missing values and normalize dataset.

## For Anomaly Detection : 
We are using the following techniques:  
1. Isolation Forests
2. K Nearest Neighbours
3. Clustering technique


Isolation Forests performs better than most, read more here : 
https://en.wikipedia.org/wiki/Isolation_forest

