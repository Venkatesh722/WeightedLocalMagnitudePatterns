# WeightedLocalMagnitudePatterns
WeightedLocalMagnitudePatterns for recoginition of (morphed images || faceswapped images)

In this project I developed a working code based on IEEE paper which is on Digital Face Presentation Attack Detection via Weighted Local Magnitude Patterns.Developed a model using SVM which gives me a accuracy of about 93 % .

# Dataset Used
Dataset Name :- Large-scale CelebFaces Attributes(CelebA) Dataset


Online Link to dataset:- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


Type of Landmarks used: 81 landmark points

#Running the project

First run the program in the command line using the command python recognize.py --training images/training --testing images/testing

Then after building the pickle.model run the predict.py file to see the results


