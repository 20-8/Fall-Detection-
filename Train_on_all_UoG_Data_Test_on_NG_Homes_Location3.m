%% This code is to train the classifier when Alexnet was used to extract features and trained those features with KNN and test was performed....
%%... on NG Homes data collected in february 2019 at three different
%%locations-------------- The Test is Performed by Selecting One person by
%%at a time.


clc
clear all
clc




convnet = alexnet;
convnet.Layers % Take a look at the layers

rootFolder = 'C:\Users\ss414j\Desktop\Train on All UoG Data Test on NG Homes Data\Train\';
categories = {'a','b','c','d','e'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
% imds.ReadFcn = @readFunctionTrain;
% Change the number 50 to as many training images as you would like to use
% how does increasing the number of images change the 
% accuracy of the classifier?
[trainingSet, ~] = splitEachLabel(imds, 113, 'randomize');  %% 92 is the number of images in one class

featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer);

YTrain=trainingSet.Labels;

sizOF_TraininFeatures=size(trainingFeatures);

len=length(YTrain); %%% Total number of observations


X=reshape(trainingFeatures, 14*21*4096, len);  %%% len is the total number of observations is 300 

Y=(trainingSet.Labels)';
YY=Y';
XX=X';

Mdl = fitcecoc(XX,YY)

prediction_Train = predict(Mdl,XX);                %% Preparing for  confusion matrix for test dataset

Conf_Mat_Training = confusionmat(prediction_Train,YY)  % Confusion Matrix for Train Data 



 

 %% Call test files
 rootFolder = 'C:\Users\ss414j\Desktop\Train on All UoG Data Test on NG Homes Data\Test on Location 3\';
testSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
     

testFeatures = activations(convnet, testSet, featureLayer);

[a1 b1 c1 d1] = size(testFeatures)

testFeatures=reshape(testFeatures, a1*b1*c1, d1);  % 48 is the total number of observations


predictedLabels = predict(Mdl, testFeatures');

Confusion_Matrix = confusionmat(predictedLabels,testSet.Labels);  % Confusion Matrix for Train Data 



Accuracy_Test  = 100*sum(diag(Confusion_Matrix)./sum(Confusion_Matrix(:)))            %%  Accuracy of the Test Data




% plotconfusion(predictedLabels,testSet.Labels)


 


 
 
 
 