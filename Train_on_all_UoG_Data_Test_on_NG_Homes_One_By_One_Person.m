%% This code is to train the classifier when Alexnet was used to extract features and trained those features with KNN and test was performed....
%%... on NG Homes data collected in february 2019 at three different
%%locations-------------- The Test is Performed by Selecting One person by
%%at a time.


clc
clear all
clc


BaseName='P';
for k=37:49
FileName_without1{k-17,1}=[BaseName,num2str(k)]
end

FileName_P08='P08';

% Confusion_Matrix=cell(length(FileName), 1);

FileName=[FileName_P08 ;FileName_without1];



 for tt=1         :length(FileName)
     
     
     
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

% data_and_labels=[XX YY];
% 
Mdl = fitcknn(XX,YY)

% Mdl = fitcdiscr(XX,YY)



prediction_Train = predict(Mdl,XX);                %% Preparing for  confusion matrix for test dataset

Conf_Mat_Training = confusionmat(prediction_Train,YY)  % Confusion Matrix for Train Data 




%% Testing Code
 



%%  Get P18 to P36 in one cell arrayy  --------- Each Personal Files and store in an array --- Get it automatically




 %% Get Particular Volunteer Files only 
 

 %% Call test files
     rootFolder = 'C:\Users\ss414j\Desktop\Train on All UoG Data Test on NG Homes Data\Test\';
testSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
     %%
     
     s1= FileName{tt,1}    %% Get Files for Specific Person , say P018, P19 and so on up until P36

for t=1        :length(testSet.Files)  %% total number of files available
    
myPath = testSet.Files{t, 1} ; %% specfic file being called
C = strsplit(myPath,'MD of ')  ; %% a character is given where the string is divided
CC{t,1}=C{1,2};


tf(t,1) = strncmp(s1,CC{t, 1},3); %% string compare if it equial to s1, will save 1 otherwise 0. logical array stored. 


end



get_specific_file_indices = tf;  %% indices for specific file being called 

Single_Person_Files=testSet.Files(tf,1);
Single_Person_labels=testSet.Labels(tf,1);

 testSet.Files= Single_Person_Files;
 testSet.Labels=Single_Person_labels;
 
 
%%

testFeatures = activations(convnet, testSet, featureLayer);

[a1 b1 c1 d1] = size(testFeatures)

testFeatures=reshape(testFeatures, a1*b1*c1, d1);  % 48 is the total number of observations


predictedLabels = predict(Mdl, testFeatures');

Confusion_Matrix{tt,1} = confusionmat(predictedLabels,testSet.Labels);  % Confusion Matrix for Train Data 



Accuracy_Test(tt,1) = 100*sum(diag(Confusion_Matrix{tt,1}))./sum(Confusion_Matrix{tt,1}(:))            %%  Accuracy of the Test Data 




% plotconfusion(predictedLabels,testSet.Labels)


clearvars -except Confusion_Matrix Accuracy_Test FileName

 end




 
 
  


% x=(18:36)

figure
bar(Accuracy_Test)
 
 
 

