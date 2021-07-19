
%%%% Train on all University of Glasgow data and Test on NG Homes Location
%%%% 3 


clc
clear all
close all


imds = imageDatastore('C:\Users\ss414j\Desktop\Train on All UoG Data Test on NG Homes Data\Train\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8);





test = imageDatastore('C:\Users\ss414j\Desktop\Train on All UoG Data Test on NG Homes Data\Test on Location 3', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

    


net = googlenet;


lgraph = layerGraph(net);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)


net.Layers(1)
inputSize = net.Layers(1).InputSize;


lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);


lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])


layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);



pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);


augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


test_data = augmentedImageDatastore(inputSize(1:2),test);



options = trainingOptions('sgdm', ...
    'MiniBatchSize',5, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,test_data);

accuracy = mean(YPred == test.Labels)


y1=test.Labels;


plotconfusion(y1',YPred','Test Accuracy')





Conf_Mat_Test = confusionmat(YPred,test.Labels)  % Confusion Matrix for Train Data 
plotconfusion(YPred,test.Labels)


% 
% 
% Conf_Mat_Test = confusionmat(predictedLabels,testSet.Labels)  % Confusion Matrix for Train Data 
% 
% overal_confusion_matrix=Conf_Mat_Training+Conf_Mat_Test
% 
% Accuracy_Training = 100*sum(diag(Conf_Mat_Training))./sum(Conf_Mat_Training(:));  % Accuracy of the Training Data 
% Accuracy_Test = 100*sum(diag(Conf_Mat_Test))./sum(Conf_Mat_Test(:))  ;         %%  Accuracy of the Test Data 
% Accuracy_of_the_System=100*sum(diag(overal_confusion_matrix))./sum(overal_confusion_matrix(:))  %%  Overal Data  
% 
% 
% overal_confusion_matrix=Conf_Mat_Training+Conf_Mat_Test;
% 
% 
% 
% 
% 
% 
