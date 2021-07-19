

clear all
clc
close all


BaseName='P';
for k=18:36
FileName_without1{k-17,1}=[BaseName,num2str(k)]
end

FileName_P08='P08';

% Confusion_Matrix=cell(length(FileName), 1);

FileName=[FileName_P08 ;FileName_without1];

 
    
 


for tt=1 :length(FileName)
    
    
    
    UoG = imageDatastore('H:\Ancortek Data matlab Processing\All Spectrogram Pictures\Train on All UoG Data Test on NG Homes Data\Train\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

Cumbria= imageDatastore('H:\Ancortek Data matlab Processing\All Spectrogram Pictures\West Cumbria Old Age Home\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

save_cumbria_files=Cumbria.Files;
save_cumbria_labels=Cumbria.Labels;

Cumbria.Files=[save_cumbria_files;UoG.Files];
Cumbria.Labels=[save_cumbria_labels;UoG.Labels];

[imdsTrain,imdsValidation] = splitEachLabel(Cumbria,0.8);



testSet = imageDatastore('H:\Ancortek Data matlab Processing\All Spectrogram Pictures\Train on All UoG Data Test on NG Homes Data\Test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

    


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
 
 
 %%%%%%%%










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


test_data = augmentedImageDatastore(inputSize(1:2),testSet);



options = trainingOptions('sgdm', ...
    'MiniBatchSize',5, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,test_data);

accuracy(tt,1) = mean(YPred == testSet.Labels)


y1=testSet.Labels;


plotconfusion(y1',YPred','Test Accuracy')





    Conf_Mat_Test{tt,1} = confusionmat(YPred,testSet.Labels)  % Confusion Matrix for Train Data 
plotconfusion(YPred,testSet.Labels)

    
    
    
     
        
    
    
    
    
    
    
    
    
    
end
