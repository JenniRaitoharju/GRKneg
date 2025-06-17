%This files contains a simple demo for applying the Generalized Reference
%Kernel with Negative Samples for OCSVM on already preprocessed datasets

%Add libraries
addpath('libSVMmex');

%Add datasets
addpath('Datasets');

%Load a preprocessed dataset and select data split
%Note that training data includes also negative samples
dataset = 'Datasets/iris_targetclass_1'; % Preprocessed dataset (See 'Datasets/AboutData.txt')
datasplit = 1; %Each dataset has 5 different splits into train/test, select 1-5
load (dataset); 
Traindata=traindata5sets{1, datasplit};
Trainlabels=trainlabels5sets{1, datasplit};
Testdata=testdata5sets{1, datasplit};
Testlabels=testlabels5sets{1, datasplit};

%Select only positive data for training
Negdata=Traindata(:, Trainlabels==-1); %Negative training data is needed for reference kernel options 5-7
Traindata=Traindata(:, Trainlabels==1);

%Keep only a fraction of the negative train data
negN = 10;
Negdata = Negdata(:, randperm(size(Negdata, 2)));
negN = min( length(Negdata), negN );
if negN == -1
    negN = length(Negdata);
end
Negdata = Negdata(:, 1:negN); 

%Define experimental setup and set hyperparameter values
%See 'GRK_oneclass.m' for parameter definitions and
%'give_reference_vectors.m' for GRK variant definitions
useGRK = true; 
refoption = 7; % Proposed approach 
basekernel = @kernel_rbf;
sigma = 1;
c = 0.01; 

%Run experiment
labels = GRK_oneclass( Traindata, Testdata, Negdata, Testlabels, basekernel, useGRK, refoption, sigma, c );

%Evaluate performance metrics
results = evaluate(Testlabels,labels)


