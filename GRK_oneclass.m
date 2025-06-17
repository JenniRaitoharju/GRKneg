function labels = GRK_oneclass(Traindata, Testdata, Negdata, Testlabels, basekernel, useGRK, refoption, kernelparam, classifparam )
%Generalized Reference Kernel for One-Class Classification Support Vector
%Machine
%
%Input:
%Traindata --> 'D x N' matrix of (positive) training vectors 
%Testdata --> 'D x Ntest' matrix of test vectors 
%Negdata --> 'D x Nneg' matrix of negative training vectors 
%basekernel --> function pointer to the basekernel function
%useGRK --> boolean, GRK (true) or original method (false)
%refoption --> 1-9, selected GRKneg approach
%kernelparam --> hyperparameter for kernel (e.g., sigma for RBF)
%classifparam --> hyperparamer for classificaion (Nu for OCSVM)
%
%Output:
%labels --> 'Ntest x 1' vector of predicted labels for the test samples 

%Get the kernel matrix using the selected generalized reference kernel approach
if ~useGRK %KOriginal kernel
    [Ktrain, Ktest, Ktest_self] = basekernel(Traindata, Testdata, kernelparam);
elseif useGRK %Kernel with GRK
    [Ktrain, Ktest, Ktest_self] = reference_kernel(Traindata, Testdata, basekernel, kernelparam, refoption, Negdata);
end

N = size(Traindata,2);
Ntest = size(Testdata,2);


%Train
Ktrain_svm =  [ (1:size(Ktrain,2))' , Ktrain' ];
Ktest_svm =  [ (1:size(Ktest,2))' , Ktest' ];
svm_model = svmtrain(ones(size(Ktrain,1),1), Ktrain_svm, sprintf('-t 4 -q -s 2 -n %f', classifparam));
%Test
labels = svmpredict(Testlabels, Ktest_svm, svm_model, '-q');
