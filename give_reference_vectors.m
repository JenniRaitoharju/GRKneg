function M_ref = give_reference_vectors(Traindata, Negdata, refoption)
%Give reference vectors according to the selected reference option
%
%Input:
%Traindata --> 'D x N' matrix of (positive) training vectors
%Negdata --> 'D x Nneg' matrix of negative training vectors
%refoption --> 1-8, selected GRK with negatives approach
%
%Output:
%M_ref --> 'D x M' matrix of reference vectors, where M depends on the selected reference option

[D,N] = size(Traindata); 
Nneg = size(Negdata,2);
switch refoption
    case 1 % P positive training samples
        M_ref = Traindata;  
    case 2 % P positive and N negative training samples
        M_ref = [Traindata, Negdata];
    case 3  % N negative training samples
        M_ref = Negdata;
    case 4 % P positive and N negative generated samples
        Nneg = size(Negdata,2);
        avr = mean(Negdata')';
        var = std(Negdata')';
        Posrand = randn(D, N);
        Negrand = avr+var.*randn(D, Nneg);
        M_ref = [Posrand, Negrand];
    case 5 % P + N negative generated samples
        Nneg = size(Negdata,2);
        avr = mean(Negdata')';
        var = std(Negdata')';
        Negrand = avr+var.*randn(D, N+Nneg);
        M_ref = [Negrand];
    case 6 % P non-positive and N negative generated samples 
        Nneg = size(Negdata,2);
        avr = mean(Negdata')';
        var = std(Negdata')';
        Negrand = avr+var.*randn(D, Nneg);
        Posrand = randn(D, N);
        Posrand = Posrand + 0.5*sign(Posrand);
        M_ref = [Posrand, Negrand];
    case 7 % P negative generated samples and N negative training samples  
        Nneg = size(Traindata,2);
        avr = mean(Negdata')';
        var = std(Negdata')';
        Negrand = avr+var.*randn(D, Nneg);
        M_ref = [Negdata, Negrand];
    case 8 % 2P negative generated samples and N negative training samples
        Nneg = size(Traindata,2);
        avr = mean(Negdata')';
        var = std(Negdata')';
        Negrand = avr+var.*randn(D, Nneg*2);
        M_ref = [Negdata, Negrand];
    case 9 % P/2 negative generated samples and N negative training samples
        Nneg = size(Traindata,2);
        avr = mean(Negdata')';
        var = std(Negdata')';
        Negrand = avr+var.*randn(D, round(Nneg/2));
        M_ref = [Negdata, Negrand];   
end
 