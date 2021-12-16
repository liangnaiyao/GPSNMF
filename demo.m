%This is where you can start. 
clear all;
addpath('graph');
addpath('misc');
addpath('print');
%% parameters setting
parameter.maxIter = 100;  
parameter.minIter = 10; 
gui=1;
labPer=0.5;  % label rate
mu=0.01;     %  graph
gamma=1;    % gamma controls the weight of l2,1-norm regular item ,PSLF = 100
nctPer=0.6;  % common factor ratio
beta=0.01;   % contribution of the label learning
%% load data
load MSRCv1
%% normalize data
for v = 1:length(data)
    A = mapminmax(data{v},0,1); 
    data{v} = A; 
end    
rand('seed',2);
for i=1:10
    %% run
    [Vr,Vw,testLabel] = main_GPSNMF(data,label,labPer,nctPer,mu,gamma,beta,parameter);
    %% get results
    fprintf('GPSNMF^k  ');
    printResult(Vr(:,end-length(testLabel)+1:end)', testLabel', numel(unique(testLabel)), 1);
    fprintf('GPSNMF^w  ');
    printResult(Vw(:,end-length(testLabel)+1:end)', testLabel', numel(unique(testLabel)), 0);
end



