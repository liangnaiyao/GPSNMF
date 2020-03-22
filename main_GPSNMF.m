function [Vr,Vw,testLabel] = main_GPSNMF( data,label,labPer,nctPer,mu,gamma,beta)
% object: %  1/p *||X^p-U^pV^p||2+beta(||W'Vlab-Y||2+gamma||W||21)+ mu*tr(V^pL^p(V^p)')
% s.t. X,U,V>=0.
% V=[V1,...V_P;Vc]=[[Vsl,Vsul];  Vlab=V(:,1:nl)

% other input: labPer —— The percentage of label available during training, i.e., 0.1:0.1:0.5
%              nctPer —— partially shared ratio       
% output: Vr ——  for GPSNMF^k ;  
%         Vw ——  for GPSNMF^w
%         testLabel —— true label for test

%% allocation
numOfTypes = size(data,2);     % 视角数
numOfDatas = size(data{1},2);  % 样本数
parameter.mu=mu;
parameter.labPer=labPer;
parameter.nctPer=nctPer;
parameter.gamma=gamma;
parameter.labPer=labPer;
parameter.beta=beta;
nl= round(labPer*numOfDatas); 

%% divide data into training set and testing set
    Cai=randperm(numOfDatas); % 打乱样本

    label=label(Cai);
    trainLabel = label(1:nl);
    testLabel = label(nl+1:end); 
    
    for view = 1:numOfTypes   
        data{view}=data{view}(:,Cai);  
    end
 %% Construct label matrix   
 YY=[];
for ii = unique(trainLabel) 
    YY = [YY;trainLabel==ii]; % 为列向量
end
%% Construct graph Laplacian
nSmp=size(data{1},2);
if mu>0
for v = 1:length(data)
    parameter.WeightMode='Binary';
    W{v}=constructW_cai(data{v}',parameter); 
 if mu > 0
    W{v} = mu*W{v};
    DCol = full(sum(W{v},2));
    D{v} = spdiags(DCol,0,nSmp,nSmp);
    L{v} = D{v} - W{v};
    if isfield(parameter,'NormW') && parameter.NormW
        D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
        L{v} = D_mhalf*L{v}*D_mhalf;
    end
 end
end
else
  for v = 1:length(data)   
    L{v} = 0;  
    D{v}=0;
    W{v}=0;
  end  
end
 parameter.L=L;   
  parameter.D=D;
 parameter.S=W;
%% do GPSNMF
[Vr, U, V, WW] = GPSNMF(data,YY,parameter);
%% clustering
Vw= WW'* Vr;   
end