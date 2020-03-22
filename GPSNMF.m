function [Vr, U, V, W, numIter,tElapsed,object] = GPSNMF(X, Y, option, U, V, W)

% X=[Xl,Xul]: nf*ns, ns = nl+nul
% U=[Us,Uc]: nf*nt, nt = nst+nct
% V=[Vs;Vc]=[[Vsl,Vsul];[Vcl,Vcul]]: nt*ns
% Vlab=[Vsl1;Vsl2;Vsl3;Vcl]: 3*nst+nct
% W: (nst*3+nct)*nc
% Y: nc*nl 

tStart=tic;

nv = length(X); % number of views

optionDefault.maxIter = 50;  % 最大迭代次数
optionDefault.minIter = 10; % 最小迭代次数
optionDefault.dis = 1;
optionDefault.epsilon=1e-3;
optionDefault.beta = 1000; %监督项正则参数
optionDefault.gamma = 10; %21范正则参数
optionDefault.alpha = 1/nv*ones(1,nv); %各个view权重
optionDefault.nctPer = 0.5; %共享topic比例
optionDefault.nwf = 100; %总共的topic ，即需要降维的秩数
optionDefault.isUNorm =1; % U是否要归一化
optionDefault.isXNorm =1; % X是否要归一化
optionDefault.isSqrt =1; %必要为1，V是否开根号更新
optionDefault.typeB = 1; %是否要更新W

if ~exist('option','var')
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);  % 混合两个结构体
end 

nwf = option.nwf;
beta = option.beta;
gamma = option.gamma;
alpha = option.alpha;
nctPer = option.nctPer;
isUNorm = option.isUNorm;
isXNorm = option.isXNorm;
maxIter = option.maxIter;
minIter = option.minIter;
epsilon = option.epsilon;
dis = option.dis;
isSqrt = option.isSqrt;
typeB = option.typeB;
mu=option.mu;
L=option.L;
DD=option.D;
S=option.S;

if isXNorm %对列(每个样本)归一化
    for i = 1:nv
        %平方和为1，似乎更好
        X{i} = normalize(X{i},0,1);
        %和为1
    % 	X{i} = normalize(X{i},0,0);
    end
end

nf = zeros(1,nv);

for i = 1:nv
    nf(i) = size(X{i},1); %number of i-view feature
end
[nc,nl] = size(Y);  % Y为行向量，nl的大小决定了样本中包含的标签样本的多少
ns = size(X{1},2);   % 得到样本数
nul = ns-nl; % number of unlabel samples，DICS中这项为0

nst = round(nwf/(nctPer/(1-nctPer)+nv)); % number of specific factor 
nct = nwf-nst*nv; % number of common factor, round为四舍五入
nt = nst+nct;

% nwf = nst*nv+nct; %叠加后的V的维数


if ~exist('V','var')
    Vlab=[];
    Vcl = rand(nct,nl);   
    Vcul = rand(nct,nul); 
    Vc = [Vcl,Vcul];       
    for i = 1:nv
        Vsl{i} = rand(nst,nl); 
        Vsul{i} = rand(nst,nul); 
        Vs{i} = [Vsl{i},Vsul{i}]; 

        Vl{i}=[Vsl{i};Vcl]; 
        Vul{i}=[Vsul{i};Vcul];

        V{i} = [Vs{i};Vc];  

        Vlab=[Vlab;Vsl{i}]; 
    end
else  
    Vc = V{1}(1+nst:end,:);
    Vcl = Vc(:,1:nl);  
    Vcul = Vc(:,1+nl:end);
    Vlab=[];
    for i = 1:nv
        Vs{i} = V{i}(1:nst,:);       
        Vsl{i} = Vs{i}(:,1:nl);
        Vsul{i} = Vs{i}(:,1+nl:end);

        Vl{i}=[Vsl{i};Vcl];
        Vul{i}=[Vsul{i};Vcul];
        Vlab=[Vlab;Vsl{i}];
    end
end
    Vlab=[Vlab;Vcl];

if ~exist('U','var')
    for i = 1:nv
    	U{i} = rand(nf(i),nt); 
        if isUNorm %对列归一化
            %和为1，似乎更好
            U{i} = normalize(U{i},0,0); % normalize(对象，row,type),归一化列为1
            % 平方和为1
            % U{i} = normalize(U{i},0,1);
        end
    end
end

if ~exist('W','var')
    D=eye(nwf);
else
    Wi = sqrt(sum(W.*W,2)+eps);
    d = 0.5./(Wi);
    D = diag(d);
    clear Wi d  
end

for i = 1:nv     % 将数据X、基U一起划分，U逗号左边是视角特别的特征、右边是共享特征，X逗号左边带标签的样本、右边是无标签的样本
    Xl{i} = X{i}(:,1:nl);
	Xul{i} = X{i}(:,(nl+1):end);

	Us{i} = U{i}(:,1:nst);  
	Uc{i} = U{i}(:,1+nst:end); 
	obj{i}=[];
end

objY=[];
object=[];


for j=1:maxIter
    
   	%% update U
    for i = 1:nv
        U{i} = U{i}.*(max(X{i}*V{i}',eps)./max(U{i}*(V{i}*V{i}'),eps));
        if isUNorm %对列归一化
            %和为1，似乎更好
            U{i} = normalize(U{i},0,0);
        %   %平方和为1
        % 	U{i} = normalize(U{i},0,1);
        end
        Us{i} = U{i}(:,1:nst);
        Uc{i} = U{i}(:,1+nst:end); 
    end
       
    %% update W
    %21范
    A=inv(Vlab*Vlab'+gamma*D+1e-4*eye(nwf)); 
    W = A*Vlab*Y';
    Wi = sqrt(sum(W.*W,2)+eps);
    d = 0.5./(Wi);
    D = diag(d);
    clear Wi d    
    %F范
%     A=inv(Vlab*Vlab'+gamma*eye(nwf)); 
%     W = A*Vlab*Y'; 
    
    %% update V
    
    if typeB ==1        
        Wz = (abs(W)+W)/2;
        Wf = (abs(W)-W)/2;
        WW = W*W';
        WWz = (abs(WW)+WW)/2;
        WWf = (abs(WW)-WW)/2;
        Bz = beta*(WWz*Vlab+Wf*Y);
        Bf = beta*(WWf*Vlab+Wz*Y);
    else
        %W不参与迭代
        B = -beta*W*Y;
        Bz = (abs(B)+B)/2;
        Bf = (abs(B)-B)/2;    
    end
    
%     B = beta*W*(W'*Vlab-Y);
%     Bz = (abs(B)+B)/2;
%     Bf = (abs(B)-B)/2;

%     WWV = beta*W*W'*Vlab;
%     WY = beta*W*Y;
%     WWVz = (abs(WWV)+WWV)/2;
%     WWVf = (abs(WWV)-WWV)/2;
%     WYz = (abs(WY)+WY)/2;
%     WYf = (abs(WY)-WY)/2;
%     Bz = WWVz+WYf;
%     Bf = WWVf+WYz;

    
    newVlab=[];
    sumUXl=zeros(nct,nl);
    sumUUVl=zeros(nct,nl);
    sumUXul=zeros(nct,nul);
    sumUUVul=zeros(nct,nul);   XU=0;  VUU=0;  XU1=0;  VUU1=0; 
    for i = 1:nv
        % 更新Vs
        Bzs = Bz(1:nst,:);
        Bz(1:nst,:)=[];
        Bfs = Bf(1:nst,:);
        Bf(1:nst,:)=[];
        
        if isSqrt
         if mu > 0  
            Vsl{i} = Vsl{i}.*sqrt(max(alpha(i)*Us{i}'*Xl{i}+Bfs+mu*Vs{i}*S{i}(:,1:nl),eps)...  % 
                ./max(alpha(i)*Us{i}'*U{i}*Vl{i}+Bzs+mu*Vs{i}*DD{i}(:,1:nl),eps));  % 
            Vsul{i} = Vsul{i}.*sqrt(max(alpha(i)*Us{i}'*Xul{i}+mu*Vs{i}*S{i}(:,1+nl:end),eps)...
                ./max(alpha(i)*Us{i}'*U{i}*Vul{i}+mu*Vs{i}*DD{i}(:,1+nl:end),eps));
         else      
            Vsl{i} = Vsl{i}.*sqrt(max(alpha(i)*Us{i}'*Xl{i}+Bfs,eps)...
                ./max(alpha(i)*Us{i}'*U{i}*Vl{i}+Bzs,eps));
            Vsul{i} = Vsul{i}.*sqrt(max(alpha(i)*Us{i}'*Xul{i},eps)...
                ./max(alpha(i)*Us{i}'*U{i}*Vul{i},eps));
         end   
        else
            Vsl{i} = Vsl{i}.*(max(alpha(i)*Us{i}'*Xl{i}+Bfs,eps)...
                ./max(alpha(i)*Us{i}'*U{i}*Vl{i}+Bzs,eps));
            Vsul{i} = Vsul{i}.*(max(alpha(i)*Us{i}'*Xul{i},eps)...
                ./max(alpha(i)*Us{i}'*U{i}*Vul{i},eps)); 
        end
        
        %-------
        if mu > 0
            WV = Vc*S{i}(:,1:nl);
            DV = Vc*DD{i}(:,1:nl);            
            XU = XU + WV;
            VUU = VUU + DV;
            
            WV1 = Vc*S{i}(:,1+nl:end);
            DV1 = Vc*DD{i}(:,1+nl:end);            
            XU1 = XU1 + WV1;
            VUU1 = VUU1 + DV1;            
        end   
        %-----
        
        Vs{i} = [Vsl{i},Vsul{i}];
        
        newVlab=[newVlab;Vsl{i}];
        
        % 计算更新Vc相关值
        sumUXl=sumUXl+alpha(i)*Uc{i}'*Xl{i};   % 计算共享系数的时候确实用到了共享基
        sumUUVl=sumUUVl+alpha(i)*Uc{i}'*U{i}*Vl{i};
        sumUXul=sumUXul+alpha(i)*Uc{i}'*Xul{i};
        sumUUVul=sumUUVul+alpha(i)*Uc{i}'*U{i}*Vul{i};
    end
    
    % 更新Vc
    Bzc = Bz;
    Bfc = Bf;
    
     if isSqrt
         if mu>0
                     Vcl = Vcl.*sqrt(max(sumUXl+Bfc+mu*XU,eps)./max(sumUUVl+Bzc+mu*VUU,eps));
         Vcul = Vcul.*sqrt(max(sumUXul+mu*XU1,eps)./max(sumUUVul+mu*VUU1,eps));
         else
        Vcl = Vcl.*sqrt(max(sumUXl+Bfc,eps)./max(sumUUVl+Bzc,eps));
         Vcul = Vcul.*sqrt(max(sumUXul,eps)./max(sumUUVul,eps));
         end
     else
        Vcl = Vcl.*(max(sumUXl+Bfc,eps)./max(sumUUVl+Bzc,eps));
         Vcul = Vcul.*(max(sumUXul,eps)./max(sumUUVul,eps));
     end
    
    Vc = [Vcl,Vcul]; 
    
    Vlab=[newVlab;Vcl];
        
     AAA=0;
    %% 计算目标函数   
    objRec(j) = 0;
    for i =1:nv
        V{i}=[Vs{i};Vc];
        Vl{i}=[Vsl{i};Vcl];
        Vul{i}=[Vsul{i};Vcul];
        obj{i}(j) = norm(X{i}-U{i}*V{i},'fro')^2;
        objRec(j) = objRec(j) + alpha(i)*obj{i}(j);   
        AAA=AAA+mu*trace(V{i}*L{i}*V{i}');
    end
    objY(j) = norm((W'*Vlab-Y),'fro')^2+gamma*trace(W'*D*W)+AAA;  % beta（β）、gamma（γ）、alpha（π），只有这3个参数，其中只有beta要调，文中说alpha自动学习的，这里设为定值
    object(j) =objRec(j)+beta*objY(j);
    
%     fprintf('epoch %3i  objRec %10.4f   objY %10.4i   object %10.4i\n', ...
%               j, objRec(j), objY(j),object(j)); 
          
    if mod(j,5)==0 || j==maxIter
        isStop = isIterStop(object, epsilon, j, maxIter, dis, minIter);
        if isStop
            break;            
        end
    end
end

Vr=[];
for i =1:nv
    Vr = [Vr;Vs{i}];
end
Vr = [Vr;Vc];   % 学了表示，但是没有用来聚类的代码,特征串联，列表示样本

numIter = j;
tElapsed=toc(tStart);
end