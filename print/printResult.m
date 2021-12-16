function [ac1, nmi1, Pri1,AR1,F1,P1,R1,ac2, nmi2, Pri2,AR2,F2,P2,R2] = printResult(X, label, K, kmeansFlag)
	if(~exist('kmeansFlag','var'))
		kmeansFlag = 1;
	end
    for i=1:1
        if kmeansFlag == 1
            indic = litekmeans(X, K, 'Replicates',20);
            result = bestMap(label, indic);
        else
            [~, result] = max(X, [] ,2);
        end
        if kmeansFlag == 1
            [ac(i), nmi_value(i), cnt(i)] = CalcMetrics(label, result);
        else
            [ac(i), nmi_value(i), cnt(i)] = CalcMetrics0(label, result);
        end
%         [ac(i), nmi_value(i), cnt(i)] = CalcMetrics(label, indic);
        [Pri(i)] = purity(label, result);
        AR(i)=RandIndex(label, result);
        [F(i),P(i),R(i)] = compute_f(label, result);
    end
    ac1 = mean(ac); nmi1=mean(nmi_value);
    Pri1=mean(Pri); AR1=mean(AR);
    F1=mean(F);P1=mean(P);R1=mean(R);
    ac2 = std(ac); nmi2=std(nmi_value);
    Pri2=std(Pri); AR2=std(AR);
    F2=std(F);P2=std(P);R2=std(R);    
    fprintf('ac: %0.2f\tnmi:%0.2f\tpur:%0.2f\tar:%0.2f\tf_sc:%0.2f\tpre:%0.2f\trec:%0.2f\n', ac1*100, nmi1*100, Pri1*100,AR1*100,F1*100,P1*100,R1*100);
%     fprintf('ac: %0.2f\tnmi:%0.2f\tpur:%0.2f\tar:%0.2f\tf_sc:%0.2f\tpre:%0.2f\trec:%0.2f\n', ac2*100, nmi2*100, Pri2*100,AR2*100,F2*100,P2*100,R2*100);
end
