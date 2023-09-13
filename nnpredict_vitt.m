function [avcost_pred] = nnpredict_vitt(wih,xh,xpred,H,y_avcost,b)

    % inizialization
    n = length(xpred);
    xpred = (xpred-min(xpred))/(max(xpred)-min(xpred));
    up = [ones(n,1) xpred'];
    vstar = ones(n,H);
    v = ones(n,H);
    o = ones(length(xpred),1);

    % computation of vstar and v
    for h=1:H
        for p=1:length(xpred)
            vstar(p,h)=sum(wih(:,h).*(up(p,:))');
            v(p,h)=1/(1+exp(-vstar(p,h)));
        end
    end

    % computation of the output o
    for p=1:length(xpred)
        o(p)=b+sum(xh.*(v(p,:)'));
    end
    
    % inverse normalization of the output 
    avcost_pred = (max(y_avcost)-min(y_avcost))*o+min(y_avcost);   
   

end