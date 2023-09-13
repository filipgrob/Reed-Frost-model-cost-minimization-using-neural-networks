function [o,xh,wih,SSE,b] = nnbackprop_vitt(xp,y_avcost,mu,H)

% AGGIUNGI DESCRIZIONE FUNZIONE 

    % step1: initialization
    I = 2; % number of input nodes
    m_max = 10000; % max number of iterations
    n = length(xp);
    tol = 10^(-7);
    
    % error sum of square
    SSE = 100*ones(m_max,1); 
    SSE(2) = 101;
    
    wih = 0.5*rand(I,H); 
    xh = 0.5*rand(H,1);
    b = unifrnd(0,0.5);
    
    xp_norm = (xp-min(xp))/(max(xp)-min(xp)); % normalized input
    up = [ones(n,1) xp_norm'];
    
    vstar = ones(n,H);
    v = ones(n,H);
    o = ones(n,1);
    
    y_avcost_norm = (y_avcost-min(y_avcost))/(max(y_avcost)-min(y_avcost)); % normalized y_avcost

    
    m = 0;
    k = 2;
    while abs(SSE(k)-SSE(k-1)) > tol && m <= m_max
        
        % step2-3: computation of v* and v
        for h=1:H
            for p=1:n
                vstar(p,h)=sum(wih(:,h).*(up(p,:))');
                v(p,h)=1/(1+exp(-vstar(p,h)));
            end
        end

        % step4: computation of the output o
        for p = 1:n
            o(p) = b+sum(xh.*(v(p,:)'));
        end

        % step5: update of wih and xh
        b = b+mu*sum(y_avcost_norm-o);
        for i = 1:I
            for h = 1:H
                wih(i,h) = wih(i,h)+mu*sum((y_avcost_norm-o)*xh(h).*v(:,h).*(1-v(:,h)).*up(:,i));
            end
        end
        for h = 1:H
            xh(h) = xh(h)+mu*sum((y_avcost_norm-o).*v(:,h));
        end

        % step6: computation of SSE
        m = m+1;
        k = k+1;
        SSE(k) = sum((y_avcost_norm-o).^2);
    end
        % step7: inverse normalization of the output
        o = (max(y_avcost)-min(y_avcost))*o+min(y_avcost); 

end