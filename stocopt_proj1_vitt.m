clear all
close all
clc

% initialization
n=1000; % number of simulations
pgrid=[0.003, 0.00275, 0.0025, 0.00225, 0.002, 0.00175, 0.0015...
     0.00125, 0.001, 0.00075, 0.0005]; %probability of one to one infection

% inizialization of susceptible
Sn = zeros(60,1000); 
S0=999; 
Sn(1,:) = S0;

% inizialization of infected
In = zeros(60,1000);
I0=1;
In(1,:) = I0;

avcost=zeros(length(pgrid),1);
Hsim=zeros(100,1);
musim=zeros(100,1);
Hmax=zeros(100,1);
mumax=zeros(100,1);

for t=1:100
    
    % simulations
    for s=1:length(pgrid)
        for k=1:1000
            for i=1:59
                %In(i+1,k)=binornd(Sn(i,k),1-(1-pgrid(s))^(In(i,k)));
                %original version
                In(i+1,k)=binornd(Sn(i,k),1-(1-pgrid(1))^(In(i,k)));
                Sn(i+1,k)=Sn(i,k)-In(i+1,k);
            end
        end
        %avcost(s)=mean(sum(In))+(0.003/pgrid(s))^(9)-1;
        %original version
        avcost(s)=mean(sum(In))+(0.003/pgrid(1))^(9)-1;
    end

    % backpropagation + gridsearch for H and mu
    grid=zeros(6,10);
    i=1;
    j=1;
    for H = 3:2:13
        for mu = 0.01:0.01:0.1
            % H = 3;
            % mu = 0.05;
            [o,xh,wih,SSE,b] = nnbackprop_vitt(pgrid,log(avcost),mu,H);
            [o,log(avcost)];
            xpred = linspace(0.0005,0.003,1000);
            ypred=interp1(pgrid,log(avcost),xpred,'linear');
            [avcost_pred] = nnpredict_vitt(wih,xh,xpred,H,log(avcost),b);
            grid(i,j) = max(abs(ypred'-avcost_pred));
            j=j+1;
        end
        j=1;
        i=i+1;
    end
    
    mingrid=min(grid,[],'all','linear');
    [Hsim(t),musim(t)]=find(grid==mingrid);
    maxgrid=max(grid,[],'all','linear');
    [Hmax(t),mumax(t)]=find(grid==maxgrid);
    'min'
    [Hsim(t),musim(t)]
    'max'
    [Hmax(t),mumax(t)]
end
[Hsim, musim]
[Hmax, mumax] 
% Looking at [Hsim, musim] and [Hmax, mumax] we found that
% H=3, mu=0.06 is the best combination while H=9, mu=0.1
% is the worst one. 
% Here we report as a comment the functions we used to 
% make the plot of the report:

% plot(xpred,ypred,'r',xpred,avcost_pred1,'b');
% title('Plot of Neural Network prediction','FontSize',12);
% legend('log(avcost)','avcost pred');

% plot(pgrid,avcost,'-o')
% xlabel('Value of p')
% ylabel('Total average cost')
% title('Plot of total average cost')

% semilogx((1:length(SSE)),(SSE))
% axis([0 1000 -4 4])
% title('Plot of SSE','FontSize',12);