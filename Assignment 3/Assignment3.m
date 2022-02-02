clc;
clear;
close all;
%% Reading Data

data= xlsread('Data2.xlsx');
x1 = (data(1:23,1:2))';
x2 = (data(24:55,1:2))';

y = (data(:,3))';
x = [x1,x2];
n=numel(y);
ClassA=find(y==1);
ClassB=find(y==-1);
scr = zeros(1);
%% MODIFIED OPTIMIZATION PROBLEM
for c = 1:1:10
    C=c;
    H=zeros(n,n);  

    for i=1:n
        for j=i:n
            H(i,j)=y(i)*y(j)*x(:,i)'*x(:,j);
            H(j,i)=H(i,j);
        end
    end

    f=-ones(n,1);
    %EQUALITY CONSTRAINTS
    Aeq=y;
    beq=0;

    %SETTIG THE BOUNDS
    %lb = lower bound
    %ub = upper bouund
    lb=zeros(n,1);
    ub=C*ones(n,1);

    %SETTING THE OPTIMIZER OPTIONS LIKE NUMBER OF ITERATIONS, ETC.
    options = optimset('Display', 'off','LargeScale', 'off','MaxIter',100);

    %INVOKING THE QUADPROG SOLVER
    alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub,[])';

    AlmostZero=(abs(alpha)<max(abs(alpha))/1e5);

    alpha(AlmostZero)=0;
    S=find(alpha>0 & alpha<C);
    w=0;

    %GETTING THE BIAS OF DECISION BOUNDARY
    for i=S
        w=w+alpha(i)*y(i)*x(:,i);
    end
    b=mean(y(S)-w'*x(:,S));
    %% VISULAIZING RESULTS
    Line=@(x1,x2) w(1)*x1+w(2)*x2+b;
    LineA=@(x1,x2) w(1)*x1+w(2)*x2+b+1;
    LineB=@(x1,x2) w(1)*x1+w(2)*x2+b-1;
    figure;
    plot(x(1,ClassA),x(2,ClassA),'ro');
    hold on;
    plot(x(1,ClassB),x(2,ClassB),'bs');
    plot(x(1,S),x(2,S),'ko','MarkerSize',12);
    x1min=min(x(1,:));
    x1max=max(x(1,:));
    x2min=min(x(2,:));
    x2max=max(x(2,:));
    handle=ezplot(Line,[x1min x1max x2min x2max]);
    set(handle,'Color','k','LineWidth',2);
    handleA=ezplot(LineA,[x1min x1max x2min x2max]);
    set(handleA,'Color','k','LineWidth',1,'LineStyle',':');
    handleB=ezplot(LineB,[x1min x1max x2min x2max]);
    set(handleB,'Color','k','LineWidth',1,'LineStyle',':');
    legend('Class A','Class B');

    %% Accuracy plots
    true=w'*x+b;
    class=sign(true);
    true_class = y;
    % rr = linspace(1,50,50)';
    rr = linspace(1,55,55)';
    figure
    scatter(rr,true_class,'O')
    hold on
    scatter(rr,class,'*')
    legend({'True Class','Predicted Class'},'FontSize',12)
    %% SCORING
    nr = 0; %initializing number of correctly classified points
    dr = 55;%tota; number of points
    for i=1:55
        if class(i)==true_class(i)
            nr = nr + 1;
        end
    end
    %accuracy or score
    score = nr/dr
    scr(c)=score;
close
clc
end
cx = 1:1:10;
scatter(cx,scr)
% close
% clc
