close 
close all
clc
x1 = [1,2;2,3;3,3;4,5;5,5]'; %vanilla

x1 = [1,2;2,3;3,3;4,1;4,5;5,5]';
x2 = [1,0;2,1;3,1;3,2;5,3;6,5]';
figure
plot(x1(1,:),x1(2,:),'ro','MarkerSize',20)
hold on
plot(x2(1,:),x2(2,:),'ko','MarkerSize',20)
hold on
tn = [ones(length(x1),1);-ones(length(x2),1)]';
tn = load('q2tn.mat','tn');
tn = tn.tn;
x = [x1,x2];
x = load('q2x.mat','x');
x = x.x;
C = 10;
H = (tn'*tn).*(x'*x)+eye(length(x))*0.001;
% H = (tn'*tn).*(x'*x);
f = -ones(length(x),1);

Aineq = [eye(length(x));-eye(length(x))];
bineq = [C*ones(length(x),1);zeros(length(x),1)];

Aeq = [tn;zeros(length(x)-1,length(x))];
beq = zeros(length(x),1);

[lambda,FVAL,EXITFLAG] = quadprog(H,f,Aineq,bineq,Aeq,beq);
svm_index = find(round(lambda,2));
w = (lambda.*tn')'*x';

Ns = length(svm_index);
tn_n = tn(svm_index);
x_n = x(:,svm_index);
pos_svm = x_n(:,find(tn_n(tn_n>0)));
neg_svm = x_n(:,length(find(tn_n(tn_n>0)))+1:end);
lambda_n = lambda(svm_index,:);
b=0;
for i = 1:Ns
    i=1;
    b=b+(tn_n(:,i)-(lambda_n.*tn_n')'*x_n'*x_n(:,i));
end
b = b/Ns;
y = w*x+b;
class=sign(y);
true_class = tn;
% rr = [1,2,3,4,5,6,7,8,9,10,11];
% plot([1,6],[0,5])
w
b