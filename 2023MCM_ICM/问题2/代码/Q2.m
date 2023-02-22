%% https://zhuanlan.zhihu.com/p/563996021
%% ׼������
clc;clear;close all;
load('data.mat')
%% ��������
%����ѵ�����ݺͲ�������
name=Xy7%%��������Xy1 Xy2 Xy3 Xy4 Xy5 Xy6 Xy7

[m,n]=size(name);
train_num=round(0.8*m); %�Ա��� 
x_train_data=name(1:train_num,1:n-1);
y_train_data=name(1:train_num,n);
%��������
x_test_data=name(train_num+1:end,1:n-1);
y_test_data=name(train_num+1:end,n);

x_train_data=x_train_data';
y_train_data=y_train_data';
x_test_data=x_test_data';
%% ��׼��
[x_train_regular,x_train_maxmin] = mapminmax(x_train_data);
[y_train_regular,y_train_maxmin] = mapminmax(y_train_data);
%% ��ʼ������
EMS_all=[];
TIME=[];
num_iter_all=5;
for NN=1:num_iter_all
input_num=size(x_train_data,1); %������������
hidden_num=6;   %���ز���Ԫ����
output_num=size(y_train_data,1); %�����������

% �Ŵ��㷨������ʼ��
iter_num=20;                         %���������������
group_num=10;                      %��Ⱥ��ģ
cross_pro=0.4;                       %�������
mutation_pro=0.05;                  %������ʣ������˵�Ƚ�С
%����Ż�����Ҫ˼������Ż���������ĳ�ʼѡ�񣬳�ʼѡ�����Ч���û����нϴ�Ӱ���
num_all=input_num*hidden_num+hidden_num+hidden_num*output_num+output_num;%�����ܲ�����ֻ��һ�����ز�
lenchrom=ones(1,num_all);  %��Ⱥ�ܳ���
limit=[-2*ones(num_all,1) 2*ones(num_all,1)];    %��ʼ����������Χ

  t1=clock;
%% ��ʼ����Ⱥ
input_data=x_train_regular;
output_data=y_train_regular;
for i=1:group_num
    initial=rand(1,length(lenchrom));  %����0-1�������
    initial_chrom(i,:)=limit(:,1)'+(limit(:,2)-limit(:,1))'.*initial; %���Ⱦɫ�����ʽ��һ��Ϊһ��Ⱦɫ��
    fitness_value=fitness1(initial_chrom(i,:),input_num,hidden_num,output_num,input_data,output_data);
    fitness_group(i)=fitness_value;
end
[bestfitness,bestindex]=min(fitness_group);
bestchrom=initial_chrom(bestindex,:);  %��õ�Ⱦɫ��
avgfitness=sum(fitness_group)/group_num; %Ⱦɫ���ƽ����Ӧ��                              
trace=[avgfitness bestfitness]; % ��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
%% ��������
input_chrom=initial_chrom;
% iter_num=1;
 for num=1:iter_num
    % ѡ��  
     [new_chrom,new_fitness]=select(input_chrom,fitness_group,group_num);   %�ѱ��ֺõ������������Ǻ���Ⱥ����һ��
%      avgfitness=sum(new_fitness)/group_num; 
    %����  
     new_chrom=Cross(cross_pro,lenchrom,new_chrom,group_num,limit);
    % ����  
     new_chrom=Mutation(mutation_pro,lenchrom,new_chrom,group_num,num,iter_num,limit);     
    % ������Ӧ��   
    for j=1:group_num  
        sgroup=new_chrom(j,:); %���� 
        new_fitness(j)=fitness1(sgroup,input_num,hidden_num,output_num,input_data,output_data);     
    end  
    %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
    [newbestfitness,newbestindex]=min(new_fitness);
    [worestfitness,worestindex]=max(new_fitness);
    % ������һ�ν�������õ�Ⱦɫ��
    if  bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=new_chrom(newbestindex,:);
    end
    new_chrom(worestindex,:)=bestchrom;
    new_fitness(worestindex)=bestfitness;
    avgfitness=sum(new_fitness)/group_num;
    trace=[trace;avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
 end
%%
figure(1)
[r ,~]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['Adaptability curves  ' 'Termination algebra��' num2str(iter_num)]);

xlabel('Evolutionary algebra');ylabel('Adaptability');
legend('Optimal adaptation');
 
%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
net=newff(x_train_regular,y_train_regular,hidden_num,{'tansig','purelin'});
w1=bestchrom(1:input_num*hidden_num);   %��������ز�֮���Ȩ�ز���
B1=bestchrom(input_num*hidden_num+1:input_num*hidden_num+hidden_num); %���ز���Ԫ��ƫ��
w2=bestchrom(input_num*hidden_num+hidden_num+1:input_num*hidden_num+...
    hidden_num+hidden_num*output_num);  %���ز�������֮���ƫ��
B2=bestchrom(input_num*hidden_num+hidden_num+hidden_num*output_num+1:input_num*hidden_num+...
    hidden_num+hidden_num*output_num+output_num); %�������Ԫ��ƫ��
%����Ȩֵ��ֵ
net.iw{1,1}=reshape(w1,hidden_num,input_num);
net.lw{2,1}=reshape(w2,output_num,hidden_num);
net.b{1}=reshape(B1,hidden_num,1);
net.b{2}=reshape(B2,output_num,1);
net.trainParam.epochs=200;          %����������
net.trainParam.lr=0.1;                        %ѧϰ��
net.trainParam.goal=0.00001;
[net,~]=train(net,x_train_regular,y_train_regular);

%���������ݹ�һ��
x_test_regular = mapminmax('apply',x_test_data,x_train_maxmin);
%���뵽�����������
y_test_regular=sim(net,x_test_regular);
%���õ������ݷ���һ���õ�Ԥ������
GA_BP_predict=mapminmax('reverse',y_test_regular,y_train_maxmin);
errors_nn=sum(abs(GA_BP_predict'-y_test_data)./(y_test_data))/length(y_test_data);
EcRMSE=sqrt(sum((errors_nn).^2)/length(errors_nn));
t2=clock;
Time_all=etime(t2,t1);

EMS_all=[EMS_all,EcRMSE];
TIME=[TIME,Time_all];
end
%%
figure(2)
% EMS_all=[0.149326909497551,0.142964890977319,0.145465721759172,0.144173052409406,0.155684223205026,0.142331921077465,0.144810383902860,0.144137917725977,0.149229175194219,0.143762158676095];
plot(EMS_all,'LineWidth',2)
xlabel('Number of experiments')
ylabel('Error')
hold on
figure(3)
color=[111,168,86;128,199,252;112,138,248;184,84,246]/255;
plot(y_test_data,'Color',color(2,:),'LineWidth',1)
hold on
plot(GA_BP_predict,'*','Color',color(1,:))
hold on
legend('Real data','Predicted data')
disp('The relative error is��')
disp(EcRMSE)
titlestr=['BP Neural Networks','   The error is��',num2str(min(EcRMSE))];
title(titlestr)

%% ����
x_test_regular = mapminmax('apply',EERIE,x_train_maxmin);
%���뵽�����������
y_test_regular=sim(net,x_test_regular);
%���õ������ݷ���һ���õ�Ԥ������
result=mapminmax('reverse',y_test_regular,y_train_maxmin);
disp('The predicted result is��')
disp(result)