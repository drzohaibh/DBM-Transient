c = 'case1';                     % exchange c in ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7'];
                               
%% Select different cases >>>> Transient characteristic calculation
load('Bonn.mat');                % Download data
state = 'transient';
switch c
    case 'case1'
        trainx(101:400,:) = []; 
        epoch_n = [100; 50; 5; 2];
    case 'case2'
        trainx = [trainx(101:200,:); trainx(401:500,:)];
        epoch_n = [100; 50; 5; 5];
    case 'case3'
        trainx = [trainx(201:300,:); trainx(401:500,:)];
        epoch_n = [100; 50; 5; 4];
    case 'case4'
        trainx(1:300,:) = [];
        epoch_n = [200; 20; 5; 5];
    case 'case5'
        trainx(101:200,:) = []; 
        epoch_n = [200; 20; 5; 5];
    case 'case6'
        trainx(1:100,:) = [];
        epoch_n = [200; 20; 5; 5];
    case 'case7'
        trainx(:,13:14) = [];
        epoch_n = [200; 20; 5; 5];
end

RBM1;
RBM2;
RBM3;
for out_d = 2:10
    RBM4;
    save(['Different_dimension\', c, '_', state, '_', num2str(out_d), 'd.mat'],'new_data2')
end

%% converged characteristic calculation
load('Bonn.mat'); 
state = 'converged';
switch c
    case 'case1'
        trainx(101:400,:) = []; 
        epoch_n = [100; 50; 50; 50];
    case 'case2'
        trainx = [trainx(101:200,:); trainx(401:500,:)];
        epoch_n = [100; 50; 50; 50];
    case 'case3'
        trainx = [trainx(201:300,:); trainx(401:500,:)];
        epoch_n = [100; 50; 50; 40];
    case 'case4'
        trainx(1:300,:) = [];
        epoch_n = [200; 50; 50; 50];
    case 'case5'
        trainx(101:200,:) = []; 
        epoch_n = [200; 50; 50; 50];
    case 'case6'
        trainx(1:100,:) = [];
        epoch_n = [200; 50; 50; 50];
    case 'case7'
        trainx(:,13:14) = [];
        epoch_n = [200; 50; 50; 50];
end
RBM1;
RBM2;
RBM3;
for out_d = 2:10
    RBM4;
    save(['Different_dimension\', c, '_', state, '_', num2str(out_d), 'd.mat'],'new_data2')
end