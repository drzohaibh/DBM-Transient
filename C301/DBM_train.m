load('c301_x.mat')
non_n = 9992;
sei_n = 757;

%% Transient characteristic calculation
state = 'transient';
epoch_n = [50; 1; 1; 1];
RBM1;
RBM2;
RBM3;
for output_d = 2:10
    RBM4;
    save(['Different_dimension\', 'DBM_', state, '_', num2str(output_d), 'd.mat'],'new_data2')
end

%% converged characteristic calculation
state = 'converged';
epoch_n = [50; 50; 50; 50];
RBM1;
RBM2;
RBM3;
for output_d = 2:10
    RBM4;
    save(['Different_dimension\', 'DBM_', state, '_', num2str(output_d), 'd.mat'],'new_data2')
end