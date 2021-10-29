clear all;
clc;
close all;

warning('off')

%% load data 
load('rest_mats_ADHD.mat')
load('behav_vec_ADHDrs.mat');

%%
all_mats = rest_mats;   % FC matrix, d
for p = 1:size(all_mats,3)
    for q = 1:size(all_mats,1)
        all_mats(q,q, p) = 0;
    end   
end
all_mats = 0.5*log((1+all_mats)./(1-all_mats));
all_behav = behav_vec;  % behavior vector, 263x1

%% 
% threshold for FC selection
thresh = 0.01;

%% Define parameters
no_sub = size(all_mats,3);
no_node = size(all_mats,1);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);

whether_standard = 0;

% load positive- and negative- networks
load('TD_AA.mat');
load('TD_BB.mat');

%% LOOCV
for leftout = 1:no_sub 
    disp(['Leaving out subject #' num2str(leftout) '.']);
    % leave out one subject
    test_mat = all_mats(:,:,leftout);
    pos_mask = AA;
    neg_mask = BB;
    
    %% calculate the FC sum of posives edges and negative edges
     %% just mean value and correlation
        tmp = test_mat.*pos_mask;
        behav_pred_pos(leftout) = sum(tmp(:));  
        tmp = test_mat.*neg_mask;
        behav_pred_neg(leftout) = sum(tmp(:));  
end

%% Compare the predicted performance and ture behavior performance

disp('Pearson correlation:')
[R_pos, P_pos] = corr(behav_pred_pos, all_behav)
[R_neg, P_neg] = corr(behav_pred_neg, all_behav)

%% plot part
% positive part 
x = all_behav';
y = behav_pred_pos';

[h, p] = sort(x);
x = x(p);
y = y(p);

figure(1);
plot(x, y, 'bx');
hold on;
[p, s] = polyfit(x, y, 1);
[yfit, dy] = polyconf(p, x, s, 'predopt', 'curve');
plot(x, yfit, 'color', 'r');
plot(x, yfit-dy, 'color', 'r','linestyle',':');
plot(x, yfit+dy, 'color', 'r','linestyle',':');

ylabel('Network strength');
title({'High-TD network', ['r = ' num2str(R_pos) ', p = ' num2str(P_pos)]}, 'FontSize', 12);

% negative part 
x = all_behav';
y = behav_pred_neg';

[h, p] = sort(x);
x = x(p);
y = y(p);

figure(2);
plot(x, y, 'bx');
hold on;
[p, s] = polyfit(x, y, 1);
[yfit, dy] = polyconf(p, x, s, 'predopt', 'curve');
plot(x, yfit, 'color', 'r');
plot(x, yfit-dy, 'color', 'r','linestyle',':');
plot(x, yfit+dy, 'color', 'r','linestyle',':');

ylabel('Network strength');
title({'Low-TD network', ['r = ' num2str(R_neg) ', p = ' num2str(P_neg)]}, 'FontSize', 12);

%% Permutation test
% positive part 
R_pos_array = [];
for p = 1:10000
    permIdx = randperm(length(all_behav));
    behav_vec_Perm = all_behav(permIdx);
    [R_pos_Perm, drop] = corr(behav_pred_pos, behav_vec_Perm);
    R_pos_array(p) = R_pos_Perm;
end
R_pos_model = R_pos;
permP = length(find(R_pos_array>R_pos_model))/length(R_pos_array);
figure(3);
hist(R_pos_array, 24);
hold on;
plot([R_pos_model R_pos_model], [0 1400], 'r-');
xlabel('R value');
ylabel('Frequency');
title(['Permutation test p = ' num2str(permP)]);

% negative part 
R_neg_array = [];
for p = 1:10000
    permIdx = randperm(length(all_behav));
    behav_vec_Perm = all_behav(permIdx);
    [R_neg_Perm, drop] = corr(behav_pred_neg, behav_vec_Perm);
    R_neg_array(p) = R_neg_Perm;
end
R_neg_model = R_neg;
permP = length(find(R_neg_array>R_neg_model))/length(R_neg_array);
figure(4);
hist(R_neg_array, 24);
hold on;
plot([R_neg_model R_neg_model], [0 1400], 'r-');
xlabel('R value');
ylabel('Frequency');
title(['Permutation test p = ' num2str(permP)]);