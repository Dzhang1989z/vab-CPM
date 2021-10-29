clear all;
clc;
close all;

warning('off')

%% load DATA 
% MRI
load('rest_mats.mat');

% Bahavior data
load('behav_vec_TD.mat');

% removed the outliers by SD methods
    Q1=prctile(behav_vec,25);
    Q3=prctile(behav_vec,75);
    IQR = Q3-Q1;
    Qmax = Q3 + 1.5*IQR;
    Qmin = Q1 - 1.5*IQR;
    dropIdx1 = union(find(behav_vec>=Qmax), find(behav_vec<=Qmin));
    % using normal distribution
    tmp = (behav_vec - mean(behav_vec))./std(behav_vec);
    dropIdx2 = union(find(tmp>3), find(tmp<-3));
    dropIdx = union(dropIdx1, dropIdx2);
    
    behav_vec(dropIdx) = [];
    all_behav = behav_vec;  % behavior vector, 263x1
    
    %% 
    all_mats = rest_mats;   % FC matrix, d
    all_mats(:,:,dropIdx) = [];
    for pp = 1:size(all_mats,3)
        for qq = 1:size(all_mats,1)
            all_mats(qq,qq, pp) = 0;
        end   
    end
    all_mats = 0.5*log((1+all_mats)./(1-all_mats));

    % threshold for FC selection
    thresh = 0.01;

    %% Define parameters
    no_sub = size(all_mats,3);
    no_node = size(all_mats,1);

    behav_pred_pos = zeros(no_sub,1);
    behav_pred_neg = zeros(no_sub,1);

    pos_mat_save = zeros(no_node,no_node, no_sub);
    neg_mat_save = zeros(no_node,no_node, no_sub);

    whether_standard = 1;

    %% LOOCV
    for leftout = 1:no_sub 
        disp(['Leaving out subject #' num2str(leftout) '.']);
        % leave out one subject
        train_mats = all_mats;
        train_mats(:,:,leftout) = [];
        train_vecs = reshape(train_mats, [], size(train_mats, 3));   %% 71284x141
        test_mat = all_mats(:,:,leftout);

        %% adding standardization
        if whether_standard == 1
            mean_train = mean(train_vecs,2);   %% 71284x1
            std_train = std(train_vecs')';     %% 71284x1
            train_vecs = (train_vecs - repmat(mean_train,[1 size(train_vecs, 2)]))./ repmat(std_train, [1 size(train_vecs, 2)]);    
            train_vecs(isnan(train_vecs)) = 0;

            train_mats = reshape(train_vecs, [no_node no_node (no_sub-1)]);

            test_mat = (test_mat - reshape(mean_train, no_node, no_node))./(reshape(std_train, no_node, no_node));
            test_mat(isnan(test_mat)) = 0;
        end

        %%    
        train_behav = all_behav;
        train_behav(leftout) = [];

        % calculate the correlation between FC and behavior performance 
        edge_no = size(train_vecs,1);
        r_mat = zeros(1, edge_no);
        p_mat = zeros(1, edge_no);

        % method 1, Spearman rank correlation  
        [r_mat,p_mat] = corr(train_vecs', train_behav, 'type',  'Spearman');  %%  train_vecs = 25281x163,   train_behav=163x1,  r_mat, p_mat = 25281x1

        % selecting correlation-related edges based on correlation
        pos_mask = zeros(no_node, no_node);
        neg_mask = zeros(no_node, no_node);

        pos_edges = find(r_mat > 0 & p_mat < thresh);
        neg_edges = find(r_mat < 0 & p_mat < thresh);

        pos_mask(pos_edges) = 1;
        neg_mask(neg_edges) = 1;

        %%
        pos_mat_save(:,:,leftout) = pos_mask;
        neg_mat_save(:,:,leftout) = neg_mask;    

       %% calculate the FC sum of posives edges and negative edges
       %% sum value prediction
        train_sumpos = zeros(no_sub-1,1);
        train_sumneg = zeros(no_sub-1,1);

        for ss = 1:length(train_sumpos)  % n-1 subjects totally 
            train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask))/2;
            train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask))/2;
        end
        
        test_sumpos = sum(sum(test_mat.*pos_mask))/2;
        test_sumneg = sum(sum(test_mat.*neg_mask))/2;   

        % train linear regression model based on the FC sum feature
        fit_pos = polyfit(train_sumpos, train_behav, 1);
        fit_neg = polyfit(train_sumneg, train_behav, 1);

        % do the cross validation job     
        behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
        behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);      
    end
    disp('Pearson correlation:')
    [R_pos, P_pos] = corr(behav_pred_pos, all_behav)
    [R_neg, P_neg] = corr(behav_pred_neg, all_behav)
    
    save(['TD_pos_mat_NC001'], 'pos_mat_save');  
    save(['TD_neg_mat_NC001'], 'neg_mat_save');  

   %% plot of positive network prediction
    x = all_behav';
    y = behav_pred_pos';
   
    [h, p] = sort(x);
    x = x(p);
    y = y(p);

    t = figure(1);
    plot(x, y, 'bx');
    hold on;
    [p, s] = polyfit(x, y, 1);
    [yfit, dy] = polyconf(p, x, s, 'predopt', 'curve');
    plot(x, yfit, 'color', 'r');
    plot(x, yfit-dy, 'color', 'r','linestyle',':');
    plot(x, yfit+dy, 'color', 'r','linestyle',':');
    title({'Positive network', ['r = ' num2str(R_pos) ', p = ' num2str(P_pos)]}, 'FontSize', 12);
    
    %% plot of negative network prediction
    %% part 1
    x = all_behav';
    y = behav_pred_neg';

    [h, p] = sort(x);
    x = x(p);
    y = y(p);

    t = figure(2);
    plot(x, y, 'bx');
    hold on;
    [p, s] = polyfit(x, y, 1);
    [yfit, dy] = polyconf(p, x, s, 'predopt', 'curve');
    plot(x, yfit, 'color', 'r');
    plot(x, yfit-dy, 'color', 'r','linestyle',':');
    plot(x, yfit+dy, 'color', 'r','linestyle',':');
    title({'Negative network', ['r = ' num2str(R_neg) ', p = ' num2str(P_neg)]}, 'FontSize', 12);
    
%% 10000 times permutation test for positive- and negative network
% positive part
R_pos_array = [];
for p = 1:10000
    permIdx = randperm(length(all_behav));
    behav_vec_Perm = all_behav(permIdx);
    [R_pos_Perm, drop] = corr(behav_pred_pos, behav_vec_Perm);
    R_pos_array(p) = R_pos_Perm;
end
% 
R_pos_model = R_pos;
permP = length(find(R_pos_array>R_pos_model))/length(R_pos_array);
figure(4);
hist(R_pos_array, 24);
hold on;
plot([R_pos_model R_pos_model], [0 1400], 'r-');
xlabel('R value');
ylabel('Frequency');
title(['Permutation test p = ' num2str(permP)])

% negative part
R_neg_array = [];
for p = 1:10000
    permIdx = randperm(length(all_behav));
    behav_vec_Perm = all_behav(permIdx);
    [R_neg_Perm, drop] = corr(behav_pred_neg, behav_vec_Perm);
    R_neg_array(p) = R_neg_Perm;
end
% 
R_neg_model = R_neg;
permP = length(find(R_neg_array>R_neg_model))/length(R_neg_array);
figure(5);
hist(R_neg_array, 24);
hold on;
plot([R_neg_model R_neg_model], [0 1400], 'r-');
xlabel('R value');
ylabel('Frequency');
title(['Permutation test p = ' num2str(permP)])

