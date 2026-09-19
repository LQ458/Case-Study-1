clear;
close all;

k = 80; % Change k here to compare different numbers of clusters.
max_iter = 100;
rng(1050); % Repeatable random initialization.

%% Read the data
train = readmatrix('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train(:,785) = 0;
test = readmatrix('mnist_test_200.csv');
correctlabels = test(:,785);
test(:,785) = 0;

%% Assign images, update centroids, and record cost
centroids = initialize_centroids(train,k);
cost_iteration = zeros(max_iter,1);
for iter = 1:max_iter
    old_assignments = train(:,785);
    for i = 1:size(train,1)
        train(i,785) = assign_vector_to_centroid(train(i,:),centroids);
    end
    centroids = update_Centroids(train,k,centroids);
    residuals = train(:,1:784) - centroids(train(:,785),1:784);
    cost_iteration(iter) = sum(residuals(:).^2);
    if isequal(train(:,785),old_assignments)
        break;
    end
end
cost_iteration = cost_iteration(1:iter);

%% Assign each centroid the most common training label in its cluster
centroid_labels = zeros(k,1);
for j = 1:k
    members = train(:,785)==j;
    if any(members)
        centroid_labels(j) = mode(trainsetlabels(members));
    else
        % If a cluster is empty, use the nearest training image's label.
        d = sum((train(:,1:784)-centroids(j,1:784)).^2,2);
        [~,nearest] = min(d);
        centroid_labels(j) = trainsetlabels(nearest);
    end
end
save('classifierdata.mat','centroids','centroid_labels');

%% Figure 1: k-means cost
figure('Color','w');
if isprop(gcf,'Theme'), set(gcf,'Theme','light'); end
plot(1:iter,cost_iteration,'-o','LineWidth',1.5);
xlabel('Iteration'); ylabel('K-means cost (WCSS)');
title(sprintf('Training cost, k = %d',k)); grid on;

%% Figure 2: centroids and their digit labels
figure('Color','w');
if isprop(gcf,'Theme'), set(gcf,'Theme','light'); end
plotsize = ceil(sqrt(k));
for j = 1:k
    subplot(plotsize,plotsize,j);
    imagesc(reshape(centroids(j,1:784),[28 28])',[0 255]);
    axis image off;
    title(sprintf('%d: %d',j,centroid_labels(j)));
end
colormap gray;

function y = initialize_centroids(data,num_centroids)
    random_index = randperm(size(data,1));
    y = data(random_index(1:num_centroids),:);
end

function [index,vec_distance] = assign_vector_to_centroid(data,centroids)
    d = zeros(size(centroids,1),1);
    for j = 1:size(centroids,1)
        d(j) = norm(data(1:784)-centroids(j,1:784));
    end
    [vec_distance,index] = min(d);
end

function new_centroids = update_Centroids(data,K,previous_centroids)
    % Retain the previous position when a cluster has no members.
    new_centroids = previous_centroids;
    for j = 1:K
        members = data(:,785)==j;
        if any(members)
            new_centroids(j,1:784) = mean(data(members,1:784),1);
        end
    end
end
