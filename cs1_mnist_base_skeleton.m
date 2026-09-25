
clear all;
close all;

%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:

  % 1. initialize_centroids
  % This function sets the initial values of the centroids
  
  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance
  
  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.


%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);

% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images

%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image

figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(1,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

k=400; % set k
max_iter=50; % set the number of iterations of the algorithm

rng(1050);
[centroids,train,cost_iteration]=run_kmeans(train,k,max_iter);

centroid_labels=label_centroids(train,trainsetlabels,k);
save('classifierdata.mat','centroids','centroid_labels');

%% This section of code plots the k-means cost as a function of the number
% of iterations

figure;
plot(cost_iteration,'-o','LineWidth',2);
xlabel('Iteration');
ylabel('K-means cost');
title('K-means cost vs. iteration');
grid on;
set(gca,'FontSize',12);


%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.

figure;
colormap('gray');

plotsize = ceil(sqrt(k));
[~,order]=sort(centroid_labels);

for ind=1:k
    
    centr=centroids(order(ind),[1:784]);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    axis image off;

end
sgtitle('Centroids sorted by digit label');

k_values=[10 20 50 100 200 300 400 500 1000];
train_accuracy=zeros(length(k_values),1);
test_accuracy=zeros(length(k_values),1);
final_cost=zeros(length(k_values),1);
digits_with_centroid=zeros(length(k_values),1);

for n=1:length(k_values)
    rng(1050);
    [k_centroids,k_train,k_cost]=run_kmeans(train,k_values(n),max_iter);
    k_labels=label_centroids(k_train,trainsetlabels,k_values(n));
    train_accuracy(n)=mean(k_labels(k_train(:,785))==trainsetlabels);
    k_predictions=zeros(size(test,1),1);
    for i=1:size(test,1)
        k_predictions(i)=k_labels(assign_vector_to_centroid(test(i,:),k_centroids));
    end
    test_accuracy(n)=mean(k_predictions==correctlabels);
    final_cost(n)=k_cost(end);
    digits_with_centroid(n)=length(unique(k_labels));
end

k_results=table(k_values',100*train_accuracy,100*test_accuracy,final_cost,digits_with_centroid, ...
    'VariableNames',{'k','train_accuracy','test_accuracy','cost','digits_with_centroid'})

digit_average=zeros(10,784);
own_average_distance=zeros(10,1);
for d=0:9
    members=train(trainsetlabels==d,1:784);
    digit_average(d+1,:)=mean(members,1);
    for i=1:size(members,1)
        own_average_distance(d+1)=own_average_distance(d+1)+norm(members(i,:)-digit_average(d+1,:))/size(members,1);
    end
end

average_distance=zeros(10,10);
for a=1:10
    for b=1:10
        average_distance(a,b)=norm(digit_average(a,:)-digit_average(b,:));
    end
end

digit_names=string(0:9);
average_distance_table=array2table(round([average_distance own_average_distance]), ...
    'RowNames',digit_names,'VariableNames',[digit_names "own_digit"])

function [centroids,data,cost_iteration]=run_kmeans(data,k,max_iter)

data(:,785)=zeros(size(data,1),1);

%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

centroids=initialize_centroids(data,k);

%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm

for iter=1:max_iter
    
    previous_assignment=data(:,785);
    for i=1:size(data,1)
        [index, vec_distance]=assign_vector_to_centroid(data(i,:),centroids);
        data(i,785)=index;
        cost_iteration(iter)=cost_iteration(iter)+vec_distance^2;
    end
    centroids=update_Centroids(data,k,centroids);
    if isequal(data(:,785),previous_assignment)
        break;
    end
    
end
cost_iteration=cost_iteration(1:iter);

end

%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).

function y=initialize_centroids(data,num_centroids)

random_index=randperm(size(data,1));

centroids=data(random_index(1:num_centroids),:);

y=centroids;

end

%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)

distances=zeros(size(centroids,1),1);
for j=1:size(centroids,1)
    distances(j)=norm(data(1:784)-centroids(j,1:784));
end
[vec_distance,index]=min(distances);

end


%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images, the value of k, and the
% previous centroids.
% It returns a new set of centroids based on the current assignment of the
% training images.

function new_centroids=update_Centroids(data,K,previous_centroids)

new_centroids=previous_centroids;
for j=1:K
    members=data(:,785)==j;
    if any(members)
        new_centroids(j,1:784)=mean(data(members,1:784),1);
    end
end

end

function labels=label_centroids(data,trainsetlabels,K)

labels=zeros(K,1);
for j=1:K
    labels(j)=mode(trainsetlabels(data(:,785)==j));
end

end
