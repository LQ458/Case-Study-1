%% Case Study 1: classify test images
% Ethan Wu
% Run after training, or load classifierdata.mat automatically below.
% For competition data, put test in the workspace before running this file.
if ~exist('centroids','var') || ~exist('centroid_labels','var')
    load('classifierdata.mat','centroids','centroid_labels');
end
if ~exist('test','var')
    test = readmatrix('mnist_test_200.csv');
    correctlabels = test(:,785);
    test(:,785) = 0;
end

%% Predict using only the pixel columns
n_test = size(test,1);
predictions = zeros(n_test,1);
for i = 1:n_test
    prediction_index = assign_vector_to_centroid(test(i,:),centroids);
    predictions(i) = centroid_labels(prediction_index);
end

%% Flag pixels outside the stated grayscale range
% Flagged images are still classified and included in the overall accuracy.
outliers = double(any(test(:,1:784)<0 | test(:,1:784)>255,2));
figure('Color','w');
if isprop(gcf,'Theme'), set(gcf,'Theme','light'); end
stem(1:n_test,outliers,'filled');
xlabel('Test image index'); ylabel('Outlier flag');
title('Pixels outside [0,255]'); ylim([-0.1 1.2]); grid on;

%% Figure 4: true and predicted labels
figure('Color','w');
if isprop(gcf,'Theme'), set(gcf,'Theme','light'); end
if exist('correctlabels','var') && numel(correctlabels)==n_test
    correctlabels = correctlabels(:);
    correct_count = sum(predictions==correctlabels);
    accuracy = correct_count/n_test;
    fprintf('Correct: %d/%d; accuracy: %.2f%%\n', ...
        correct_count,n_test,100*accuracy);
    plot(1:n_test,correctlabels,'o'); hold on;
    plot(1:n_test,predictions,'x'); hold off;
    legend('True label','Prediction','Location','best');
    title(sprintf('Predictions: %d/%d correct',correct_count,n_test));
else
    plot(1:n_test,predictions,'x');
    title('Predicted digit labels');
end
xlabel('Test image index'); ylabel('Digit'); yticks(0:9); grid on;

function [index,vec_distance] = assign_vector_to_centroid(data,centroids)
    d = zeros(size(centroids,1),1);
    for j = 1:size(centroids,1)
        d(j) = norm(data(1:784)-centroids(j,1:784));
    end
    [vec_distance,index] = min(d);
end
