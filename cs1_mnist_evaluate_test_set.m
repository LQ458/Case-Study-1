%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

load('classifierdata.mat');

n_test=size(test,1);
predictions = zeros(n_test,1);
outliers = zeros(n_test,1);
distances = zeros(n_test,1);

% loop through the test set, figure out the predicted number
for i = 1:n_test

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

predictions(i) = centroid_labels(prediction_index);
distances(i) = vec_distance;

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
for i = 1:n_test
    if min(test(i,1:784))<0 || max(test(i,1:784))>255
        outliers(i)=1;
    end
end

%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(outliers,'filled','LineWidth',1.5);
xlabel('Test image');
ylabel('Outlier flag');
title('Test images with pixels outside 0 to 255');
ylim([0 1.2]);
grid on;
set(gca,'FontSize',12);

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o','LineWidth',1.5);
hold on;
plot(predictions,'x','LineWidth',1.5);
title('Predictions');
xlabel('Test image');
ylabel('Digit');
yticks(0:9);
legend('Correct label','Prediction','Location','northoutside','Orientation','horizontal');
grid on;
set(gca,'FontSize',12);

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

accuracy=sum(correctlabels==predictions)/n_test

outlier_count=sum(outliers)
outlier_correct=sum(correctlabels(outliers==1)==predictions(outliers==1))
median_distance_outliers=median(distances(outliers==1))
median_distance_others=median(distances(outliers==0))
[~,farthest]=sort(distances,'descend');
outliers_among_farthest=sum(outliers(farthest(1:outlier_count)))

confusion=zeros(10,10);
for i = 1:n_test
    confusion(correctlabels(i)+1,predictions(i)+1)=confusion(correctlabels(i)+1,predictions(i)+1)+1;
end
confusion_table=array2table(confusion,'RowNames',"true_"+string(0:9),'VariableNames',"predicted_"+string(0:9))

figure;
colormap('gray');
outlier_index=find(outliers==1);
wrong_index=find(predictions~=correctlabels & outliers==0);
for ind=1:min(4,length(outlier_index))
    subplot(2,4,ind);
    imagesc(reshape(test(outlier_index(ind),1:784),[28 28])',[0 255]);
    axis image off;
    title([num2str(correctlabels(outlier_index(ind))) ' as ' num2str(predictions(outlier_index(ind)))]);
end
for ind=1:min(4,length(wrong_index))
    subplot(2,4,ind+4);
    imagesc(reshape(test(wrong_index(ind),1:784),[28 28])',[0 255]);
    axis image off;
    title([num2str(correctlabels(wrong_index(ind))) ' as ' num2str(predictions(wrong_index(ind)))]);
end

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)

distances=zeros(size(centroids,1),1);
for j=1:size(centroids,1)
    distances(j)=norm(data(1:784)-centroids(j,1:784));
end
[vec_distance,index]=min(distances);

end
