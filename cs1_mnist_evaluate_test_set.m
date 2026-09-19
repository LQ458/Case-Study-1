%% Evaluate a saved classifier or a model already present in the workspace.
% Standalone: loads classifierdata.mat and the provided test CSV.
% Competition: supply test (n x 784 or n x 785); optional correctlabels.
% The 785th column of supplied test is ignored, not treated as a true label.
eval_dir = fileparts(mfilename('fullpath'));
if ~exist('centroids','var') || ~exist('centroid_labels','var')
    model = load(fullfile(eval_dir,'classifierdata.mat'),'centroids','centroid_labels');
    centroids = model.centroids; centroid_labels = model.centroid_labels;
end
if ~exist('test','var')
    test_file = fullfile(eval_dir,'mnist_test_200.csv');
    if ~isfile(test_file), test_file = fullfile(eval_dir,'..','mnist_test_200.csv'); end
    raw_test = readmatrix(test_file);
    correctlabels = raw_test(:,785);
    test = [raw_test(:,1:784), zeros(size(raw_test,1),1)];
end
assert(size(test,2)>=784 && all(isfinite(test(:))), 'Invalid test pixels.');
assert(size(centroids,2)==785 && numel(centroid_labels)==size(centroids,1), 'Invalid model.');
assert(all(isfinite(centroids(:))) && all(ismember(centroid_labels,0:9)), 'Invalid centroid values or labels.');
centroid_labels = centroid_labels(:);
n_test = size(test,1);
predictions = zeros(n_test,1);
nearest_distances = zeros(n_test,1);
nearest_centroids = zeros(n_test,1);
for i = 1:n_test
    [prediction_index,vec_distance] = assign_vector_to_centroid(test(i,:),centroids);
    predictions(i) = centroid_labels(prediction_index);
    nearest_centroids(i) = prediction_index;
    nearest_distances(i) = vec_distance;
end
% Observable data-quality flag; not a claim to find all corrupted images.
% The stated gray scale is [0,255]. Flags never change the digit predictions.
outliers = double(any(test(:,1:784)<0 | test(:,1:784)>255,2));
if ~exist('evaluation_make_figures','var'), evaluation_make_figures = true; end
if ~exist('evaluation_verbose','var'), evaluation_verbose = true; end
accuracy = NaN; correct_count = NaN; confusion_counts = [];
if exist('correctlabels','var') && ~isempty(correctlabels)
    correctlabels = correctlabels(:);
    assert(numel(correctlabels)==n_test && all(ismember(correctlabels,0:9)), 'Invalid correctlabels.');
    correct_count = sum(correctlabels==predictions);
    accuracy = correct_count/n_test;
    confusion_counts = accumarray([correctlabels+1,predictions+1],1,[10,10]);
    if evaluation_verbose
        fprintf('Test accuracy: %d/%d = %.2f%%\n',correct_count,n_test,100*accuracy);
        fprintf('Pixel-range flags: %d/%d (all samples retained in accuracy).\n',sum(outliers),n_test);
    end
end
if evaluation_make_figures
    figure('Name','Outlier flags','Color','w');
    if exist('theme','file'), theme(gcf,'light'); end
    stem(1:n_test,outliers,'filled','MarkerSize',3); ylim([-0.1,1.2]);
    xlabel('Test image index'); ylabel('Pixel-range flag'); grid on;
    figure('Name','Figure 4: predictions','Color','w');
    if exist('theme','file'), theme(gcf,'light'); end
    if exist('correctlabels','var') && ~isempty(correctlabels)
        plot(1:n_test,correctlabels,'o','MarkerSize',4); hold on;
    end
    plot(1:n_test,predictions,'x','MarkerSize',4); hold off;
    xlabel('Test image index'); ylabel('Digit'); yticks(0:9); grid on;
    if ~isnan(accuracy)
        legend('True label','Prediction','Location','bestoutside');
        title(sprintf('Predictions: %d/%d correct (%.1f%%)',correct_count,n_test,100*accuracy));
    else
        title('Predicted digit labels');
    end
end

function [index,vec_distance] = assign_vector_to_centroid(data,centroids)
    % Direct difference avoids including the assignment column.
    distances = sqrt(sum((centroids(:,1:784)-data(1,1:784)).^2,2));
    [vec_distance,index] = min(distances);
end
