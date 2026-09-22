function [Xtrain, ytrain, Xtest, ytest] = load_mnist()
%LOAD_MNIST Read the case-study CSVs and split pixels from labels.
%   Column 785 of each CSV holds the digit label; columns 1:784 hold the
%   vectorized 28x28 image. The repository root is one level above this file.

here = fileparts(mfilename('fullpath'));
root = fileparts(here);

train = readmatrix(fullfile(root,'mnist_train_1500.csv'));
test  = readmatrix(fullfile(root,'mnist_test_200.csv'));

Xtrain = train(:,1:784);
ytrain = train(:,785);
Xtest  = test(:,1:784);
ytest  = test(:,785);
end
