function f = new_figure()
%NEW_FIGURE Figure styled exactly like the ones the skeleton scripts make.
%   Default MATLAB figure size, white background, forced light theme. Every
%   analysis figure goes through here so the report's plots are visually
%   consistent with Figures 1-4 produced by cs1_mnist_base_skeleton.m and
%   cs1_mnist_evaluate_test_set.m.

f = figure('Color','w');
if isprop(f,'Theme'), set(f,'Theme','light'); end
end
