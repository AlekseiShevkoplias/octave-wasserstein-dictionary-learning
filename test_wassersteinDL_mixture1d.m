%% Wasserstein NMF on histograms representation of mixtures of gaussians
% This script reproduces the experiment from the original paper
% Modified to automatically save all plots in the 'figures' directory

mkdir('figures');

%% Generate the data

ndata = 100;
niidsample = 1000;
ndiscretize = 100;

minVal = -12;
maxVal = 12;
x = linspace(minVal, maxVal, ndiscretize);

mu = [-6; 0; 6];
shiftVariance = 2 * eye(3);
mean = (randn(ndata, size(mu, 1)) * chol(shiftVariance))' + mu;
p = rand(3, ndata);
sigma = 1;

data = zeros(ndiscretize, ndata);
for i = 1:ndata
    weights = p(:, i) / sum(p(:, i));
    sample_counts = round(weights * niidsample);
    total_samples = sum(sample_counts);
    a = zeros(total_samples, 1);
    idx = 1;
    for j = 1:3
        cnt = sample_counts(j);
        mu_j = mean(j, i);
        a(idx:idx + cnt - 1) = randn(cnt, 1) * sqrt(sigma) + mu_j;
        idx = idx + cnt;
    end
    valid = (a > minVal) & (a < maxVal);
    data(:, i) = hist(a(valid), x)';
end

data = bsxfun(@rdivide, data, sum(data));

%% Visualize data samples
minY = 0;
maxY = 0.1;
YtickStep = 0.02;
indices = 1:3;
fontSize = 30;
lineWidth = 4;
axisValues = [minVal, maxVal, minY, maxY];

dataLegendArray = cell(numel(indices), 1);
for i = 1:numel(indices)
    dataLegendArray{i} = ['x_{', num2str(i), '}'];
end

figure;
plotDictionary(x, data(:, indices), axisValues, lineWidth, fontSize, YtickStep, mu, dataLegendArray, 'Data samples');
print('figures/data_samples.png', '-dpng', '-r300');

%% Build the cost matrix
M = abs(bsxfun(@minus, x', x));
M = M / median(M(:));

%% Set the parameters
options.stop = 1e-3;
options.verbose = 2;
options.D_step_stop = 5e-5;
options.lambda_step_stop = 5e-4;
options.alpha = 0.5;
options.Kmultiplication = 'symmetric';
options.GPU = 0;

k = 3;
dictionaryLegendArray = cell(k, 1);
for i = 1:k
    dictionaryLegendArray{i} = ['d_{', num2str(i), '}'];
end

gamma = 1 / 50;
wassersteinOrder = 1;

%% Perform Wasserstein NMF
rho1 = 0.1;
rho2 = 0.1;
[D, lambda, objectives] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, rho1, rho2, options);

figure;
plot(objectives, 'LineWidth', 2);
xlabel('Number of outer iterations');
ylabel('Objective');
title('Wasserstein NMF objective');
print('figures/wasserstein_nmf_objective.png', '-dpng', '-r300');

figure;
plotDictionary(x, D, axisValues, lineWidth, fontSize, YtickStep, mu, dictionaryLegendArray, 'Wasserstein NMF');
print('figures/wasserstein_nmf_dictionary.png', '-dpng', '-r300');

%% Perform Wasserstein DL
options.alpha = 0;
options.D_step_stop = 1e-7;
options.lambda_step_stop = 1e-7;
[D_DL, lambda_DL, objectives] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, 0, 0, options);

figure;
plot(objectives, 'LineWidth', 2);
xlabel('Number of outer iterations');
ylabel('Objective');
title('Wasserstein DL objective');
print('figures/wasserstein_dl_objective.png', '-dpng', '-r300');

axisValues(3) = floor(min(D_DL(:) * 100)) / 100;
figure;
plotDictionary(x, D_DL, axisValues, lineWidth, fontSize, YtickStep, mu, dictionaryLegendArray, 'Wasserstein DL');
print('figures/wasserstein_dl_dictionary.png', '-dpng', '-r300');

%% Compare reconstructions
width = 1200;
height = 600;
figure('Position', [1 1 width height]);

axisValues(3) = 0;
i = 1;

subplot(1, 2, 1);
plotDictionary(x, data(:, i), axisValues, lineWidth, fontSize, YtickStep, mu, 'x', 'Data');

subplot(1, 2, 2);
plotDictionary(x, [D * lambda(:, i), D_DL * lambda_DL(:, i)], axisValues, lineWidth, fontSize, YtickStep, mu, ...
    {'NMF reconstruction', 'DL reconstruction'}, 'Reconstruction');
print('figures/reconstruction_comparison.png', '-dpng', '-r300');

