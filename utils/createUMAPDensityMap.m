function [densityMap, xEdges, yEdges] = createUMAPDensityMap(umapReduction, varargin)
% createUMAPDensityMap - Create a smoothed density map from 2D UMAP reduction
%
% Inputs:
%   umapReduction - Nx2 matrix of UMAP coordinates (N frames x 2 dimensions)
%   varargin - Optional name-value pairs:
%     'NumBins' - Number of bins for histogram (default: 200)
%     'SigmaGaussian' - Standard deviation for Gaussian smoothing (default: 3.5)
%     'PlotResult' - Whether to plot the result (default: true)
%
% Outputs:
%   densityMap - 200x200 (or NumBins x NumBins) smoothed density map
%   xEdges - Bin edges for x-axis
%   yEdges - Bin edges for y-axis

% Parse input arguments
p = inputParser;
addRequired(p, 'umapReduction', @(x) ismatrix(x) && size(x,2) == 2);
addParameter(p, 'umapReduction_for_edges', @(x) ismatrix(x) && size(x,2) == 2);
addParameter(p, 'NumBins', 200, @(x) isscalar(x) && x > 0);
addParameter(p, 'SigmaGaussian', 3.5, @(x) isscalar(x) && x > 0);
addParameter(p, 'PlotResult', true, @islogical);
addParameter(p, 'OverlayFrames', true, @islogical);
parse(p, umapReduction, varargin{:});

numBins = p.Results.NumBins;
sigmaGaussian = p.Results.SigmaGaussian;
plotResult = p.Results.PlotResult;
overlayFrames = p.Results.OverlayFrames;
umapReduction_for_edges = p.Results.umapReduction_for_edges;

% Extract x and y coordinates
x = umapReduction_for_edges(:, 1);
y = umapReduction_for_edges(:, 2);

% Create histogram bins
xMin = min(x);
xMax = max(x);
yMin = min(y);
yMax = max(y);

% Add small padding to ensure all points are included
xRange = xMax - xMin;
yRange = yMax - yMin;
padding = 0.01; % 1% padding

xEdges = linspace(xMin - padding*xRange, xMax + padding*xRange, numBins + 1);
yEdges = linspace(yMin - padding*yRange, yMax + padding*yRange, numBins + 1);

x = umapReduction(:, 1);
y = umapReduction(:, 2);

% Create 2D histogram
[counts, ~, ~] = histcounts2(x, y, xEdges, yEdges);

% Create 2D Gaussian kernel for smoothing
% Kernel size should be large enough to capture the Gaussian (typically 6*sigma)
kernelSize = ceil(6 * sigmaGaussian);
if mod(kernelSize, 2) == 0
    kernelSize = kernelSize + 1; % Ensure odd size for symmetric kernel
end

% Create meshgrid for kernel
[kx, ky] = meshgrid(-(kernelSize-1)/2:(kernelSize-1)/2, -(kernelSize-1)/2:(kernelSize-1)/2);

% Create 2D Gaussian kernel
gaussianKernel = exp(-(kx.^2 + ky.^2) / (2 * sigmaGaussian^2));
gaussianKernel = gaussianKernel / sum(gaussianKernel(:)); % Normalize

% Apply Gaussian smoothing using convolution
densityMap = conv2(counts, gaussianKernel, 'same');
% densityMap = conv2(counts*100/sum(sum(counts)),
% gaussianKernel,'same');%percent instead of counts

% Display information
fprintf('Created density map from %d data points\n', size(umapReduction, 1));
fprintf('Histogram bins: %dx%d\n', numBins, numBins);
fprintf('Gaussian smoothing sigma: %.1f\n', sigmaGaussian);
fprintf('Kernel size: %dx%d\n', kernelSize, kernelSize);

% Plot result if requested
if plotResult
%     figure;

    % Create bin centers for plotting
    xCenters = (xEdges(1:end-1) + xEdges(2:end)) / 2;
    yCenters = (yEdges(1:end-1) + yEdges(2:end)) / 2;
    
    % Plot density map
    imagesc(xCenters, yCenters, densityMap');
    axis xy; % Correct orientation
%     CT = cbrewer('seq','Purples',100,'pchip');
    CT = cbrewer('seq','Greys',100,'pchip');
    CT(1,:) = [1,1,1];
%     colormap(CT);
%     colormap(hot); % You can change this to other colormaps like 'jet', 'parula', etc.
    colormap(magma);
%     clim([0 50]);
    colorbar;
    title('UMAP Density Map (Smoothed)');
    xlabel('UMAP Dimension 1');
    ylabel('UMAP Dimension 2');
    if overlayFrames
    % Optional: overlay original scatter plot with low alpha
    hold on;
    scatter(x, y, 0.5, 'w', 'MarkerFaceAlpha', 0.01, 'MarkerEdgeAlpha', 0.01);
    hold off;
    end    
end
end