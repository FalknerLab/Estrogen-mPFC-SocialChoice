% Function to visualize DBSCAN clusters on density map
function visualizeDBSCANClusters(densityMap, umapReduction, clusterIds, xEdges, yEdges, varargin)
% visualizeDBSCANClusters - Overlay DBSCAN cluster boundaries on density map
%
% Inputs:
%   densityMap - 2D density map
%   umapReduction - Nx2 UMAP coordinates
%   clusterIds - Nx1 cluster labels from DBSCAN (noise points typically -1)
%   xEdges, yEdges - Bin edges from density map creation
%   varargin - Optional name-value pairs:
%     'LineWidth' - Width of cluster boundaries (default: 2)
%     'ShowNoise' - Whether to show noise points (default: false)
%     'Alpha' - Transparency of cluster regions (default: 0.3)
%     'BoundaryMethod' - 'convex', 'voronoi', or 'alphashape' (default: 'voronoi')
%     'AlphaRadius' - Alpha shape parameter (default: auto)

% Parse inputs
p = inputParser;
addRequired(p, 'densityMap');
addRequired(p, 'umapReduction');
addRequired(p, 'clusterIds');
addRequired(p, 'xEdges');
addRequired(p, 'yEdges');
addParameter(p, 'OverlayFrames', true, @islogical);
addParameter(p, 'LineWidth', 2, @(x) isscalar(x));
addParameter(p, 'ShowNoise', false, @islogical);
addParameter(p, 'Alpha', 0.3, @(x) isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'BoundaryMethod', 'voronoi', @(x) ismember(x, {'convex', 'voronoi', 'alphashape'}));
addParameter(p, 'AlphaRadius', [], @(x) isempty(x) || (isscalar(x) && x > 0));
parse(p, densityMap, umapReduction, clusterIds, xEdges, yEdges, varargin{:});


lineWidth = p.Results.LineWidth;
showNoise = p.Results.ShowNoise;
alpha = p.Results.Alpha;
boundaryMethod = p.Results.BoundaryMethod;
alphaRadius = p.Results.AlphaRadius;
overlayFrames = p.Results.OverlayFrames;

% Get unique cluster IDs (excluding noise if specified)
uniqueClusters = unique(clusterIds);
if ~showNoise
    uniqueClusters = uniqueClusters(uniqueClusters ~= -1); % Remove noise (-1)
end
% Extract x and y coordinates
x = umapReduction(:, 1);
y = umapReduction(:, 2);
% Create bin centers for plotting
xCenters = (xEdges(1:end-1) + xEdges(2:end)) / 2;
yCenters = (yEdges(1:end-1) + yEdges(2:end)) / 2;

% figure;
% 
% % Plot density map as background
% imagesc(xCenters, yCenters, densityMap');
% axis xy;
% % colormap(viridis);
% %     CT = cbrewer('seq','Reds',100,'pchip');
% %     CT(1,:) = [1,1,1];
% %     colormap(CT);
% %     colormap(hot); % You can change this to other colormaps like 'jet', 'parula', etc.
%     colormap(magma);
% %     clim([0 1000]);
%     colorbar;
% hold on;
% if overlayFrames
%     % Optional: overlay original scatter plot with low alpha
%     scatter(x, y, 0.5, 'w', 'MarkerFaceAlpha', 0.01, 'MarkerEdgeAlpha', 0.01);
% end 
hold on;
% Generate colors for clusters
colors = lines(length(uniqueClusters));

if strcmp(boundaryMethod, 'voronoi')
    % Create Voronoi diagram for non-overlapping boundaries
    createVoronoiBoundaries(umapReduction, clusterIds, uniqueClusters, colors, lineWidth, showNoise);
    
elseif strcmp(boundaryMethod, 'alphashape')
    % Use alpha shapes for more accurate cluster boundaries
    createAlphaShapeBoundaries(umapReduction, clusterIds, uniqueClusters, colors, lineWidth, alpha, alphaRadius);
    
else
    % Original convex hull method
    createConvexHullBoundaries(umapReduction, clusterIds, uniqueClusters, colors, lineWidth, alpha);
end

% Show noise points if requested
if showNoise && any(clusterIds == -1)
    noisePoints = umapReduction(clusterIds == -1, :);
    scatter(noisePoints(:,1), noisePoints(:,2), 5, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5);
end

% title(sprintf('DBSCAN Clusters (%d) - %s boundaries', length(uniqueClusters), boundaryMethod));
xlabel('UMAP Dimension 1');
ylabel('UMAP Dimension 2');

end

% Helper function for Voronoi boundaries (non-overlapping)
function createVoronoiBoundaries(umapReduction, clusterIds, uniqueClusters, colors, lineWidth, showNoise)
% Remove noise points for Voronoi calculation
validMask = clusterIds ~= -1;
validPoints = umapReduction(validMask, :);
validClusterIds = clusterIds(validMask);

if length(validPoints) < 3, return; end

% Calculate cluster centroids
centroids = zeros(length(uniqueClusters), 2);
for i = 1:length(uniqueClusters)
    clusterId = uniqueClusters(i);
    clusterMask = validClusterIds == clusterId;
    centroids(i, :) = mean(validPoints(clusterMask, :));
end

% Create Voronoi diagram from centroids
try
    [vx, vy] = voronoi(centroids(:,1), centroids(:,2));
    
    % Plot Voronoi boundaries
%         line_colr = 'k';
    line_colr = [0.8 0.8 0.8];
    plot(vx, vy, 'w-', 'LineWidth', lineWidth, 'Color', line_colr);
    
    % Plot cluster centroids and labels
    for i = 1:length(uniqueClusters)
%         plot(centroids(i,1), centroids(i,2), 'o', 'Color', colors(i,:), ...
%              'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'LineWidth', 2);
        text(centroids(i,1), centroids(i,2), sprintf('C%d', uniqueClusters(i)), ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 10, ...
             'HorizontalAlignment', 'center');
    end
    
catch
        % Just Plot cluster centroids and labels
    for i = 1:length(uniqueClusters)
%         plot(centroids(i,1), centroids(i,2), 'o', 'Color', colors(i,:), ...
%              'MarkerSize', 8, 'MarkerFaceColor', colors(i,:), 'LineWidth', 2);
        text(centroids(i,1), centroids(i,2), sprintf('C%d', uniqueClusters(i)), ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 10, ...
             'HorizontalAlignment', 'center');
    end

%     fprintf('Voronoi diagram failed, using scatter plot instead\n');
%     % Fallback to scatter plot
%     for i = 1:length(uniqueClusters)
%         clusterId = uniqueClusters(i);
%         clusterMask = validClusterIds == clusterId;
%         clusterPoints = validPoints(clusterMask, :);
%         scatter(clusterPoints(:,1), clusterPoints(:,2), 10, colors(i,:), 'filled');
%     end
end
end

% Helper function for Alpha Shape boundaries
function createAlphaShapeBoundaries(umapReduction, clusterIds, uniqueClusters, colors, lineWidth, alpha, alphaRadius)
for i = 1:length(uniqueClusters)
    clusterId = uniqueClusters(i);
    clusterMask = clusterIds == clusterId;
    
    if sum(clusterMask) < 3, continue; end
    
    clusterPoints = umapReduction(clusterMask, :);
    
    try
        % Create alpha shape
        if isempty(alphaRadius)
            % Auto-determine alpha radius
            distances = pdist(clusterPoints);
            alphaRadius = median(distances) * 0.5;
        end
        
        shp = alphaShape(clusterPoints(:,1), clusterPoints(:,2), alphaRadius);
        
        % Plot alpha shape boundary
        plot(shp, 'FaceColor', colors(i,:), 'FaceAlpha', alpha, ...
             'EdgeColor', colors(i,:), 'LineWidth', lineWidth);
        
        % Add cluster label at centroid
        centroid = mean(clusterPoints);
        text(centroid(1), centroid(2), sprintf('C%d', clusterId), ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 10, ...
             'HorizontalAlignment', 'center');
             
    catch
        % Fallback to scatter if alpha shape fails
        scatter(clusterPoints(:,1), clusterPoints(:,2), 10, colors(i,:), 'filled');
    end
end
end

% Helper function for Convex Hull boundaries
function createConvexHullBoundaries(umapReduction, clusterIds, uniqueClusters, colors, lineWidth, alpha)
for i = 1:length(uniqueClusters)
    clusterId = uniqueClusters(i);
    clusterMask = clusterIds == clusterId;
    
    if sum(clusterMask) < 3, continue; end
    
    clusterPoints = umapReduction(clusterMask, :);
    
    try
        % Create convex hull
        k = convhull(clusterPoints(:,1), clusterPoints(:,2));
        
        % Plot boundary
        plot(clusterPoints(k,1), clusterPoints(k,2), ...
             'Color', colors(i,:), 'LineWidth', lineWidth);
        
        % Fill if alpha > 0
        if alpha > 0
            fill(clusterPoints(k,1), clusterPoints(k,2), colors(i,:), ...
                 'FaceAlpha', alpha, 'EdgeColor', 'none');
        end
        
        % Add label
        centroid = mean(clusterPoints);
        text(centroid(1), centroid(2), sprintf('C%d', clusterId), ...
             'Color', 'white', 'FontWeight', 'bold', 'FontSize', 10, ...
             'HorizontalAlignment', 'center');
             
    catch
        scatter(clusterPoints(:,1), clusterPoints(:,2), 10, colors(i,:), 'filled');
    end
end
end