function [mhlnbs_dist, euc_dist] = myfunc_visualizeLDA_WithEllipses(projectedData, behaviorLabels, alpha,class_colors,euc_or_mhlnbs)
    % projectedData: Projected data in LDA space
    % behaviorLabels: Original behavior labels
    % alpha: Confidence level (default is 0.05 for 95% confidence)
    
    if nargin < 3
        alpha = 0.05; % Default 95% confidence
    end
    chi2_val = sqrt(chi2inv(1-alpha, 2));
    % Get unique labels
    uniqueLabels = unique(behaviorLabels);
    n_classes = length(uniqueLabels);
    
%     % Create a color map for different classes
%     class_colors = lines(n_classes);
    
%     % Create a figure
%     figure;
%     hold on;
    
    % For storing ellipse centers for legend
    centers_handle = zeros(n_classes, 1);
    robust_cov_matrices = cell(1,n_classes);
    lda_robust_centers = zeros(2,n_classes);
    % Plot each class with a different color and confidence ellipse
    for i = 1:n_classes
        class_idx = behaviorLabels == uniqueLabels(i);
        class_data = projectedData(:, class_idx);
        
        if size(projectedData, 1) >= 2
            % 2D plot with ellipse
            h = scatter(class_data(1, :), class_data(2, :), 40, class_colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5);
            hold on
            % Compute robust location and scatter - robust to outliers
                [mcd_cov, mcd_center] = robustcov(class_data', 'OutlierFraction', 0.1);
                center = mcd_center';
                covariance = mcd_cov;
                robust_cov_matrices{i} = covariance;
                lda_robust_centers(:,i) = center;
%                 %Alternatively if you are just using centroids - will be
%                 %influenced more by outliers
%                 center = mean(class_data, 2);
%                 covariance = cov(class_data');
            
            % Draw the 95% confidence ellipse
%             drawEllipse(center, covariance, alpha, class_colors(i,:));
            % Compute eigenvalues and eigenvectors for ellipse shape
            [eig_vec, eig_val] = eig(covariance);
            
            % Convert eigenvalues to ellipse axes (scaling factor from chi-square distribution)
            axes_lengths = chi2_val * sqrt(diag(eig_val));
            
            % Generate ellipse points
            theta = linspace(0, 2*pi, 100);
            ellipse_points = eig_vec * diag(axes_lengths) * [cos(theta); sin(theta)] + center;
            
            % Plot the ellipse
            plot(ellipse_points(1,:), ellipse_points(2,:), 'Color', class_colors(i,:), 'LineWidth', 2);
            hold on
            % Plot the center of the ellipse
            plot(center(1), center(2), 'x', 'Color', 'k', 'MarkerSize', 12, 'LineWidth', 2);
            hold on
        else
            % 1D plot
            scatter(class_data(1, :), zeros(1, sum(class_idx)), 40, class_colors(i,:), 'filled');
        end
    end
    
    % Add labels and legend
    if size(projectedData, 1) >= 2
        xlabel('LD1'); ylabel('LD2');
    else
        xlabel('LD1');
    end
    
%     title('LDA Projection with 95% Confidence Ellipses');
    grid on;
    hold on

    % Add lines, pairwise mahalanobis dist
    mhlnbs_dist = [];
    euc_dist = [];
%     total_samples = length(behaviorLabels);
    % Compute and plot pairwise Mahalanobis distances
    for i = 1:n_classes
        for j = i+1:n_classes
            % Pooled covariance matrix
            pooled_cov = 0.5 * (robust_cov_matrices{i} + robust_cov_matrices{j});
            mahalanobis_dist = sqrt((lda_robust_centers(:,i) - lda_robust_centers(:,j))' * inv(pooled_cov) * (lda_robust_centers(:,i) - lda_robust_centers(:,j)));
            mhlnbs_dist(i,j) = mahalanobis_dist;
            % Euclidean Distance
            euc_dist(i,j) = norm(lda_robust_centers(:,i) - lda_robust_centers(:,j));
            % Draw line connecting class centers
            line([lda_robust_centers(1,i), lda_robust_centers(1,j)], ...
                 [lda_robust_centers(2,i), lda_robust_centers(2,j)], ...
                 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1);
             
            % Annotate line with Mahalanobis distance
            mid_x = (lda_robust_centers(1,i) + lda_robust_centers(1,j)) / 2;
            mid_y = (lda_robust_centers(2,i) + lda_robust_centers(2,j)) / 2;
            if euc_or_mhlnbs == 2
            text(mid_x, mid_y, sprintf('%.2f', mhlnbs_dist(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
            elseif euc_or_mhlnbs == 1
            text(mid_x, mid_y, sprintf('%.2f', euc_dist(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
            end
        end
    end
end

