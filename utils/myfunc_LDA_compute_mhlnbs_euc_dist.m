function [mhlnbs_dist, euc_dist] = myfunc_LDA_compute_mhlnbs_euc_dist(projectedData, behaviorLabels)
    % Get unique labels
    uniqueLabels = unique(behaviorLabels);
    n_classes = length(uniqueLabels);
    robust_cov_matrices = cell(1,n_classes);
    lda_robust_centers = zeros(2,n_classes);
    % Plot each class with a different color and confidence ellipse
    for i = 1:n_classes
        class_idx = behaviorLabels == uniqueLabels(i);
        class_data = projectedData(:, class_idx);
        
        % Compute robust location and scatter - robust to outliers
        [mcd_cov, mcd_center] = robustcov(class_data', 'OutlierFraction', 0.1);
        center = mcd_center';
        covariance = mcd_cov;
        robust_cov_matrices{i} = covariance;
        lda_robust_centers(:,i) = center;
    end

    mhlnbs_dist = [];
    euc_dist = [];
    % Compute and plot pairwise Mahalanobis distances
    for i = 1:n_classes
        for j = i+1:n_classes
            % Pooled covariance matrix
            pooled_cov = 0.5 * (robust_cov_matrices{i} + robust_cov_matrices{j});
            mahalanobis_dist = sqrt((lda_robust_centers(:,i) - lda_robust_centers(:,j))' * inv(pooled_cov) * (lda_robust_centers(:,i) - lda_robust_centers(:,j)));
            mhlnbs_dist(i,j) = mahalanobis_dist;
            % Euclidean Distance
            euc_dist(i,j) = norm(lda_robust_centers(:,i) - lda_robust_centers(:,j));
        end
    end

end