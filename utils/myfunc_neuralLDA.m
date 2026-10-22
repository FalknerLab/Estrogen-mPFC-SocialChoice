function [projectedData, W, eigenvalues] = myfunc_neuralLDA(neuralActivity, behaviorLabels)
   % neuralActivity: Matrix of size (n neurons × t time bins)
    % behaviorLabels: Vector of size (t time bins × 1) with class labels
    
    % Get dimensions
    [n_neurons, n_samples] = size(neuralActivity);
    
    % Get unique class labels

    uniqueLabels = unique(behaviorLabels);
    n_classes = length(uniqueLabels);
    
    % Compute overall mean
    overall_mean = mean(neuralActivity, 2);
    
    % Initialize between-class scatter matrix
    S_B = zeros(n_neurons, n_neurons);
    
    % Initialize within-class scatter matrix
    S_W = zeros(n_neurons, n_neurons);
    
    % Class counts and means
    class_means = zeros(n_neurons, n_classes);
    class_counts = zeros(n_classes, 1);
    
    % Calculate class means and counts
    for i = 1:n_classes
        class_idx = behaviorLabels == uniqueLabels(i);
        class_data = neuralActivity(:, class_idx);
        class_counts(i) = sum(class_idx);
        class_means(:, i) = mean(class_data, 2);
    end

    % Calculate between-class scatter matrix
    for i = 1:n_classes
        mean_diff = class_means(:, i) - overall_mean;
        S_B = S_B + class_counts(i) * (mean_diff * mean_diff');
    end
    
    % Calculate within-class scatter matrix
    for i = 1:n_classes
        class_idx = behaviorLabels == uniqueLabels(i);
        class_data = neuralActivity(:, class_idx);

        centered_data = class_data - class_means(:, i);
        S_W = S_W + (centered_data * centered_data');
    end
    
    % Solve the generalized eigenvalue problem
    % We need to compute inv(S_W) * S_B
    % But to avoid numerical issues, we'll use eig(A, B) 
    [eigenvectors, eigenvalues_diag] = eig(S_B, S_W);
    
    % Convert diagonal matrix to vector and sort
    eigenvalues = diag(eigenvalues_diag);
    [eigenvalues, sort_idx] = sort(eigenvalues, 'descend');
    
    % Sort eigenvectors accordingly
    eigenvectors = eigenvectors(:, sort_idx);
    
    % Keep only n_classes-1 eigenvectors (maximum number of discriminants)
    max_discriminants = min(n_classes-1, n_neurons);
    W = eigenvectors(:, 1:max_discriminants);
    eigenvalues = eigenvalues(1:max_discriminants);
    
    % Project the data onto the LDA space
    projectedData = W' * neuralActivity;
end