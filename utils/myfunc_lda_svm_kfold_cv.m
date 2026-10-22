function decoding_accuracy = myfunc_lda_svm_kfold_cv(k, neuralActivity, behaviorLabels)

    % neuralActivity: nNeurons x nSamples
    % behaviorLabels: nSamples x 1 (or 1 x nSamples)
    % k: number of CV folds

    % Make sure labels are a column
    behaviorLabels = behaviorLabels(:);

    % Create k-fold partition
    cv = cvpartition(behaviorLabels, 'KFold', k);

    % Store fold accuracies
    fold_accuracies = zeros(k,1);

    for fold = 1:k

        % Training/testing split
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        X_train = neuralActivity(:, trainIdx);
        y_train = behaviorLabels(trainIdx);

        X_test = neuralActivity(:, testIdx);
        y_test = behaviorLabels(testIdx);

        % Fit LDA on training data only
        [~, W_train, ~] = myfunc_neuralLDA(X_train, y_train);

        % Project training and test data into LDA space
        X_train_proj = W_train' * X_train;   % dims x nTrain
        X_test_proj  = W_train' * X_test;    % dims x nTest

        % fitcecoc expects observations x features
        X_train_proj = X_train_proj';
        X_test_proj  = X_test_proj';

        % Train multiclass linear SVM
        t = templateSVM('KernelFunction', 'linear', 'Standardize', true);

        svmModel = fitcecoc(X_train_proj, y_train, ...
            'Learners', t, ...
            'Coding', 'onevsone');

        % Predict on held-out test data
        y_pred = predict(svmModel, X_test_proj);

        % Accuracy for this fold
        fold_accuracies(fold) = mean(y_pred == y_test);
    end

    % Mean CV accuracy
    decoding_accuracy = mean(fold_accuracies);

end
