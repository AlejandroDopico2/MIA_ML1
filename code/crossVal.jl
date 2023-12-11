using Statistics;
using Flux;
using Flux.Losses;
using Random;
using Flux:train!;

# Define a function that performs k-fold cross-validation
# N: the number of patterns
# k: the number of folds
function crossvalidation(N::Int64, k::Int64)
    # Calculate the number of patterns in each fold
    fold_size = ceil(Int, N / k)
    # Repeat the indices of the folds k times
    idx = repeat(1:k, fold_size)[1:N]
    # Shuffle the indices randomly
    shuffle!(idx)
    # Return the shuffled indices
    return idx
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # Check that there are enough instances per class to perform k-fold cross-validation
    @assert all(sum(targets, dims=1) .>= k) "There are not enough instances per class to perform a $(k)-fold cross-validation"
    
    # Initialize the index vector to zeros
    idx = Int.(zeros(size(targets, 1)))
    
    # Iterate over the classes and perform stratified k-fold cross-validation
    for class in 1:size(targets, 2)
        # Find the indices of the patterns that belong to the current class
        class_indices = findall(targets[:, class])
        # Perform stratified k-fold cross-validation on the patterns that belong to the current class
        class_folds = crossvalidation(length(class_indices), k)
        # Update the index vector with the indices of the folds for the current class
        idx[class_indices] .= class_folds
    end
    
    # Return the index vector
    return idx
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    # Convert the targets to a binary matrix using one-hot encoding
    binary_targets = oneHotEncoding(targets)
    # Perform stratified k-fold cross-validation on the binary matrix
    idx = crossvalidation(binary_targets, k)
    return idx
end

function trainClassANN(topology::AbstractArray{<:Int,1}, 
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
        kFoldIndices::	Array{Int64,1}; 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
        validationRatio::Real=0.0, maxEpochsVal::Int=20)

    # Set random seed for reproducibility
    Random.seed!(42)

    # Compute number of folds
    kFolds = maximum(kFoldIndices)

    # Initialize array to store test accuracies for each fold
    testAccuracies = Array{Tuple{Float64, Float64}, 1}(undef, kFolds)

    # Unpack training dataset
    (inputs, targets) = trainingDataset

    # Loop over folds
    for numFold in 1:kFolds
        
        # Split data into training and test sets for current fold
        trainingInputs = inputs[kFoldIndices .!= numFold, :]
        trainingTargets = targets[kFoldIndices .!= numFold, :]
        testInputs = inputs[kFoldIndices .== numFold, :]
        testTargets = targets[kFoldIndices .== numFold, :]

        # Initialize array to store test accuracies for each repetition
        testAccuraciesForEachRepetition = Array{Float64, 1}(undef, repetitionsTraining)

        # Loop over repetitions
        for repetition in 1:repetitionsTraining

            # Split training set into training and validation sets if validation ratio is greater than 0
            if validationRatio > 0.0

                # Split training set into training and validation sets
                (trainingIndexes, validationIndexes) = holdOut(size(inputs, 1), validationRatio)
                validationInputs = trainingInputs[validationIndexes,:]
                validationTargets = trainingTargets[validationIndexes,:]
                trainingInputs = trainingInputs[trainingIndexes,:]
                trainingTargets = trainingTargets[trainingIndexes,:]

                # Train model with current fold and repetition using training and validation sets
                model, _ = trainClassANN(topology, (trainingInputs, trainingTargets);
                    validationDataset = (validationInputs, validationTargets),
                    testDataset = (testInputs, testTargets), transferFunctions = transferFunctions,
                    maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate, maxEpochsVal = maxEpochsVal
                )

            else

                # Train model with current fold and repetition using only training set
                model, _ = trainClassANN(topology, (trainingInputs, trainingTargets);
                    testDataset = (testInputs, testTargets), transferFunctions = transferFunctions,
                    maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate, maxEpochsVal = maxEpochsVal
                )

            end

            # Compute test accuracy for current repetition
            outputs = model(testInputs')'
            testAccuraciesForEachRepetition[repetition] = accuracy(outputs, testTargets)

        end

        # Compute mean and standard deviation of test accuracies for current fold
        testAccuracies[numFold] = (mean(testAccuraciesForEachRepetition), std(testAccuraciesForEachRepetition))
    end

    # Compute mean and standard deviation of test accuracies over all folds
    testAccuracy = mean(fold[1] for fold in testAccuracies)
    testAccuracyStd = std(fold[2] for fold in testAccuracies)

    return testAccuracy, testAccuracyStd
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    kFoldIndices::	Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,repetitionsTraining::Int=1, 
    validationRatio::Real=0.0, maxEpochsVal::Int=20)

# Unpack inputs and targets from training dataset
inputs, targets = trainingDataset

# Reshape targets to be a column vector
targets = reshape(targets, :, 1)

# Call trainClassANN with inputs, targets, and kFoldIndices
return trainClassANN(topology, (inputs, targets), kFoldIndices;
    transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss,
    learningRate = learningRate, repetitionsTraining = repetitionsTraining,
    validationRatio = validationRatio, maxEpochsVal = maxEpochsVal
)
end