module DataHandling

"""
Designed to handle data manipulation and partitioning, this module offers tools for splitting data into training and test sets,
managing datasets for model training and validation, and facilitating data-related tasks essential for machine learning workflows.
It provides utilities to manage and partition data effectively.
"""

using Random

export holdOut, crossvalidation, splitCrossValidationData, splitTrainAndValidation

"""
    holdOut(N::Int, P::Float64)

The `holdOut` function performs a hold-out split on a dataset by randomly dividing it into a training set and a test set.

## Arguments
- `N::Int`: The total number of samples in the dataset.
- `P::Float64`: The proportion of samples to be included in the test set. Must be a value between 0.0 and 1.0.

## Returns
A tuple `(train_indexes, test_indexes)` containing the indexes of the training set and the test set, respectively.
"""
function holdOut(N::Int, P::Float64)
    # Verify that P is a valid probability
    @assert ((P >= 0.0) && (P <= 1.0))

    # Generate random permutation of integers from 1 to N
    indexes = randperm(N)

    # Calculate the size of the test set
    test_size = Int(ceil(N * P))

    # Select the indexes of the test and training sets
    test_indexes = indexes[1:test_size]
    train_indexes = indexes[test_size+1:end]

    # Verify that the length of the training and test sets is correct
    @assert length(train_indexes) + length(test_indexes) == N

    # Return the indexes of the training and test sets
    return (train_indexes, test_indexes)
end

"""
    holdOut(N::Int, Pval::Float64, Ptest::Float64)

Split the dataset into training, validation, and test sets using the hold-out method.

# Arguments
- `N::Int`: The total number of samples in the dataset.
- `Pval::Float64`: The proportion of samples to allocate for the validation set.
- `Ptest::Float64`: The proportion of samples to allocate for the test set.

# Returns
A tuple `(train_indexes, val_indexes, test_indexes)` containing the indices of the samples in the training, validation, and test sets, respectively.
"""
function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    # Verify that Pval and Ptest are valid probabilities
    @assert ((Pval >= 0.0) && (Pval <= 1.0))
    @assert ((Ptest >= 0.0) && (Ptest <= 1.0))
    @assert ((Pval + Ptest) <= 1.0)

    train_val_indexes, test_indexes = holdOut(N, Ptest)

    train_val_size = length(train_val_indexes)

    train_indexes, val_indexes = holdOut(train_val_size, Pval * N / train_val_size)

    train_indexes = train_val_indexes[train_indexes]
    val_indexes = train_val_indexes[val_indexes]

    @assert length(train_indexes) + length(val_indexes) + length(test_indexes) == N

    return (train_indexes, val_indexes, test_indexes)
end


"""
    crossvalidation(N::Int64, k::Int64)

Generate shuffled indices for cross-validation.

# Arguments
- `N::Int64`: The total number of patterns.
- `k::Int64`: The number of folds.

# Returns
- An array of shuffled indices representing the folds for cross-validation.

"""
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

"""
    crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)

Generate shuffled indices for stratified k-fold cross-validation.

# Arguments
- `targets::AbstractArray{Bool,2}`: The target labels for the patterns.
- `k::Int64`: The number of folds.

# Returns
- An array of shuffled indices representing the folds for stratified k-fold cross-validation.

"""
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    # Check that there are enough instances per class to perform k-fold cross-validation
    @assert all(sum(targets, dims = 1) .>= k) "There are not enough instances per class to perform a $(k)-fold cross-validation"

    # Initialize the index vector to zeros
    idx = Int.(zeros(size(targets, 1)))

    # Iterate over the classes and perform stratified k-fold cross-validation
    for class = 1:size(targets, 2)
        # Find the indices of the patterns that belong to the current class
        class_indices = findall(targets[:, class])
        # Perform stratified k-fold cross-validation on the patterns that belong to the current class
        class_folds = crossvalidation(length(class_indices), k)
        # Update the index vector with the indices of the folds for the current class
        idx[class_indices] .= class_folds
    end

    # Return the index vector
    return idx
end

# function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
#     print("Entra aqui")
#     # Convert the targets to a binary matrix using one-hot encoding
#     binary_targets = oneHotEncoding(targets)
#     # Perform stratified k-fold cross-validation on the binary matrix
#     idx = crossvalidation(binary_targets, k)
#     return idx
# end

"""
    crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)

Generate shuffled indices for stratified k-fold cross-validation.

# Arguments
- `targets::AbstractArray{<:Any,1}`: The target labels for the patterns.
- `k::Int64`: The number of folds.

# Returns
- An array of shuffled indices representing the folds for stratified k-fold cross-validation.

"""
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    # Get the unique classes in the targets vector
    classes = unique(targets)
    # Initialize the index vector to zeros
    idx = Int.(zeros(length(targets)))

    # Iterate over the classes and perform stratified k-fold cross-validation
    for class in classes
        # Find the indices of the patterns that belong to the current class
        classIdx = (targets .== class)
        # Count the number of instances in the current class
        numInstances = sum(classIdx)
        # Check that there are enough instances per class to perform k-fold cross-validation
        @assert (numInstances .>= k) "There are not enough instances per class to perform a $(k)-fold cross-validation"
        # Perform stratified k-fold cross-validation on the patterns that belong to the current class
        classFolds = crossvalidation(numInstances, k)
        # Update the index vector with the indices of the folds for the current class
        idx[classIdx] .= classFolds
    end

    return idx
end

"""
    splitCrossValidationData(
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}},
        numFold::Int64,
        kFoldIndices::Array{Int64,1}
    )

Split the training dataset into training and test sets based on the specified fold index.

# Arguments
- `trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}}`: The training dataset consisting of input features and target labels.
- `numFold::Int64`: The index of the fold to be used as the test set.
- `kFoldIndices::Array{Int64,1}`: The array of shuffled indices representing the folds for cross-validation.

# Returns
- A tuple `(trainingInputs, trainingTargets, testInputs, testTargets)` containing the training and test sets.

"""
function splitCrossValidationData(
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,2}},
    numFold::Int64,
    kFoldIndices::Array{Int64,1},
)

    inputs, targets = trainingDataset

    trainingInputs = inputs[kFoldIndices.!=numFold, :]
    trainingTargets = targets[kFoldIndices.!=numFold, :]

    testInputs = inputs[kFoldIndices.==numFold, :]
    testTargets = targets[kFoldIndices.==numFold, :]

    return (trainingInputs, trainingTargets, testInputs, testTargets)

end

"""
    splitTrainAndValidation(inputs, targets, validationRatio)

Split the inputs and targets into training and validation sets based on the specified validation ratio.

# Arguments
- `inputs`: The input features.
- `targets`: The target labels.
- `validationRatio`: The ratio of data to be used for validation.

# Returns
- A tuple `(trainingInputs, trainingTargets, validationInputs, validationTargets)` containing the training and validation sets.

"""
function splitTrainAndValidation(inputs, targets, validationRatio)
    (trainingIndexes, validationIndexes) = holdOut(size(inputs, 1), validationRatio)
    validationInputs = inputs[validationIndexes, :]
    validationTargets = targets[validationIndexes, :]
    trainingInputs = inputs[trainingIndexes, :]
    trainingTargets = targets[trainingIndexes, :]

    return trainingInputs, trainingTargets, validationInputs, validationTargets
end

end
