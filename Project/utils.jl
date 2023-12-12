using Statistics;
using Flux;
using Flux.Losses;
using Random;
using Flux: train!;

"""
    oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})

Encode categorical feature using one-hot encoding.

# Arguments
- `feature`: An array of categorical values to be encoded.
- `classes`: An array of unique classes in the feature.

# Returns
- `one_hot`: A binary matrix representing the one-hot encoded feature.
"""
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    if num_classes == 2
        one_hot = reshape(feature .== classes[2], :, 1)
    else
        one_hot = zeros(Bool, length(feature), num_classes)

        for i = 1:num_classes
            one_hot[:, i] .= (feature .== classes[i])
        end

        return one_hot
    end
end

"""
    oneHotEncoding(feature::AbstractArray{<:Any,1})

Encode categorical feature using one-hot encoding.

# Arguments
- `feature`: An array of categorical values.

# Returns
An encoded matrix where each column represents a unique category and each row represents an instance.

"""
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

"""
    oneHotEncoding(feature::AbstractArray{Bool,1})

Encode a boolean feature vector into a one-hot encoded matrix.

# Arguments
- `feature`: A 1-dimensional boolean array representing the feature vector.

# Returns
- A matrix where each row corresponds to a one-hot encoded representation of the input feature vector.
"""
function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1)
end

"""
    calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})

Calculate the minimum and maximum values for each column of the dataset.

# Arguments
- `dataset`: A 2-dimensional array of real numbers.

# Returns
A tuple containing the minimum and maximum values for each column of the dataset.

"""
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (minimum(dataset, dims = 1), maximum(dataset, dims = 1))
end


"""
    calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})

Calculate the mean and standard deviation of each column in the dataset.

# Arguments
- `dataset`: A 2-dimensional array of real numbers.

# Returns
A tuple containing the mean and standard deviation of each column in the dataset.

"""
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (mean(dataset, dims = 1), std(dataset, dims = 1))
end

"""
    normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

Normalize the dataset using the min-max normalization method.

# Arguments
- `dataset::AbstractArray{<:Real,2}`: The dataset to be normalized.
- `normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}`: The normalization parameters, a tuple containing the minimum and maximum values for each feature.

# Returns
- `dataset::AbstractArray{<:Real,2}`: The normalized dataset.
"""
function normalizeMinMax!(
    dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}},
)
    min, max = normalizationParameters

    dataset .-= min
    dataset ./= (max .- min)

    dataset[:, vec(min .== max)] .= 0

    return dataset
end

"""
    normalizeMinMax!(dataset::AbstractArray{<:Real,2})

Normalize the dataset using min-max normalization in place.

# Arguments
- `dataset::AbstractArray{<:Real,2}`: The dataset to be normalized.

# Returns
- `dataset::AbstractArray{<:Real,2}`: The normalized dataset.
"""
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

"""
    normalizeMinMax(dataset, normalizationParameters)

Normalize the given dataset using the min-max normalization method.

# Arguments
- `dataset::AbstractArray{<:Real,2}`: The dataset to be normalized.
- `normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}`: The normalization parameters obtained from the training dataset.

# Returns
- `normalizedDataset::AbstractArray{<:Real,2}`: The normalized dataset.
"""
function normalizeMinMax(
    dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}},
)

    dataset_copy = copy(dataset)

    return normalizeMinMax!(dataset_copy, normalizationParameters)

end

"""
    normalizeMinMax(dataset::AbstractArray{<:Real,2})

Normalize the values of a dataset using the min-max scaling technique.

# Arguments
- `dataset`: The dataset to be normalized.

# Returns
- A new normalized dataset.
"""
function normalizeMinMax(dataset::AbstractArray{<:Real,2})

    dataset_copy = copy(dataset)

    return normalizeMinMax!(dataset_copy)

end

"""
    normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

Normalize the dataset by subtracting the mean and dividing by the standard deviation.

Arguments
- `dataset`: The dataset to be normalized.
- `normalizationParameters`: A tuple containing the mean and standard deviation of the dataset.

Returns
- The normalized dataset.
"""
function normalizeZeroMean!(
    dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}},
)

    mean, std = normalizationParameters

    dataset .-= mean
    dataset ./= std
    dataset[:, vec(std .== 0)] .= 0

    return dataset
end

"""
    normalizeZeroMean!(dataset::AbstractArray{<:Real,2})

Normalize the dataset by subtracting the mean value of each feature from all the data points.

# Arguments
- `dataset`: A 2-dimensional array of real numbers representing the dataset.

# Returns
- The normalized dataset.
"""
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end

"""
    normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

Normalize the dataset by subtracting the mean of each feature from the corresponding feature values.

# Arguments
- `dataset`: The input dataset as a 2-dimensional array of real numbers.
- `normalizationParameters`: A tuple containing two arrays representing the mean and standard deviation of each feature in the dataset.

# Returns
- A normalized copy of the input dataset.
"""
function normalizeZeroMean(
    dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}},
)
    dataset_copy = copy(dataset)

    return normalizeZeroMean!(dataset_copy, normalizationParameters)
end

"""
    normalizeZeroMean(dataset::AbstractArray{<:Real,2})

Normalize the dataset by subtracting the mean of each column.

# Arguments
- `dataset`: The input dataset as a 2-dimensional array.

# Returns
- A normalized copy of the dataset.

"""
function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    dataset_copy = copy(dataset)

    return normalizeZeroMean!(dataset_copy)
end

"""
    classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)

This function takes an array of outputs and classifies them based on a given threshold. If the array has only one column, it applies a binary classification by comparing each element with the threshold. If the array has multiple columns, it finds the index of the maximum value for each row and sets the corresponding element in the output array to true, while setting all other elements to false.

# Arguments
- `outputs`: An array of outputs to be classified.
- `threshold`: The threshold value for binary classification. Defaults to 0.5.

# Returns
- An array of the same size as `outputs` with the classified results.

"""
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real = 0.5)
    if size(outputs, 2) == 1
        outputs = outputs .> threshold
    else
        _, indicesMaxEachInstance = findmax(outputs, dims = 2)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
    end

    return outputs
end

"""
    accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

Compute the accuracy of a binary classification model.

# Arguments
- `outputs::AbstractArray{Bool,1}`: The predicted outputs of the model.
- `targets::AbstractArray{Bool,1}`: The true targets.

# Returns
- `accuracy::Float64`: The accuracy of the model, calculated as the proportion of correct predictions.
"""
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    @assert (size(outputs) == size(targets)) "Outputs and targets must have same number of rows"

    return mean(outputs .== targets)
end

"""
    accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})

Compute the accuracy of a classification model given the predicted outputs and the target labels.

# Arguments
- `outputs::AbstractArray{Bool,2}`: A 2-dimensional array representing the predicted outputs of the model.
- `targets::AbstractArray{Bool,2}`: A 2-dimensional array representing the target labels.

# Returns
- `accuracy::Float64`: The accuracy of the model, calculated as the proportion of correctly classified instances.
"""
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})

    if size(outputs, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    end

    @assert (size(outputs) == size(targets)) "Outputs and targets must have same number of rows"

    _, indicesMaxEachInstanceInOutputs = findmax(outputs, dims = 2)
    _, indicesMaxEachInstanceInTargets = findmax(targets, dims = 2)

    return mean(indicesMaxEachInstanceInOutputs .== indicesMaxEachInstanceInTargets)
end

"""
    accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

Compute the accuracy of a binary classification model.

This function takes in two arrays, `outputs` and `targets`, and computes the accuracy of a binary classification model. The `outputs` array contains the predicted values, which are assumed to be real numbers. The `targets` array contains the true labels, which are assumed to be boolean values.

The optional `threshold` parameter specifies the threshold value for converting the predicted values to binary predictions. By default, the threshold is set to 0.5.

# Arguments
- `outputs::AbstractArray{<:Real,1}`: An array of predicted values.
- `targets::AbstractArray{Bool,1}`: An array of true labels.
- `threshold::Real=0.5`: The threshold value for converting predicted values to binary predictions.

# Returns
- `accuracy::Float64`: The accuracy of the binary classification model.
"""
function accuracy(
    outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1};
    threshold::Real = 0.5,
)

    outputs = outputs .> threshold
    return accuracy(outputs, targets)
end

"""
    accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)

Compute the accuracy of a classification model given the predicted outputs and the ground truth targets.

# Arguments
- `outputs`: An `AbstractArray` of predicted outputs, where each column represents the predicted output for a single sample.
- `targets`: An `AbstractArray` of ground truth targets, where each column represents the target for a single sample.
- `threshold`: A `Real` value representing the threshold for classification. Outputs above this threshold are considered as positive.

# Returns
- `accuracy`: A `Float64` value representing the accuracy of the classification model.
"""
function accuracy(
    outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2};
    threshold::Real = 0.5,
)

    if size(outputs, 2) == 1
        return accuracy(outputs, targets, threshold)
    else
        outputs = classifyOutputs(outputs)
        return accuracy(outputs, targets)
    end
end

"""
    buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

Builds a feedforward artificial neural network (ANN) for classification tasks.

# Arguments
- `numInputs::Int`: The number of input features.
- `topology::AbstractArray{<:Int,1}`: An array specifying the number of neurons in each hidden layer.
- `numOutputs::Int`: The number of output classes.
- `transferFunctions::AbstractArray{<:Function,1}`: (optional) An array of activation functions for each hidden layer. Default is `σ` (sigmoid function) for all layers.

# Returns
- `ann`: The constructed ANN model.
"""
function buildClassANN(
    numInputs::Int,
    topology::AbstractArray{<:Int,1},
    numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
)
    ann = Chain()
    numInputsLayer = numInputs

    for numHiddenLayer = 1:length(topology)
        neurons = topology[numHiddenLayer]
        ann =
            Chain(ann..., Dense(numInputsLayer, neurons, transferFunctions[numHiddenLayer]))
        numInputsLayer = neurons
    end

    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end

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

# Split dataset into training, validation, and test sets with 60% for training, 20% for validation, and 20% for testing
N = 10
train_indexes, val_indexes, test_indexes = holdOut(N, 0.2, 0.1)

"""
Calculates the confusion matrix and performance metrics for a multiclass classification problem.

# Arguments
- `outputs::AbstractArray{Bool,2}`: A matrix of predicted outputs, where each row corresponds to a sample and each column corresponds to a class. The values should be binary (true/false) indicating whether the sample belongs to the corresponding class.
- `targets::AbstractArray{Bool,2}`: A matrix of target outputs, where each row corresponds to a sample and each column corresponds to a class. The values should be binary (true/false) indicating whether the sample belongs to the corresponding class.
- `weighted::Bool=true`: A boolean indicating whether to calculate the weighted or unweighted performance metrics. If `weighted=true`, the metrics are weighted by the number of samples in each class.

# Returns
A named tuple containing the following performance metrics:
- `accuracy`: The overall accuracy of the classification.
- `errorRate`: The overall error rate of the classification.
- `sensitivity`: A vector of sensitivities (true positive rate) for each class.
- `specificity`: A vector of specificities (true negative rate) for each class.
- `precision`: A vector of precisions (positive predictive value) for each class.
- `negative_predictive_value`: A vector of negative predictive values for each class.
- `fScore`: A vector of F1 scores for each class.
- `confusion_matrix`: The confusion matrix, where each row corresponds to a predicted class and each column corresponds to a true class.
"""

function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}} = (
        Array{eltype(trainingDataset[1]),2}(undef, 0, 0),
        falses(0, 0),
    ),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}} = (
        Array{eltype(trainingDataset[1]),2}(undef, 0, 0),
        falses(0, 0),
    ),
    transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
    maxEpochs::Int = 1000,
    minLoss::Real = 0.0,
    learningRate::Real = 0.01,
    maxEpochsVal::Int = 20,
    showText::Bool = false,
)

    # Split into inputs/targets
    trainingInputs, trainingTargets = trainingDataset
    validationInputs, validationTargets = validationDataset
    testInputs, testTargets = testDataset

    useValidation = length(validationInputs) > 0
    useTest = length(testInputs) > 0

    # Check that the targets corresponding to the inputs have the same number of samples
    @assert (size(trainingInputs, 1) == size(trainingTargets, 1)) "Number of training inputs and targets do not match"
    @assert (size(validationInputs, 1) == size(validationTargets, 1)) "Number of validation inputs and targets do not match"
    @assert (size(testInputs, 1) == size(testTargets, 1)) "Number of test inputs and targets do not match"

    if (useValidation)
        @assert (size(trainingInputs, 2) == size(validationInputs, 2)) "Number of attributes for training and validation do not match"
    end
    if (size(testInputs, 1) != 0)
        @assert (size(trainingInputs, 2) == size(testInputs, 2)) "Number of attributes for training and test do not match"
    end

    # Build the network
    nInputs, nOutputs = size(trainingInputs, 2), size(trainingTargets, 2)
    ann = buildClassANN(nInputs, topology, nOutputs; transferFunctions)

    if showText
        println("ANN network built: $ann")
    end

    # Loss
    loss(x, y) =
        (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) :
        Losses.crossentropy(ann(x), y)
    # Metric progress
    trainingLosses = Array{Float32}(undef, 0)
    validationLosses = Array{Float32}(undef, 0)
    testLosses = Array{Float32}(undef, 0)
    trainingAccs = Array{Float32}(undef, 0)
    validationAccs = Array{Float32}(undef, 0)
    testAccs = Array{Float32}(undef, 0)

    # Train for n=max_epochs (at most) epochs
    currentEpoch = 0

    # Calculate, store and print last loss/accuracy
    function calculateMetrics()

        # Losses
        trainingLoss = loss(trainingInputs', trainingTargets')
        validationLoss =
            (size(validationInputs, 1) != 0) ? loss(validationInputs', validationTargets') :
            0
        testLoss = (size(testInputs, 1) != 0) ? loss(testInputs', testTargets') : 0

        # Accuracies
        trainingOutputs = ann(trainingInputs')

        validationAcc = 0
        testAcc = 0

        if useValidation
            validationOutputs = ann(validationInputs')
            if size(validationOutputs, 1) == 1
                validationAcc = accuracy(vec(validationOutputs'), vec(validationTargets))
            else
                validationAcc = accuracy(validationOutputs', validationTargets)
            end
        end

        if useTest
            testOutputs = ann(testInputs')
            if size(testOutputs, 1) == 1
                testAcc = accuracy(vec(testOutputs'), vec(testTargets))
            else
                testAcc = accuracy(testOutputs', testTargets)
            end
        end

        if size(trainingOutputs, 1) == 1
            trainingAcc = accuracy(vec(trainingOutputs'), vec(trainingTargets))
        else
            trainingAcc = accuracy(trainingOutputs', trainingTargets)
        end

        # Update the history of losses and accuracies
        push!(trainingLosses, trainingLoss)
        push!(validationLosses, validationLoss)
        push!(testLosses, testLoss)
        push!(trainingAccs, trainingAcc)
        push!(validationAccs, validationAcc)
        push!(testAccs, testAcc)

        # Show text
        if showText && (currentEpoch % 50 == 0)
            println(
                "Epoch ",
                currentEpoch,
                ": \n\tTraining loss: ",
                trainingLoss,
                ", accuracy: ",
                100 * trainingAcc,
                "% \n\tValidation loss: ",
                validationLoss,
                ", accuracy: ",
                100 * validationAcc,
                "% \n\tTest loss: ",
                testLoss,
                ", accuracy: ",
                100 * testAcc,
                "%",
            )
        end

        return trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc
    end

    # Compute and store initial metrics
    trainingLoss, _, validationLoss, _, _, _ = calculateMetrics()

    # Best model at validation set
    numEpochsValidation = 0
    bestValidationLoss = validationLoss

    if (useValidation)
        bestAnn = deepcopy(ann)
    else
        bestAnn = ann  # if no validation, we want to return the ANN that is trained in every cycle
    end

    # Start the training

    while (currentEpoch < maxEpochs) &&
              (trainingLoss > minLoss) &&
              (numEpochsValidation < maxEpochsVal)

        # Update epoch number
        currentEpoch += 1

        # Fit the model
        Flux.train!(
            loss,
            Flux.params(ann),
            [(trainingInputs', trainingTargets')],
            ADAM(learningRate),
        )

        # Compute and store metrics
        trainingLoss, _, validationLoss, _, _, _ = calculateMetrics()

        # Update validation early stopping only if validation set given
        if (useValidation)
            if (validationLoss < bestValidationLoss)
                bestValidationLoss = validationLoss
                numEpochsValidation = 0
                bestAnn = deepcopy(ann)
            else
                numEpochsValidation += 1
            end
        end

    end

    # Print stop reason and final metrics

    if showText
        println(
            "Final results for epoch $currentEpoch:
        \n\tTraining loss: $(trainingLosses[end]), accuracy: $(100 * trainingAccs[end])%
        \n\tValidation loss: $(validationLosses[end]), accuracy: $(100 * validationAccs[end])%
        \n\tTest loss: $(testLosses[end]), accuracy: $(100 * testAccs[end])%",
        )

        println("\nStopping criteria:
            \n\tMax. epochs: $(currentEpoch >= maxEpochs)
            \n\tMin. loss: $(trainingLoss <= minLoss)
            \n\tNum. epochs validation: $(numEpochsValidation >= maxEpochsVal)")
    end

    return bestAnn,
    trainingLosses,
    validationLosses,
    testLosses,
    trainingAccs,
    validationAccs,
    testAccs
end

"""
    trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

Train a classification artificial neural network (ANN) using the specified topology and training dataset.

# Arguments
- `topology::AbstractArray{<:Int,1}`: The topology of the ANN, specifying the number of neurons in each layer.
- `trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}`: The training dataset, consisting of input features and corresponding target labels.
- `validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}`: The validation dataset (optional), used for early stopping and model selection.
- `testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}`: The test dataset (optional), used for evaluating the trained model.
- `transferFunctions::AbstractArray{<:Function,1}`: The transfer functions to be used in each layer of the ANN (optional).
- `maxEpochs::Int`: The maximum number of training epochs (optional).
- `minLoss::Real`: The minimum loss value to achieve before stopping training (optional).
- `learningRate::Real`: The learning rate used in the gradient descent optimization algorithm (optional).
- `maxEpochsVal::Int`: The maximum number of epochs to wait for validation loss improvement before stopping training (optional).
- `showText::Bool`: Whether to display training progress and results in the console (optional).

# Returns
- The trained classification ANN model.

"""
function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}} = (
        Array{eltype(trainingDataset[1]),2}(undef, 0, 0),
        falses(0),
    ),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}} = (
        Array{eltype(trainingDataset[1]),2}(undef, 0, 0),
        falses(0),
    ),
    transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)),
    maxEpochs::Int = 1000,
    minLoss::Real = 0.0,
    learningRate::Real = 0.01,
    maxEpochsVal::Int = 20,
    showText::Bool = false,
)

    trainingInputs, trainingTargets = trainingDataset
    testInputs, testTargets = testDataset
    validationInputs, validationTargets = validationDataset
    testTargets = reshape(testTargets, :, 1)
    validationTargets = reshape(validationTargets, :, 1)

    trainingTargets = reshape(trainingTargets, :, 1)

    is_val_empty = size(validationDataset[1]) == (0, 0)
    is_test_empty = size(testDataset[1]) == (0, 0)

    return trainClassANN(
        topology,
        (trainingInputs, trainingTargets);
        validationDataset = (validationInputs, validationTargets),
        testDataset = (testInputs, testTargets),
        transferFunctions = transferFunctions,
        maxEpochs = maxEpochs,
        minLoss = minLoss,
        learningRate = learningRate,
        maxEpochsVal = maxEpochsVal,
        showText = showText,
    )

end

"""
    confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

Compute the confusion matrix and various evaluation metrics based on the predicted outputs and target labels.

# Arguments
- `outputs::AbstractArray{Bool,1}`: The predicted outputs of a classification model.
- `targets::AbstractArray{Bool,1}`: The target labels for the corresponding inputs.

# Returns
- A named tuple containing the following evaluation metrics:
    - `accuracy`: The accuracy of the model.
    - `errorRate`: The error rate of the model.
    - `sensitivity`: The sensitivity (true positive rate) of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `fScore`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2x2 array.

"""
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    @assert(length(outputs) == length(targets))

    tp = sum(outputs .& targets)
    tn = sum(.!outputs .& .!targets)
    fp = sum(outputs .& .!targets)
    fn = sum(.!outputs .& targets)

    nPatterns = length(targets)

    accuracy = (tp + tn) / nPatterns
    errorRate = 1 - accuracy

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    precision = tp / (tp + fp)
    negative_predictive_value = tn / (tn + fn)

    if isnan(sensitivity) && isnan(precision)
        sensitivity = 1.0
        precision = 1.0
    elseif isnan(specificity) && isnan(negative_predictive_value)
        specificity = 1.0
        negative_predictive_value = 1.0
    end

    sensitivity = isnan(sensitivity) ? 0.0 : sensitivity
    precision = isnan(precision) ? 0.0 : precision
    specificity = isnan(specificity) ? 0.0 : specificity
    negative_predictive_value =
        isnan(negative_predictive_value) ? 0.0 : negative_predictive_value

    confusion_matrix = [tp fp; fn tn]

    fScore =
        (precision == sensitivity == 0) ? 0 :
        2 * (precision * sensitivity) / (precision + sensitivity)

    return (
        accuracy = accuracy,
        errorRate = errorRate,
        sensitivity = sensitivity,
        specificity = specificity,
        precision = precision,
        negative_predictive_value = negative_predictive_value,
        fScore = fScore,
        confusion_matrix = confusion_matrix,
    )
end

"""
    confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

Compute the confusion matrix and various evaluation metrics based on the predicted outputs and target labels.

# Arguments
- `outputs::AbstractArray{<:Real,1}`: The predicted outputs of a classification model.
- `targets::AbstractArray{Bool,1}`: The target labels for the corresponding inputs.
- `threshold::Real`: The threshold value used to convert the continuous outputs to binary predictions (optional, default is 0.5).

# Returns
- A named tuple containing the following evaluation metrics:
    - `accuracy`: The accuracy of the model.
    - `errorRate`: The error rate of the model.
    - `sensitivity`: The sensitivity (true positive rate) of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `fScore`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2x2 array.

"""
function confusionMatrix(
    outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1};
    threshold::Real = 0.5,
)
    outputs = outputs .> threshold
    return confusionMatrix(outputs, targets)
end

"""
    oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})

Train multiple binary classifiers using the one-vs-all strategy and generate the corresponding outputs for each class.

# Arguments
- `inputs::AbstractArray{<:Real,2}`: The input features for the classification task.
- `targets::AbstractArray{Bool,2}`: The target labels for the corresponding inputs.

# Returns
- A 2-dimensional matrix of binary outputs, where each column represents the predicted outputs for a specific class.

"""
function oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    numInstances, numClasses = size(targets)

    # Create a 2-dimensional matrix of real values, with as many rows as patterns and as many columns as classes
    outputs = Array{Float32,2}(undef, numInstances, numClasses)

    # Make a loop that iterates over each class
    for numClass = 1:numClasses
        # Create the desired outputs corresponding to that class
        desiredOutputs = targets[:, numClass]

        # Train the corresponding model with those inputs and the new desired outputs corresponding to that class
        model, _ = trainClassANN([2, 2], (inputs, desiredOutputs))

        # Apply the model to the inputs to calculate the outputs, which will be copied into the previously created matrix
        outputs[:, numClass] .= model(inputs')'
    end

    vmax = maximum(outputs, dims = 2)
    outputs = (outputs .== vmax)

    return outputs
end

"""
    confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

Compute the confusion matrix and various evaluation metrics based on the predicted outputs and target labels for multi-class classification.

# Arguments
- `outputs::AbstractArray{Bool,2}`: The predicted outputs of a multi-class classification model.
- `targets::AbstractArray{Bool,2}`: The target labels for the corresponding inputs.
- `weighted::Bool`: Whether to compute weighted metrics based on class imbalance (optional, default is true).

# Returns
- A named tuple containing the following evaluation metrics:
    - `accuracy`: The accuracy of the model.
    - `errorRate`: The error rate of the model.
    - `sensitivity`: The sensitivity (true positive rate) of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `fScore`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2-dimensional array.

"""
function confusionMatrix(
    outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2};
    weighted::Bool = true,
)
    @assert size(outputs) == size(targets)
    @assert size(outputs, 2) > 1

    numClasses = size(outputs, 2)
    sensitivity = zeros(numClasses)
    specificity = zeros(numClasses)
    precision = zeros(numClasses)
    negative_predictive_value = zeros(numClasses)
    fScore = zeros(numClasses)
    confusion_matrix = zeros(Int, numClasses, numClasses)

    for i = 1:numClasses
        classOutputs = outputs[:, i]
        classTargets = targets[:, i]
        if any(classTargets)
            classMetrics = confusionMatrix(classOutputs, classTargets)
            sensitivity[i] = classMetrics.sensitivity
            specificity[i] = classMetrics.specificity
            precision[i] = classMetrics.precision
            negative_predictive_value[i] = classMetrics.negative_predictive_value
            fScore[i] = classMetrics.fScore
        end

        for j = 1:numClasses
            confusion_matrix[i, j] = sum(targets[:, i] .& outputs[:, j])
        end
    end

    numInstancesFromEachClass = vec(sum(targets, dims = 1))

    if weighted
        weights = numInstancesFromEachClass ./ size(targets, 1)
        sensitivity = sum(weights .* sensitivity)
        specificity = sum(weights .* specificity)
        precision = sum(weights .* precision)
        negative_predictive_value = sum(weights .* negative_predictive_value)
        fScore = sum(weights .* fScore)
    else
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        precision = mean(precision)
        negative_predictive_value = mean(negative_predictive_value)
        fScore = mean(fScore)
    end

    acc = accuracy(targets, outputs)
    errorRate = 1 - acc

    return (
        accuracy = acc,
        errorRate = errorRate,
        sensitivity = sensitivity,
        specificity = specificity,
        precision = precision,
        negative_predictive_value = negative_predictive_value,
        fScore = fScore,
        confusion_matrix = confusion_matrix,
    )
end

"""
    confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

Compute the confusion matrix and various evaluation metrics based on the predicted outputs and target labels for multi-class classification.

# Arguments
- `outputs::AbstractArray{<:Real,2}`: The predicted outputs of a multi-class classification model.
- `targets::AbstractArray{Bool,2}`: The target labels for the corresponding inputs.
- `weighted::Bool`: Whether to compute weighted metrics based on class imbalance (optional, default is true).

# Returns
- A named tuple containing the following evaluation metrics:
    - `accuracy`: The accuracy of the model.
    - `errorRate`: The error rate of the model.
    - `sensitivity`: The sensitivity (true positive rate) of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `fScore`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2-dimensional array.

"""
function confusionMatrix(
    outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2};
    weighted::Bool = true,
)
    boolOutputs = classifyOutputs(outputs)
    return confusionMatrix(boolOutputs, targets, weighted = weighted)
end

"""
    confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)

Compute the confusion matrix and various evaluation metrics based on the predicted outputs and target labels for multi-class classification.

# Arguments
- `outputs::AbstractArray{<:Any,1}`: The predicted outputs of a multi-class classification model.
- `targets::AbstractArray{<:Any,1}`: The target labels for the corresponding inputs.
- `weighted::Bool`: Whether to compute weighted metrics based on class imbalance (optional, default is true).

# Returns
- A named tuple containing the following evaluation metrics:
    - `accuracy`: The accuracy of the model.
    - `errorRate`: The error rate of the model.
    - `sensitivity`: The sensitivity (true positive rate) of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `fScore`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2-dimensional array.

"""
function confusionMatrix(
    outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1};
    weighted::Bool = true,
)
    @assert(all([in(output, unique(targets)) for output in outputs]))

    classes = unique(targets)
    return confusionMatrix(
        oneHotEncoding(outputs, classes),
        oneHotEncoding(targets, classes),
    )

    return confusionMatrix(boolOutputs, boolTargets, weighted = weighted)
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
end;

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

"""
    create_model(modelType::Symbol, modelHyperparameters::Dict)

Create a machine learning model based on the specified model type and hyperparameters.

# Arguments
- `modelType::Symbol`: The type of the machine learning model to create. Supported types are :SVM, :kNN, and :DecisionTree.
- `modelHyperparameters::Dict`: A dictionary containing the hyperparameters for the model.

# Returns
- An instance of the specified machine learning model.

"""
function create_model(modelType::Symbol, modelHyperparameters::Dict)
    if modelType == :SVM
        return SVC(
            kernel = modelHyperparameters["kernel"],
            degree = get(modelHyperparameters, "degree", 3),
            gamma = get(modelHyperparameters, "gamma", "scale"),
            C = modelHyperparameters["C"],
            class_weight = get(modelHyperparameters, "class_weight", nothing),
        )
    elseif modelType == :kNN
        return KNeighborsClassifier(modelHyperparameters["numNeighboors"])
    elseif modelType == :DecisionTree
        return DecisionTreeClassifier(
            max_depth = modelHyperparameters["maxDepth"],
            class_weight = get(modelHyperparameters, "class_weight", nothing),
            random_state = 42,
        )
    else
        error("Model type not supported")
    end
end

"""
    train_ann_model(modelHyperparameters, inputs, targets, testInputs, testTargets)

Train an artificial neural network (ANN) model based on the specified hyperparameters and evaluate its performance on the test set.

# Arguments
- `modelHyperparameters`: A dictionary containing the hyperparameters for the ANN model.
- `inputs`: The input features for training the ANN model.
- `targets`: The target labels for training the ANN model.
- `testInputs`: The input features for evaluating the ANN model.
- `testTargets`: The target labels for evaluating the ANN model.

# Returns
- A named tuple containing the evaluation metrics of the trained ANN model on the test set:
    - `accuracy`: The accuracy of the model.
    - `error_rate`: The error rate of the model.
    - `recall`: The recall (true positive rate) of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `f1_score`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2-dimensional array.

"""
function train_ann_model(modelHyperparameters, inputs, targets, testInputs, testTargets)
    testAccuraciesForEachRepetition =
        Array{Float64,1}(undef, modelHyperparameters["repetitions"])
    testRecallForEachRepetition =
        Array{Float64,1}(undef, modelHyperparameters["repetitions"])
    testErrorRateForEachRepetition =
        Array{Float64,1}(undef, modelHyperparameters["repetitions"])
    testSpecificityForEachRepetition =
        Array{Float64,1}(undef, modelHyperparameters["repetitions"])
    testPrecisionForEachRepetition =
        Array{Float64,1}(undef, modelHyperparameters["repetitions"])
    testNegative_predictive_valueForEachRepetition =
        Array{Float64,1}(undef, modelHyperparameters["repetitions"])
    testfScoreForEachRepetition =
        Array{Float64,1}(undef, modelHyperparameters["repetitions"])
    testConfusionMatrix = []

    for numTraining = 1:modelHyperparameters["repetitions"]
        if modelHyperparameters["validationRatio"] > 0.0
            trainingInputs, trainingTargets, validationInputs, validationTargets =
                splitTrainAndValidation(
                    inputs,
                    targets,
                    modelHyperparameters["validationRatio"],
                )
            model, _ = trainClassANN(
                modelHyperparameters["topology"],
                (trainingInputs, trainingTargets);
                validationDataset = (validationInputs, validationTargets),
                testDataset = (testInputs, testTargets),
                transferFunctions = modelHyperparameters["transferFunctions"],
                maxEpochs = modelHyperparameters["maxEpochs"],
                learningRate = modelHyperparameters["learningRate"],
                maxEpochsVal = modelHyperparameters["maxEpochsVal"],
            )

        else
            model, _ = trainClassANN(
                modelHyperparameters["topology"],
                (inputs, targets);
                testDataset = (testInputs, testTargets),
                transferFunctions = modelHyperparameters["transferFunctions"],
                maxEpochs = modelHyperparameters["maxEpochs"],
                learningRate = modelHyperparameters["learningRate"],
                maxEpochsVal = modelHyperparameters["maxEpochsVal"],
            )
        end

        testOutputs = model(testInputs')'
        testAccuraciesForEachRepetition[numTraining],
        testErrorRateForEachRepetition[numTraining],
        testRecallForEachRepetition[numTraining],
        testSpecificityForEachRepetition[numTraining],
        testPrecisionForEachRepetition[numTraining],
        testNegative_predictive_valueForEachRepetition[numTraining],
        testfScoreForEachRepetition[numTraining],
        testConfusionMatrix =
            (size(testTargets, 2) == 1) ?
            confusionMatrix(vec(testOutputs), vec(testTargets)) :
            confusionMatrix(testOutputs, testTargets)

    end

    return (
        accuracy = mean(testAccuraciesForEachRepetition),
        error_rate = mean(testErrorRateForEachRepetition),
        recall = mean(testRecallForEachRepetition),
        specificity = mean(testSpecificityForEachRepetition),
        precision = mean(testPrecisionForEachRepetition),
        negative_predictive_value = mean(testNegative_predictive_valueForEachRepetition),
        f1_score = mean(testfScoreForEachRepetition),
        confusion_matrix = testConfusionMatrix,
    )
end

function modelCrossValidation(
    modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1},
)
    @assert(size(inputs, 1) == length(targets))
    @assert (in(modelType, [:ANN, :SVM, :kNN, :DecisionTree])) "Model type $(modelType) is not supported"

    Random.seed!(42)

    kFolds = maximum(crossValidationIndices)
    testAccuracies = Array{Float64,1}(undef, kFolds)
    testRecalls = Array{Float64,1}(undef, kFolds)
    testErrorRates = Array{Float64,1}(undef, kFolds)
    testSpecificities = Array{Float64,1}(undef, kFolds)
    testPrecisions = Array{Float64,1}(undef, kFolds)
    testNPVs = Array{Float64,1}(undef, kFolds)
    testfScores = Array{Float64,1}(undef, kFolds)

    if modelType == :ANN
        targets = oneHotEncoding(targets)
    end

    for numFold = 1:kFolds
        trainingInputs = inputs[crossValidationIndices.!=numFold, :]
        trainingTargets = targets[crossValidationIndices.!=numFold, :]

        testInputs = inputs[crossValidationIndices.==numFold, :]
        testTargets = targets[crossValidationIndices.==numFold, :]

        if modelType != :ANN
            model = create_model(modelType, modelHyperparameters)
            model, _ = train_and_predict(
                model,
                trainingInputs,
                trainingTargets,
                testInputs,
                testTargets,
            )

            testOutputs = predict(model, testInputs)
            # testOutputs = oneHotEncoding(testOutputs, unique(testTargets))
            # testTargets = oneHotEncoding(vec(testTargets))
            # println(testOutputs)
            testAccuracy,
            testErrorRate,
            testRecall,
            testSpecificity,
            testPrecision,
            testNPV,
            testfScore,
            _ = confusionMatrix(testOutputs, vec(testTargets))
        else
            testAccuracy,
            testErrorRate,
            testRecall,
            testSpecificity,
            testPrecision,
            testNPV,
            testfScore,
            _ = train_ann_model(
                modelHyperparameters,
                trainingInputs,
                trainingTargets,
                testInputs,
                testTargets,
            )
        end

        testAccuracies[numFold] = testAccuracy
        testRecalls[numFold] = testRecall
        testErrorRates[numFold] = testErrorRate
        testSpecificities[numFold] = testSpecificity
        testPrecisions[numFold] = testPrecision
        testNPVs[numFold] = testNPV
        testfScores[numFold] = testfScore
    end

    return Dict{String,Any}(
        "accuracy" => mean(testAccuracies),
        "std_accuracy" => std(testAccuracies),
        "recall" => mean(testRecalls),
        "std_recall" => std(testRecalls),
        "specificity" => mean(testSpecificities),
        "std_specificity" => std(testSpecificities),
        "precision" => mean(testPrecisions),
        "std_precision" => std(testPrecisions),
        "f1_score" => mean(testfScores),
        "std_f1_score" => std(testfScores),
    )
end

"""
    createAndTrainFinalModel(
        modelType::Symbol,
        modelHyperparameters::Dict,
        trainingInputs::AbstractArray{<:Real,2},
        trainingTargets::AbstractArray{<:Any,1},
        testInputs::AbstractArray{<:Real,2},
        testTargets::AbstractArray{<:Any,1}
    )

Create and train the final machine learning model based on the specified model type, hyperparameters, training inputs, training targets, test inputs, and test targets.

# Arguments
- `modelType::Symbol`: The type of the machine learning model to create and train. Supported types are :ANN, :SVM, :kNN, and :DecisionTree.
- `modelHyperparameters::Dict`: A dictionary containing the hyperparameters for the model.
- `trainingInputs::AbstractArray{<:Real,2}`: The input features for training the model.
- `trainingTargets::AbstractArray{<:Any,1}`: The target labels for training the model.
- `testInputs::AbstractArray{<:Real,2}`: The input features for evaluating the model.
- `testTargets::AbstractArray{<:Any,1}`: The target labels for evaluating the model.

# Returns
- A dictionary containing the evaluation metrics of the trained model on the test set:
    - `accuracy`: The accuracy of the model.
    - `recall`: The recall (true positive rate) of the model.
    - `errorRate`: The error rate of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `f1_score`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2-dimensional array.

"""
function createAndTrainFinalModel(
    modelType::Symbol,
    modelHyperparameters::Dict,
    trainingInputs::AbstractArray{<:Real,2},
    trainingTargets::AbstractArray{<:Any,1},
    testInputs::AbstractArray{<:Real,2},
    testTargets::AbstractArray{<:Any,1},
)
    @assert(size(trainingInputs, 1) == length(trainingTargets))
    @assert(size(testInputs, 1) == length(testTargets))
    @assert (in(modelType, [:ANN, :SVM, :kNN, :DecisionTree])) "Model type $(modelType) is not supported"

    Random.seed!(42)

    if modelType == :ANN
        trainingTargets = oneHotEncoding(trainingTargets)
        testTargets = oneHotEncoding(testTargets)
        modelHyperparameters["repetitions"] = 1
        testAccuracy,
        testErrorRate,
        testRecall,
        testSpecificity,
        testPrecision,
        testNegativePredictiveValue,
        testfScore,
        testConfusionMatrix = train_ann_model(
            modelHyperparameters,
            trainingInputs,
            trainingTargets,
            testInputs,
            testTargets,
        )
    else
        model = create_model(modelType, modelHyperparameters)
        model,
        testAccuracy,
        testErrorRate,
        testRecall,
        testSpecificity,
        testPrecision,
        testNegativePredictiveValue,
        testfScore,
        testConfusionMatrix = train_and_predict(
            model,
            trainingInputs,
            trainingTargets,
            testInputs,
            testTargets,
        )
    end


    return Dict{String,Any}(
        "accuracy" => testAccuracy,
        "recall" => testRecall,
        "errorRate" => testErrorRate,
        "specificity" => testSpecificity,
        "precision" => testPrecision,
        "negative_predictive_value" => testNegativePredictiveValue,
        "f1_score" => testfScore,
        "confusion_matrix" => testConfusionMatrix,
    )
end

"""
    createAndTrainFinalEnsemble(
        estimators::AbstractArray{Symbol,1},
        modelsHyperParameters::AbstractArray{<:AbstractDict, 1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}},
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}};
        ensembleType::Symbol = :VotingHard,
        final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)
    )

Create and train the final ensemble model based on the specified estimators, models hyperparameters, training dataset, and test dataset.

# Arguments
- `estimators::AbstractArray{Symbol,1}`: An array of symbols representing the types of models to be used as estimators.
- `modelsHyperParameters::AbstractArray{<:AbstractDict, 1}`: An array of dictionaries containing the hyperparameters for each model.
- `trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}}`: A tuple containing the training inputs and training targets.
- `testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}}`: A tuple containing the test inputs and test targets.
- `ensembleType::Symbol = :VotingHard`: The type of ensemble to create. Supported types are :VotingHard and :VotingSoft.
- `final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)`: The hyperparameters for the final estimator model.

# Returns
- A dictionary containing the evaluation metrics of the trained ensemble model on the test set:
    - `accuracy`: The accuracy of the model.
    - `recall`: The recall (true positive rate) of the model.
    - `errorRate`: The error rate of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `f1_score`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2-dimensional array.

"""
function createAndTrainFinalEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:AbstractDict,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,2}},
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,2}};
    ensembleType::Symbol = :VotingHard,
    final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1),
)

    Random.seed!(42)

    trainingInputs, trainingTargets = trainingDataset
    testInputs, testTargets = testDataset

    models = train_models(
        estimators,
        modelsHyperParameters,
        trainingInputs,
        trainingTargets,
        testInputs,
        testTargets,
    )

    # ensembleModel = StackingClassifier(estimators=models,
    #     final_estimator=create_model(final_estimator["modelType"], final_estimator), n_jobs=1)

    ensembleModel = create_ensemble(ensembleType, models, final_estimator)

    ensembleModel,
    testAccuracy,
    testErrorRate,
    testRecall,
    testSpecificity,
    testPrecision,
    testNegativePredictiveValue,
    testfScore,
    testConfusionMatrix = train_and_predict(
        ensembleModel,
        trainingInputs,
        trainingTargets,
        testInputs,
        testTargets,
    )


    return Dict{String,Any}(
        "accuracy" => testAccuracy,
        "recall" => testRecall,
        "errorRate" => testErrorRate,
        "specificity" => testSpecificity,
        "precision" => testPrecision,
        "negative_predictive_value" => testNegativePredictiveValue,
        "f1_score" => testfScore,
        "confusion_matrix" => testConfusionMatrix,
    )
end

"""
    createAndTrainFinalEnsemble(
        estimators::AbstractArray{Symbol,1},
        modelsHyperParameters::AbstractArray{<:AbstractDict, 1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}};
        ensembleType::Symbol = :VotingHard,
        final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)
    )

Create and train the final ensemble model based on the specified estimators, models hyperparameters, training dataset, and test dataset.

# Arguments
- `estimators::AbstractArray{Symbol,1}`: An array of symbols representing the types of models to be used as estimators.
- `modelsHyperParameters::AbstractArray{<:AbstractDict, 1}`: An array of dictionaries containing the hyperparameters for each model.
- `trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}`: A tuple containing the training inputs and training targets.
- `testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}`: A tuple containing the test inputs and test targets.
- `ensembleType::Symbol = :VotingHard`: The type of ensemble to create. Supported types are :VotingHard and :VotingSoft.
- `final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)`: The hyperparameters for the final estimator model.

# Returns
- A dictionary containing the evaluation metrics of the trained ensemble model on the test set:
    - `accuracy`: The accuracy of the model.
    - `recall`: The recall (true positive rate) of the model.
    - `errorRate`: The error rate of the model.
    - `specificity`: The specificity (true negative rate) of the model.
    - `precision`: The precision of the model.
    - `negative_predictive_value`: The negative predictive value of the model.
    - `f1_score`: The F1 score of the model.
    - `confusion_matrix`: The confusion matrix as a 2-dimensional array.

"""
function createAndTrainFinalEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:AbstractDict,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}},
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}};
    ensembleType::Symbol = :VotingHard,
    final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1),
)

    trainingInputs, trainingTargets = trainingDataset
    testInputs, testTargets = testDataset

    trainingTargets = reshape(trainingTargets, :, 1)
    testTargets = reshape(testTargets, :, 1)

    return createAndTrainFinalEnsemble(
        estimators,
        modelsHyperParameters,
        (trainingInputs, trainingTargets),
        (testInputs, testTargets);
        ensembleType = ensembleType,
        final_estimator = final_estimator,
    )

end

"""
    train_models(
        models::AbstractArray{Symbol, 1},
        hyperParameters::AbstractArray{<:AbstractDict, 1},
        trainingInputs,
        trainingTargets,
        testInputs,
        testTargets
    )

Train multiple machine learning models based on the specified models, hyperparameters, training inputs, training targets, test inputs, and test targets.

# Arguments
- `models::AbstractArray{Symbol, 1}`: An array of symbols representing the types of models to be trained.
- `hyperParameters::AbstractArray{<:AbstractDict, 1}`: An array of dictionaries containing the hyperparameters for each model.
- `trainingInputs`: The input features for training the models.
- `trainingTargets`: The target labels for training the models.
- `testInputs`: The input features for evaluating the models.
- `testTargets`: The target labels for evaluating the models.

# Returns
- An array of tuples, where each tuple contains the name and trained model.

"""
function train_models(
    models::AbstractArray{Symbol,1},
    hyperParameters::AbstractArray{<:AbstractDict,1},
    trainingInputs,
    trainingTargets,
    testInputs,
    testTargets,
)
    Random.seed!(42)

    ensemble_models = []
    @assert length(models) == length(hyperParameters)

    for i = 1:length(hyperParameters)
        modelType = models[i]
        hyperParameter = hyperParameters[i]

        model = create_model(modelType, hyperParameter)
        fit!(model, trainingInputs, vec(trainingTargets))
        name = "model_$(i)"

        push!(ensemble_models, (name, model))
    end

    return ensemble_models
end

"""
    train_and_predict(
        model,
        trainingInputs,
        trainingTargets,
        testInputs,
        testTargets
    )

Train a machine learning model using the provided training inputs and targets, and then predict the outputs for the test inputs.

# Arguments
- `model`: The machine learning model to train and predict with.
- `trainingInputs`: The input features for training the model.
- `trainingTargets`: The target labels for training the model.
- `testInputs`: The input features for predicting with the trained model.
- `testTargets`: The target labels for evaluating the predictions.

# Returns
- A tuple containing the trained model and the evaluation metrics of the predictions on the test set:
    - `testAccuracy`: The accuracy of the model.
    - `testErrorRate`: The error rate of the model.
    - `testRecall`: The recall (true positive rate) of the model.
    - `testSpecificity`: The specificity (true negative rate) of the model.
    - `testPrecision`: The precision of the model.
    - `testNegative_predictive_value`: The negative predictive value of the model.
    - `testfScore`: The F1 score of the model.
    - `testConfusionMatrix`: The confusion matrix as a 2-dimensional array.

"""
function train_and_predict(model, trainingInputs, trainingTargets, testInputs, testTargets)
    Random.seed!(42)

    fit!(model, trainingInputs, vec(trainingTargets))
    testOutputs = predict(model, testInputs)

    testAccuracy,
    testErrorRate,
    testRecall,
    testSpecificity,
    testPrecision,
    testNegative_predictive_value,
    testfScore,
    testConfusionMatrix = confusionMatrix(testOutputs, vec(testTargets))
    # accuracy, errorRate, sensitivity, specificity, precision, negative_predictive_value, fScore, confusion_matrix

    return model,
    testAccuracy,
    testErrorRate,
    testRecall,
    testSpecificity,
    testPrecision,
    testNegative_predictive_value,
    testfScore,
    testConfusionMatrix
end

"""
    create_ensemble(
        ensemble_type::Symbol,
        estimators::AbstractArray{<:Any},
        final_estimator::AbstractDict{<:Any}
    )

Create an ensemble model based on the specified ensemble type, estimators, and final estimator.

# Arguments
- `ensemble_type::Symbol`: The type of ensemble to create. Supported types are :VotingHard and :Stacking.
- `estimators::AbstractArray{<:Any}`: An array of estimators to be used in the ensemble.
- `final_estimator::AbstractDict{<:Any}`: The hyperparameters for the final estimator model.

# Returns
- An instance of the specified ensemble model.

"""
function create_ensemble(
    ensemble_type::Symbol,
    estimators::AbstractArray{<:Any},
    final_estimator::AbstractDict{<:Any},
)

    if ensemble_type == :VotingHard
        return VotingClassifier(estimators = estimators, voting = "hard", n_jobs = 1)
    elseif ensemble_type == :Stacking
        return StackingClassifier(
            estimators = estimators,
            final_estimator = create_model(final_estimator["modelType"], final_estimator),
            n_jobs = 1,
        )
    end
end

"""
    trainClassEnsemble(
        estimators::AbstractArray{Symbol,1},
        modelsHyperParameters::AbstractArray{<:AbstractDict, 1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}},
        kFoldIndices::Array{Int64,1};
        ensembleType::Symbol = :VotingHard,
        final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)
    )

Train a classification ensemble model using the specified estimators, models hyperparameters, training dataset, and k-fold cross-validation indices.

# Arguments
- `estimators::AbstractArray{Symbol,1}`: An array of symbols representing the types of models to be used as estimators.
- `modelsHyperParameters::AbstractArray{<:AbstractDict, 1}`: An array of dictionaries containing the hyperparameters for each model.
- `trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}}`: A tuple containing the training inputs and training targets.
- `kFoldIndices::Array{Int64,1}`: An array of integers representing the indices for k-fold cross-validation.
- `ensembleType::Symbol = :VotingHard`: The type of ensemble to create. Supported types are :VotingHard and :Stacking.
- `final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)`: The hyperparameters for the final estimator model.

# Returns
- A dictionary containing the evaluation metrics of the trained ensemble model using k-fold cross-validation:
    - `accuracy`: The mean accuracy of the model across all folds.
    - `std_accuracy`: The standard deviation of the accuracy across all folds.
    - `recall`: The mean recall (true positive rate) of the model across all folds.
    - `std_recall`: The standard deviation of the recall across all folds.
    - `specificity`: The mean specificity (true negative rate) of the model across all folds.
    - `std_specificity`: The standard deviation of the specificity across all folds.
    - `precision`: The mean precision of the model across all folds.
    - `std_precision`: The standard deviation of the precision across all folds.
    - `f1_score`: The mean F1 score of the model across all folds.
    - `std_f1_score`: The standard deviation of the F1 score across all folds.

"""
function trainClassEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:AbstractDict,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,2}},
    kFoldIndices::Array{Int64,1};
    ensembleType::Symbol = :VotingHard,
    final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1),
)

    kFolds = maximum(kFoldIndices)
    testAccuracies = Array{Float64,1}(undef, kFolds)
    testRecalls = Array{Float64,1}(undef, kFolds)
    testErrorRates = Array{Float64,1}(undef, kFolds)
    testSpecificities = Array{Float64,1}(undef, kFolds)
    testPrecisions = Array{Float64,1}(undef, kFolds)
    testNPVs = Array{Float64,1}(undef, kFolds)
    testfScores = Array{Float64,1}(undef, kFolds)

    for numFold = 1:kFolds
        (trainingInputs, trainingTargets, testInputs, testTargets) =
            splitCrossValidationData(trainingDataset, numFold, kFoldIndices)

        models = train_models(
            estimators,
            modelsHyperParameters,
            trainingInputs,
            trainingTargets,
            testInputs,
            testTargets,
        )

        # ensembleModel = VotingClassifier(estimators = models, voting="hard")
        # ensembleModel = StackingClassifier(estimators=models,
        #     final_estimator=create_model(final_estimator["modelType"], final_estimator), n_jobs=1)
        ensembleModel = create_ensemble(ensembleType, models, final_estimator)

        model, _ = train_and_predict(
            ensembleModel,
            trainingInputs,
            trainingTargets,
            testInputs,
            testTargets,
        )

        testOutputs = predict(model, testInputs)

        testAccuracy,
        testErrorRate,
        testRecall,
        testSpecificity,
        testPrecision,
        testNPV,
        testfScore,
        _ = confusionMatrix(testOutputs, vec(testTargets))

        testAccuracies[numFold] = testAccuracy
        testRecalls[numFold] = testRecall
        testErrorRates[numFold] = testErrorRate
        testSpecificities[numFold] = testSpecificity
        testPrecisions[numFold] = testPrecision
        testNPVs[numFold] = testNPV
        testfScores[numFold] = testfScore
    end

    return Dict{String,Any}(
        "accuracy" => mean(testAccuracies),
        "std_accuracy" => std(testAccuracies),
        "recall" => mean(testRecalls),
        "std_recall" => std(testRecalls),
        "specificity" => mean(testSpecificities),
        "std_specificity" => std(testSpecificities),
        "precision" => mean(testPrecisions),
        "std_precision" => std(testPrecisions),
        "f1_score" => mean(testfScores),
        "std_f1_score" => std(testfScores),
    )

end

"""
    trainClassEnsemble(
        estimators::AbstractArray{Symbol,1},
        modelsHyperParameters::AbstractArray{<:AbstractDict, 1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        kFoldIndices::Array{Int64,1};
        ensembleType::Symbol = :VotingHard,
        final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)
    )

Train a classification ensemble model using the specified estimators, models hyperparameters, training dataset, and k-fold cross-validation indices.

# Arguments
- `estimators::AbstractArray{Symbol,1}`: An array of symbols representing the types of models to be used as estimators.
- `modelsHyperParameters::AbstractArray{<:AbstractDict, 1}`: An array of dictionaries containing the hyperparameters for each model.
- `trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}`: A tuple containing the training inputs and training targets.
- `kFoldIndices::Array{Int64,1}`: An array of integers representing the indices for k-fold cross-validation.
- `ensembleType::Symbol = :VotingHard`: The type of ensemble to create. Supported types are :VotingHard and :Stacking.
- `final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)`: The hyperparameters for the final estimator model.

# Returns
- A dictionary containing the evaluation metrics of the trained ensemble model using k-fold cross-validation:
    - `accuracy`: The mean accuracy of the model across all folds.
    - `std_accuracy`: The standard deviation of the accuracy across all folds.
    - `recall`: The mean recall (true positive rate) of the model across all folds.
    - `std_recall`: The standard deviation of the recall across all folds.
    - `specificity`: The mean specificity (true negative rate) of the model across all folds.
    - `std_specificity`: The standard deviation of the specificity across all folds.
    - `precision`: The mean precision of the model across all folds.
    - `std_precision`: The standard deviation of the precision across all folds.
    - `f1_score`: The mean F1 score of the model across all folds.
    - `std_f1_score`: The standard deviation of the F1 score across all folds.

"""
function trainClassEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:AbstractDict,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}},
    kFoldIndices::Array{Int64,1};
    ensembleType::Symbol = :VotingHard,
    final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1),
)

    inputs, targets = trainingDataset

    targets = reshape(targets, :, 1)

    return trainClassEnsemble(
        estimators,
        modelsHyperParameters,
        (inputs, targets),
        kFoldIndices;
        ensembleType = ensembleType,
        final_estimator = final_estimator,
    )

end

"""
    trainClassEnsemble(
        baseEstimator::Symbol,
        modelsHyperParameters::AbstractDict,
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}},
        kFoldIndices::Array{Int64,1};
        NumEstimators::Int=100,
        final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)
    )

Train a classification ensemble model using the specified base estimator, models hyperparameters, training dataset, and k-fold cross-validation indices.

# Arguments
- `baseEstimator::Symbol`: The type of base estimator to be used in the ensemble.
- `modelsHyperParameters::AbstractDict`: The hyperparameters for the base estimator model.
- `trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,2}}`: A tuple containing the training inputs and training targets.
- `kFoldIndices::Array{Int64,1}`: An array of integers representing the indices for k-fold cross-validation.
- `NumEstimators::Int=100`: The number of estimators to be used in the ensemble.
- `final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1)`: The hyperparameters for the final estimator model.

# Returns
- A dictionary containing the evaluation metrics of the trained ensemble model using k-fold cross-validation:
    - `accuracy`: The mean accuracy of the model across all folds.
    - `std_accuracy`: The standard deviation of the accuracy across all folds.
    - `recall`: The mean recall (true positive rate) of the model across all folds.
    - `std_recall`: The standard deviation of the recall across all folds.
    - `specificity`: The mean specificity (true negative rate) of the model across all folds.
    - `std_specificity`: The standard deviation of the specificity across all folds.
    - `precision`: The mean precision of the model across all folds.
    - `std_precision`: The standard deviation of the precision across all folds.
    - `f1_score`: The mean F1 score of the model across all folds.
    - `std_f1_score`: The standard deviation of the F1 score across all folds.

"""
function trainClassEnsemble(
    baseEstimator::Symbol,
    modelsHyperParameters::AbstractDict,
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,2}},
    kFoldIndices::Array{Int64,1};
    NumEstimators::Int = 100,
    final_estimator::AbstractDict = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 1),
)

    estimators = fill(baseEstimator, NumEstimators)
    modelsHyperParameters = fill(modelsHyperParameters, NumEstimators)

    return trainClassEnsemble(
        estimators,
        modelsHyperParameters,
        trainingDataset,
        kFoldIndices,
    )

end
