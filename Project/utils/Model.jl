module Model

"""
Focused on training machine learning models, this module offers functionalities for model building, training with diverse parameters,
creating model ensembles, performing cross-validation, and managing the training process for both individual models and ensembles.
It serves as a comprehensive toolkit for training, evaluating, and optimizing machine learning models.
"""

include("ClassificationMetrics.jl")
include("DataHandling.jl")
include("DataPreprocessing.jl")
using .ClassificationMetrics
using .DataHandling
using .DataPreprocessing

using Flux
using Flux.Losses
using Flux: train!
using Random
using Statistics

using ScikitLearn
@sk_import svm:SVC
@sk_import tree:DecisionTreeClassifier
@sk_import neighbors:KNeighborsClassifier
@sk_import ensemble:StackingClassifier
@sk_import ensemble:VotingClassifier
@sk_import ensemble:RandomForestClassifier
@sk_import decomposition:PCA

export buildClassANN,
    trainClassANN,
    create_model,
    train_ann_model,
    createAndTrainFinalModel,
    createAndTrainFinalEnsemble,
    train_models,
    train_and_predict,
    create_ensemble,
    trainClassEnsemble,
    modelCrossValidation

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
        println(size(testTargets))
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

end
