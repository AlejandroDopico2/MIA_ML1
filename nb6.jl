using ScikitLearn

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

using Statistics

function create_model(modelType::Symbol, modelHyperparameters::Dict)
    if modelType == :SVM
        return SVC(kernel=modelHyperparameters["kernel"],
                   degree=modelHyperparameters["degree"],
                   gamma=modelHyperparameters["gamma"],
                   C=modelHyperparameters["C"])
    elseif modelType == :kNN
        return KNeighborsClassifier(modelHyperparameters["numNeighboors"])
    elseif modelType == :DecisionTree
        return DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"])
    else
        error("Model type not supported")
    end
end

function train_and_predict(model, trainingInputs, trainingTargets, testInputs, testTargets)
    fit!(model, trainingInputs, vec(trainingTargets))
    testOutputs = predict(model, testInputs)
    return testOutputs
end

function train_ann_model(modelHyperparameters, trainingInputs, trainingTargets, testInputs, testTargets)
    testAccuraciesForEachRepetition = Array{Float64, 1}(undef, modelHyperparameters["repetitions"])

    for numTraining in 1:modelHyperparameters["repetitions"]
        if modelHyperparameters["validationRatio"] > 0.0
            (trainingIndexes, validationIndexes) = holdOut(size(trainingInputs, 1), modelHyperparameters["validationRatio"])
            
            validationInputs = trainingInputs[validationIndexes, :]
            validationTargets = trainingTargets[validationIndexes, :]
            trainingInputs = trainingInputs[trainingIndexes, :]
            trainingTargets = trainingTargets[trainingIndexes, :]
            
            model, _ = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets);
                                    validationDataset = (validationInputs, validationTargets),
                                    testDataset = (testInputs, testTargets),
                                    transferFunctions = modelHyperparameters["transferFunctions"],
                                    maxEpochs = modelHyperparameters["maxEpochs"],
                                    learningRate = modelHyperparameters["learningRate"],
                                    maxEpochsVal = modelHyperparameters["maxEpochsVal"])
        else
            model, _ = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets);
                                    testDataset = (testInputs, testTargets),
                                    transferFunctions = modelHyperparameters["transferFunctions"],
                                    maxEpochs = modelHyperparameters["maxEpochs"],
                                    learningRate = modelHyperparameters["learningRate"],
                                    maxEpochsVal = modelHyperparameters["maxEpochsVal"])
        end

        testOutputs = model(testInputs')'
        testAccuraciesForEachRepetition[numTraining] = accuracy(testOutputs, testTargets)
    end

    return mean(testAccuraciesForEachRepetition)
end

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    @assert(size(inputs, 1) == length(targets))
    @assert (modelType == :ANN) || (modelType == :SVM) || (modelType == :DecisionTree) || (modelType == :kNN)
    
    kFolds = maximum(crossValidationIndices)
    testAccuracies = Array{Float64, 1}(undef, kFolds)

    if modelType == :ANN
        targets = oneHotEncoding(targets)
    end

    for numFold in 1:kFolds
        trainingInputs = inputs[crossValidationIndices .!= numFold, :]
        trainingTargets = targets[crossValidationIndices .!= numFold, :]
    
        testInputs = inputs[crossValidationIndices .== numFold, :]
        testTargets = targets[crossValidationIndices .== numFold, :]

        if modelType != :ANN
            model = create_model(modelType, modelHyperparameters)
            testOutputs = train_and_predict(model, trainingInputs, trainingTargets, testInputs, testTargets)
            testOutputs = oneHotEncoding(testOutputs)
            testTargets = oneHotEncoding(vec(testTargets))
            testAccuracy = accuracy(testOutputs, testTargets)
        else
            testAccuracy = train_ann_model(modelHyperparameters, trainingInputs, trainingTargets, testInputs, testTargets)
        end

        testAccuracies[numFold] = testAccuracy
    end

    return mean(testAccuracies), std(testAccuracies)
end

include("functions.jl")

using DelimitedFiles
using Plots
using Random

Random.seed!(42)

dataset = readdlm("iris.data", ',')

learningRate = 0.01
numMaxEpochs = 100

kFolds = 10

inputs = convert(Array{Float32,2}, dataset[:,1:4])
targets = dataset[:, 5]
normalizeMinMax!(inputs)

testAccuracies = []
testAccuraciesStd = []

n = 150

# Generar un vector de números aleatorios entre 0 y 9
random_numbers = rand(0:9, n)

# Sumar 1 a cada elemento para que estén en el rango de 1 a 10
randomCrossValidation = random_numbers .+ 1

annParameters = Dict("modelType" => :ANN, "topology" => [15, 15], "maxEpochs" => 1000,
    "learningRate" => 0.01, "maxEpochsVal" => 20,
    "repetitions" => 50, "validationRatio" => 0.2,
    "transferFunctions" => fill(σ, 2),)

svmParameters = Dict("modelType" => :SVM, "kernel" => "rbf", "degree" => 1, "gamma" => 1, "C" => 1)
knnParameters = Dict("modelType" => :kNN, "numNeighboors" => 7)
dtParameters = Dict("modelType" => :DecisionTree, "maxDepth" => 2)

hyperparameters = [annParameters, svmParameters, knnParameters, dtParameters]

for hyperparameter in hyperparameters
    println(hyperparameter)

    testAccuracy, testAccuracyStd = modelCrossValidation(hyperparameter["modelType"], hyperparameter, inputs, targets, randomCrossValidation)
    push!(testAccuracies, testAccuracy)
    push!(testAccuraciesStd, testAccuracyStd)
end

println(testAccuracies, testAccuraciesStd)