using Statistics;
using Flux;
using Flux.Losses;
using Random;
using Flux:train!;

function splitCrossValidationData(
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    numFold::Int64, 
    kFoldIndices::Array{Int64,1})

    inputs, targets = trainingDataset

    trainingInputs = inputs[kFoldIndices .!= numFold, :]
    trainingTargets = targets[kFoldIndices .!= numFold, :]
    
    testInputs = inputs[kFoldIndices .== numFold, :]
    testTargets = targets[kFoldIndices .== numFold, :]

    return (trainingInputs, trainingTargets, testInputs, testTargets)

end

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

function train_models(models::AbstractArray{Symbol, 1}, hyperParameters::AbstractArray{<:AbstractDict, 1}, trainingInputs, trainingTargets)
    Random.seed!(42)

    ensemble_models = []
    @assert length(models) == length(hyperParameters)
    
    for i in 1:length(hyperParameters)
        modelType = models[i]
        hyperParameter = hyperParameters[i]

        model = create_model(modelType, hyperParameter)
        fit!(model, trainingInputs, vec(trainingTargets))

        name = "model_$(i)"

        push!(ensemble_models, (name, model))
    end

    return ensemble_models
end

function train_and_predict(model, trainingInputs, trainingTargets, testInputs, testTargets)
    Random.seed!(42)

    fit!(model, trainingInputs, vec(trainingTargets))
    testOutputs = predict(model, testInputs)
    
    testAccuracy, _, _, _, _, _, _, _ = confusionMatrix(testOutputs, vec(testTargets))
    
    return model, testAccuracy
end

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
    modelsHyperParameters:: AbstractArray{<:AbstractDict, 1},     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
    kFoldIndices::     Array{Int64,1})

    kFolds = maximum(kFoldIndices)
    testAccuracies = Array{Float64, 1}(undef, kFolds)
    testAccuraciesEnsemble = Array{Float64, 1}(undef, kFolds)

    for numFold in 1:kFolds
        (trainingInputs, trainingTargets, testInputs, testTargets) = splitCrossValidationData(trainingDataset, numFold, kFoldIndices)

        models = train_models(estimators, modelsHyperParameters, trainingInputs, trainingTargets)
        
        ensembleModel = VotingClassifier(estimators = models, n_jobs=1, voting="hard")

        testAccuracies[numFold] = train_and_predict(ensembleModel, trainingInputs, trainingTargets, testInputs, testTargets)[2]
    end

    return (mean(testAccuracies), std(testAccuracies))

end

function trainClassEnsemble(baseEstimator::Symbol, 
    modelsHyperParameters::AbstractDict,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},     
    kFoldIndices::Array{Int64,1};
    NumEstimators::Int=100)

    estimators = fill(baseEstimator, NumEstimators)
    modelsHyperParameters = fill(modelsHyperParameters, NumEstimators)

    return trainClassEnsemble(estimators, modelsHyperParameters, trainingDataset, kFoldIndices)

end

