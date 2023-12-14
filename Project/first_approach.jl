# Binary classification

function transform_binary_class(output_data)
    binary_labels = output_data .!= "Benign"
    return binary_labels
end

binary_labels = transform_binary_class(output_data)
@assert binary_labels isa BitVector
@assert input_data isa Matrix

println("Distribution of categories of first approach: $(countmap(binary_labels))")

# Preprocessing

Random.seed!(42)

train_indexes, test_indexes = holdOut(size(input_data, 1), 0.2)

train_input = convert(Array{Float32,2}, input_data[train_indexes, :])
train_binary_output = binary_labels[train_indexes]

normalizationParameters = calculateMinMaxNormalizationParameters(train_input)

normalizeMinMax!(train_input, normalizationParameters)

test_input = convert(Array{Float32,2}, input_data[test_indexes, :])
test_binary_output = binary_labels[test_indexes]

normalizeMinMax!(test_input, normalizationParameters)

@assert size(test_input, 1) == size(test_binary_output, 1)
@assert size(train_input, 1) == size(train_binary_output, 1)

kFolds = 10
crossValidationIndexes = crossvalidation(train_binary_output, kFolds);

println(
    "-------------------------Artificial Neural Networks---------------------------------------",
)

topologies = [[20], [40], [80], [100], [60, 120], [80, 50], [80, 100], [100, 40]]

annParameters = Dict(
    "modelType" => :ANN,
    "maxEpochs" => 200,
    "learningRate" => 0.01,
    "maxEpochsVal" => 30,
    "repetitions" => 30,
    "validationRatio" => 0.1,
    "transferFunctions" => fill(σ, 2),
)

for topology in topologies
    annParameters["topology"] = topology
    metricsCV = modelCrossValidation(
        annParameters["modelType"],
        annParameters,
        train_input,
        train_binary_output,
        crossValidationIndexes,
    )
    metricsCV["topology"] = topology

    generate_latex_table(metricsCV, false)
end

println("----------------------------------------------------------------")

for topology in topologies
    annParameters["topology"] = topology
    metrics = createAndTrainFinalModel(
        annParameters["modelType"],
        annParameters,
        train_input,
        train_binary_output,
        test_input,
        test_binary_output,
    )
    metrics["topology"] = topology

    generate_latex_table(metrics, true)
end

println("-------------------------kNN---------------------------------------")

knnParameters = Dict("modelType" => :kNN, "numNeighboors" => 0)

ks = [1, 2, 3, 5, 7, 10, 15]
for k in ks
    knnParameters["numNeighboors"] = k
    metricsCV = (modelCrossValidation(
        knnParameters["modelType"],
        knnParameters,
        train_input,
        train_binary_output,
        crossValidationIndexes,
    ))
    metricsCV["topology"] = k

    generate_latex_table(metricsCV, false)
end

println("----------------------------------------------------------------")
for k in ks
    knnParameters["numNeighboors"] = k
    metrics = createAndTrainFinalModel(
        knnParameters["modelType"],
        knnParameters,
        train_input,
        train_binary_output,
        test_input,
        test_binary_output,
    )
    metrics["topology"] = k

    generate_latex_table(metrics, true)
end

println("-------------------------Decision Tree---------------------------------------")

dtParameters = Dict("modelType" => :DecisionTree, "maxDepth" => 1)

depths = [3, 5, 7, 10, 15, nothing]
for depth in depths
    dtParameters["maxDepth"] = depth
    metricsCV = (modelCrossValidation(
        dtParameters["modelType"],
        dtParameters,
        train_input,
        train_binary_output,
        crossValidationIndexes,
    ))
    metricsCV["topology"] = depth

    generate_latex_table(metricsCV, false)

end

println("----------------------------------------------------------------")

for depth in depths
    dtParameters["maxDepth"] = depth
    metrics = createAndTrainFinalModel(
        dtParameters["modelType"],
        dtParameters,
        train_input,
        train_binary_output,
        test_input,
        test_binary_output,
    )
    metrics["topology"] = depth

    generate_latex_table(metrics, true)
end

println("-------------------------SVM---------------------------------------")

svmParameters = Dict(
    "modelType" => :SVM,
    "C" => 1,
    "kernel" => "linear",
    "degree" => 3,
    "gamma" => "scale",
)

svms = [
    ("rbf", 0.1),
    ("rbf", 1.0),
    ("rbf", 10.0),
    ("poly", 0.1),
    ("poly", 1.0),
    ("linear", 0.1),
    ("linear", 1.0),
    ("linear", 10.0),
]

for (kernel, C) in svms
    svmParameters["kernel"] = kernel
    svmParameters["C"] = C
    metricsCV = (modelCrossValidation(
        svmParameters["modelType"],
        svmParameters,
        train_input,
        train_binary_output,
        crossValidationIndexes,
    ))
    metricsCV["topology"] = kernel * " & " * string(C)

    generate_latex_table(metricsCV, false)

end

println("----------------------------------------------------------------")

for (kernel, C) in svms
    svmParameters["kernel"] = kernel
    svmParameters["C"] = C
    metrics = createAndTrainFinalModel(
        svmParameters["modelType"],
        svmParameters,
        train_input,
        train_binary_output,
        test_input,
        test_binary_output,
    )
    metrics["topology"] = kernel * " & " * string(C)

    generate_latex_table(metrics, true)

end

println("-------------------------Ensembles---------------------------------------")


dtParameters = Dict("modelType" => :DecisionTree, "maxDepth" => 10)
knnParameters = Dict("modelType" => :kNN, "numNeighboors" => 2)
svmParameters = Dict("modelType" => :SVM, "kernel" => "rbf", "C" => 0.1)
Random.seed!(42)

ensemble_types = [:VotingHard, :Stacking]
final_estimators = [dtParameters, knnParameters, svmParameters]

for ensemble_type in ensemble_types
    for final_estimator in final_estimators
        metricsCV = trainClassEnsemble(
            [:DecisionTree, :kNN, :SVM],
            [dtParameters, knnParameters, svmParameters],
            (train_input, train_binary_output),
            crossValidationIndexes;
            ensembleType = ensemble_type,
            final_estimator = final_estimator,
        )
        metricsCV["topology"] = final_estimator
        generate_latex_table(metricsCV, false)

        if ensemble_type == :VotingHard
            break
        end
    end
end

println("----------------------------------------------------------------")


for ensemble_type in ensemble_types
    for final_estimator in final_estimators
        metrics = createAndTrainFinalEnsemble(
            [:DecisionTree, :kNN, :SVM],
            [dtParameters, knnParameters, svmParameters],
            (train_input, train_binary_output),
            (test_input, test_binary_output);
            ensembleType = ensemble_type,
            final_estimator = final_estimator,
        )
        metrics["topology"] = final_estimator
        generate_latex_table(metrics, true)

        if ensemble_type == :VotingHard
            break
        end
    end
end
