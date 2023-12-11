include('model.jl')
include('metrics.jl')

function oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    numInstances, numClasses = size(targets)

    # Create a 2-dimensional matrix of real values, with as many rows as patterns and as many columns as classes
    outputs = Array{Float32,2}(undef, numInstances, numClasses)

    # Make a loop that iterates over each class
    for numClass in 1:numClasses
        # Create the desired outputs corresponding to that class
        desiredOutputs = targets[:, numClass]

        # Train the corresponding model with those inputs and the new desired outputs corresponding to that class
        model, _ = trainClassANN([2, 2], (inputs, desiredOutputs))

        # Apply the model to the inputs to calculate the outputs, which will be copied into the previously created matrix
        outputs[:,numClass] .= model(inputs')'
    end
    
    vmax = maximum(outputs, dims=2);
    outputs = (outputs .== vmax);

    return outputs
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets)
    @assert size(outputs, 2) > 1
    
    numClasses = size(outputs, 2)
    sensitivity = zeros(numClasses)
    specificity = zeros(numClasses)
    precision = zeros(numClasses)
    negative_predictive_value = zeros(numClasses)
    fScore = zeros(numClasses)
    confusion_matrix = zeros(Int, numClasses, numClasses)
    
    for i in 1:numClasses
        classOutputs = outputs[:,i]
        classTargets = targets[:,i]
        if any(classTargets)
            classMetrics = confusionMatrix(classOutputs, classTargets)
            sensitivity[i] = classMetrics.sensitivity
            specificity[i] = classMetrics.specificity
            precision[i] = classMetrics.precision
            negative_predictive_value[i] = classMetrics.negative_predictive_value
            fScore[i] = classMetrics.fScore
        end

        for j in 1:numClasses
            confusion_matrix[i, j] = sum(targets[:, i] .& outputs[:, j])
        end
    end

    numInstancesFromEachClass = vec(sum(targets, dims=1))
    
    if weighted
        weights = numInstancesFromEachClass ./ size(targets, 1)
        sensitivity = sum(weights.*sensitivity)
        specificity = sum(weights.*specificity)
        precision = sum(weights.*precision)
        negative_predictive_value = sum(weights.*negative_predictive_value)
        fScore = sum(weights.*fScore)
    else
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        precision = mean(precision)
        negative_predictive_value = mean(negative_predictive_value)
        fScore = mean(fScore)
    end

    acc = accuracy(targets, outputs)
    errorRate = 1 - acc
    
    return (accuracy=acc,
            errorRate=errorRate,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            negative_predictive_value=negative_predictive_value,
            fScore=fScore,
            confusion_matrix=confusion_matrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    boolOutputs = classifyOutputs(outputs)
    return confusionMatrix(boolOutputs, targets, weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]))
    
    classes = unique(targets)
    return confusionMatrix(oneHotEncoding(outputs,classes) ,oneHotEncoding(targets,classes));
    
    return confusionMatrix(boolOutputs, boolTargets, weighted=weighted)
end