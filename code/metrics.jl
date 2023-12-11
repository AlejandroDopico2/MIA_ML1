using Statistics;
using Flux;
using Flux.Losses;
using Random;
using Flux:train!;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5)
    if size(outputs, 2) == 1
        outputs = outputs .> threshold
    else
        _, indicesMaxEachInstance = findmax(outputs, dims = 2)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
    end

    return outputs
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    @assert (size(outputs) == size(targets)) "Outputs and targets must have same number of rows"

    return mean(outputs .== targets)  
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 

    if size(outputs, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1])
    end

    @assert (size(outputs) == size(targets)) "Outputs and targets must have same number of rows"

    _, indicesMaxEachInstanceInOutputs = findmax(outputs, dims = 2)
    _, indicesMaxEachInstanceInTargets = findmax(targets, dims = 2)

    return mean(indicesMaxEachInstanceInOutputs .== indicesMaxEachInstanceInTargets)
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};
        threshold::Real=0.5)
    
    outputs = outputs .> threshold
    return accuracy(outputs, targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)

    if size(outputs, 2) == 1
        return accuracy(outputs, targets, threshold)
    else
        outputs = classifyOutputs(outputs)
        return accuracy(outputs, targets)
    end
end

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
        sensitivity = 1.
        precision = 1.
    elseif isnan(specificity) && isnan(negative_predictive_value)
        specificity = 1.
        negative_predictive_value = 1.
    end

    sensitivity = isnan(sensitivity) ? 0. : sensitivity
    precision = isnan(precision) ? 0. : precision
    specificity = isnan(specificity) ? 0. : specificity
    negative_predictive_value = isnan(negative_predictive_value) ? 0. : negative_predictive_value

    confusion_matrix = [tp fp; fn tn]

    fScore = (precision == sensitivity == 0) ? 0 : 2 * (precision * sensitivity) / (precision + sensitivity)

    return (accuracy=accuracy,
            errorRate=errorRate,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            negative_predictive_value=negative_predictive_value,
            fScore=fScore,
            confusion_matrix=confusion_matrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .> threshold
    return confusionMatrix(outputs, targets)
end