module ClassificationMetrics

"""
Dedicated to evaluating the performance of classification models, this module provides essential metrics and assessment tools.
It includes functions for classifying model outputs based on thresholds, computing accuracy metrics for binary and multi-class classification,
and generating confusion matrices. These functions enable the assessment of predictive accuracy and performance of classification models.
"""
using Statistics

export accuracy, confusionMatrix

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

end
