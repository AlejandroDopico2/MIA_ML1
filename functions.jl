using Statistics;
using Flux;
using Flux.Losses;

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    if num_classes == 2
        one_hot = reshape(data.==classes[2], :, 1)
     else
        one_hot = zeros(Bool, length(feature), num_classes)
        
        for i = 1:num_classes
            one_hot[:, i] .= (feature .== classes[i])
        end
        
        return one_hot
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1)
end

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (minimum(dataset, dims=1), maximum(dataset, dims=1))
end


function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (mean(dataset, dims=1), std(dataset, dims=1))
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    min, max = normalizationParameters

    dataset .-= min
    dataset ./= (max .- min)

    dataset[:, vec(min .== max)] .= 0

    return dataset
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

    dataset_copy = copy(dataset)

    return normalizeMinMax!(dataset_copy, normalizationParameters)
    
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2})

    dataset_copy = copy(dataset)

    return normalizeMinMax!(dataset_copy)
    
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    
    mean, std = normalizationParameters

    dataset .-= mean;
    dataset ./= std;
    dataset[:, vec(std.==0)] .= 0;

    return dataset
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    dataset_copy = copy(dataset)

    return normalizeZeroMean!(dataset_copy, normalizationParameters)
end

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    dataset_copy = copy(dataset)

    return normalizeZeroMean!(dataset_copy)
end

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

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann = Chain();
    numInputsLayer = numInputs;

    for numHiddenLayer in 1:length(topology)
        neurons = topology[numHiddenLayer]
        ann = Chain(ann..., Dense(numInputsLayer, neurons, transferFunctions[numHiddenLayer]))
        numInputsLayer = neurons
    end

    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann;
end