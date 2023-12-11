function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    if num_classes == 2
        one_hot = reshape(feature.==classes[2], :, 1)
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