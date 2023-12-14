module DataPreprocessing

"""
This module focuses on preparing raw data for machine learning tasks. It encapsulates functions to encode categorical features into a format
suitable for algorithms, calculate parameters for different normalization techniques such as min-max and zero-mean normalization, and apply these
techniques to datasets. It serves as a toolbox for transforming raw data into a usable format for machine learning models.
"""

using Statistics

export oneHotEncoding,
    calculateMinMaxNormalizationParameters,
    calculateZeroMeanNormalizationParameters,
    normalizeMinMax!,
    normalizeMinMax,
    normalizeZeroMean,
    normalizeZeroMean!
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

end
