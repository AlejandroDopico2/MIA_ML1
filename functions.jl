using Statistics;
using Flux;
using Flux.Losses;

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

using Random

function holdOut(N::Int, P::Float64)
    # Verify that P is a valid probability
    @assert ((P >= 0.0) && (P <= 1.0))

    # Generate random permutation of integers from 1 to N
    indexes = randperm(N)

    # Calculate the size of the test set
    test_size = Int(ceil(N * P))

    # Select the indexes of the test and training sets
    test_indexes = indexes[1:test_size]
    train_indexes = indexes[test_size+1:end]

    # Verify that the length of the training and test sets is correct
    @assert length(train_indexes) + length(test_indexes) == N

    # Return the indexes of the training and test sets
    return (train_indexes, test_indexes)
end

function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    # Verify that Pval and Ptest are valid probabilities
    @assert ((Pval >= 0.0) && (Pval <= 1.0))
    @assert ((Ptest >= 0.0) && (Ptest <= 1.0))
    @assert ((Pval + Ptest) <= 1.0)

    train_val_indexes, test_indexes = holdOut(N, Ptest)

    train_val_size = length(train_val_indexes)

    train_indexes, val_indexes = holdOut(train_val_size, Pval * N / train_val_size)

    train_indexes = train_val_indexes[train_indexes]
    val_indexes = train_val_indexes[val_indexes]

    @assert length(train_indexes) + length(val_indexes) + length(test_indexes) == N

    return (train_indexes, val_indexes, test_indexes)
end

# Split dataset into training, validation, and test sets with 60% for training, 20% for validation, and 20% for testing
N = 10
train_indexes, val_indexes, test_indexes = holdOut(N, 0.2, 0.1)


using Flux:train!
function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false)
    
    (trainingInputs, trainingTargets) = trainingDataset

    # Check if validation and test sets are empty
    is_val_empty = size(validationDataset[1]) == (0, 0)
    is_test_empty = size(testDataset[1]) == (0, 0)

    if !is_val_empty
        validationInputs, validationTargets = validationDataset

        @assert(size(validationInputs, 1) == size(validationTargets, 1))
        @assert(size(trainingInputs, 2) == size(validationInputs, 2))
        @assert(size(trainingTargets, 2) == size(validationTargets, 2))
    end

    if !is_test_empty
        testInputs, testTargets = testDataset

        @assert(size(testInputs, 1) == size(testTargets, 1))
        @assert(size(trainingInputs, 2) == size(testInputs, 2))
        @assert(size(trainingTargets, 2) == size(testTargets, 2))
    end
    
    # We define the ANN
    ann = buildClassANN(size(trainingInputs, 2), topology, size(trainingTargets, 2))

    # Setting up the loss function to reduce the error
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y)

    # Initialize variables for early stopping
    bestValidationLoss = Inf
    consecutiveEpochsWithoutImprovement = 0

    # This vector is going to contain the losses on each training epoch
    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]

    # Calculate the loss without training
    trainingLoss = loss(ann, trainingInputs', trainingTargets')

    if (!is_test_empty)
        testLoss = loss(ann, testInputs', testTargets')
        push!(testLosses, testLoss)
    end
    
    if (!is_val_empty)
        validationLoss = loss(ann, validationInputs', validationTargets')
        push!(validationLosses, validationLoss)
    end

    if (showText)
        if (!is_test_empty && !is_val_empty)
            println("Epoch 0: Train Loss = ", trainingLoss, ", Validation Loss = ", validationLoss, ", Test Loss = ", testLoss)
        elseif (!is_val_empty)
            println("Epoch 0: Train Loss = ", trainingLoss, ", Validation Loss = ", validationLoss)
        elseif (!is_test_empty)
            println("Epoch 0: Train Loss = ", trainingLoss, ", Test Loss = ", testLoss)
        else
            println("Epoch 0: Train Loss = ", trainingLoss)
        end
    end

    # Store this one for checking the evolution.
    push!(trainingLosses, trainingLoss)

    # Define the optimizer for the network
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Start the training until it reaches one of the stop criteria
    for numEpoch in 1:maxEpochs
        # Train for one epoch
        train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state)

        # Calculate the loss for this epoch
        trainingLoss = loss(ann, trainingInputs', trainingTargets')
        # Store it
        push!(trainingLosses, trainingLoss)

        if !is_test_empty
            testLoss = loss(ann, testDataset[1]', testDataset[2]')
            push!(testLosses, testLoss)
        end

        # If a validation set is provided, calculate the validation loss
        if !is_val_empty
            validationLoss = loss(ann, validationDataset[1]', validationDataset[2]')
            push!(validationLosses, validationLoss)

            # Check if validation loss improved
            if validationLoss < bestValidationLoss
                bestValidationLoss = validationLoss
                consecutiveEpochsWithoutImprovement = 0
                best_ann = deepcopy(ann)
            else
                consecutiveEpochsWithoutImprovement += 1
            end

            # Stop training if maxEpochsVal consecutive epochs without improvement are reached
            if consecutiveEpochsWithoutImprovement >= maxEpochsVal
                println("Early stopping after $numEpoch epochs.")
                ann = best_ann  # Return the best ANN
                break
            end
        end

        if (showText)
            if (!is_test_empty && !is_val_empty)
                println("Epoch ", numEpoch, ": Train Loss = ", trainingLoss, ", Validation Loss = ", validationLoss, ", Test Loss = ", testLoss)
            elseif (!is_val_empty)
                println("Epoch ", numEpoch, ": Train Loss = ", trainingLoss, ", Validation Loss = ", validationLoss)
            elseif (!is_test_empty)
                println("Epoch ", numEpoch, ": Train Loss = ", trainingLoss, ", Test Loss = ", testLoss)
            else
                println("Epoch ", numEpoch, ": Train Loss = ", trainingLoss)
            end
        end
        
    end

    # If a validation set is empty, the ANN to return is the one at the last cycle
    if is_val_empty
        best_ann = deepcopy(ann)
    end

    # Return the network and the evolution of the error
    if is_val_empty && is_test_empty
        return (best_ann, trainingLosses)
    elseif is_val_empty
        return (best_ann, trainingLosses, testLosses)
    elseif is_test_empty
        return (best_ann, trainingLosses, validationLosses)
    else
        return (best_ann, trainingLosses, validationLosses, testLosses)
    end
end


function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)

    trainingInputs, trainingTargets = trainingDataset
    testInputs, testTargets = testDataset
    validationInputs, validationTargets = validationDataset
    testTargets = reshape(testTargets, :, 1)
    validationTargets = reshape(validationTargets, :, 1)

    trainingTargets = reshape(trainingTargets, :, 1)

    is_val_empty = size(validationDataset[1]) == (0, 0)
    is_test_empty = size(testDataset[1]) == (0, 0)

    return trainClassANN(topology, (trainingInputs, trainingTargets);
        validationDataset = (validationInputs, validationTargets),
        testDataset = (testInputs, testTargets),
        transferFunctions = transferFunctions, maxEpochs = maxEpochs,
        minLoss = minLoss, learningRate = learningRate,
        maxEpochsVal = maxEpochsVal, showText= showText
    )

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
