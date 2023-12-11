using Statistics;
using Flux;
using Flux.Losses;
using Random;
using Flux:train!;

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

function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false)
    
    # Split into inputs/targets
    trainingInputs, trainingTargets = trainingDataset
    validationInputs, validationTargets = validationDataset
    testInputs, testTargets = testDataset
    
    useValidation = length(validationInputs) > 0
    useTest = length(testInputs) > 0

    # Check that the targets corresponding to the inputs have the same number of samples
    @assert (size(trainingInputs, 1) == size(trainingTargets, 1)) "Number of training inputs and targets do not match"
    @assert (size(validationInputs, 1) == size(validationTargets, 1)) "Number of validation inputs and targets do not match"
    @assert (size(testInputs, 1) == size(testTargets, 1)) "Number of test inputs and targets do not match"
    
    if (useValidation)
        @assert (size(trainingInputs, 2) == size(validationInputs, 2)) "Number of attributes for training and validation do not match"
    end
    if (size(testInputs, 1) != 0)
        @assert (size(trainingInputs, 2) == size(testInputs, 2)) "Number of attributes for training and test do not match"
    end
   
    # Build the network
    nInputs, nOutputs = size(trainingInputs, 2), size(trainingTargets, 2)
    ann = buildClassANN(nInputs, topology, nOutputs; transferFunctions)

    if showText
        println("ANN network built: $ann")
    end

    # Loss
    loss(x,y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y)
    # Metric progress
    trainingLosses = Array{Float32}(undef, 0)
    validationLosses = Array{Float32}(undef, 0)
    testLosses = Array{Float32}(undef, 0)
    trainingAccs = Array{Float32}(undef, 0)
    validationAccs = Array{Float32}(undef, 0)
    testAccs = Array{Float32}(undef, 0)

    # Train for n=max_epochs (at most) epochs
    currentEpoch = 0

    # Calculate, store and print last loss/accuracy
    function calculateMetrics()
        
        # Losses
        trainingLoss   = loss(trainingInputs', trainingTargets')
        validationLoss = (size(validationInputs, 1) != 0) ? loss(validationInputs', validationTargets') : 0
        testLoss       = (size(testInputs, 1) != 0) ? loss(testInputs', testTargets') : 0
     
        # Accuracies
        trainingOutputs   = ann(trainingInputs')
        
        validationAcc = 0
        testAcc = 0
        
        if useValidation
            validationOutputs = ann(validationInputs')
            validationAcc = accuracy(validationOutputs', validationTargets)
        end

        if useTest
            testOutputs = ann(testInputs')
            testAcc = accuracy(testOutputs',       testTargets)
        end
        
        trainingAcc   = accuracy(trainingOutputs',   trainingTargets)

        # Update the history of losses and accuracies
        push!(trainingLosses, trainingLoss)
        push!(validationLosses, validationLoss)
        push!(testLosses, testLoss)
        push!(trainingAccs, trainingAcc)
        push!(validationAccs, validationAcc)
        push!(testAccs, testAcc)
            
        # Show text
        if showText && (currentEpoch % 50 == 0)
            println("Epoch ", currentEpoch, 
                ": \n\tTraining loss: ", trainingLoss, ", accuracy: ", 100 * trainingAcc, 
                "% \n\tValidation loss: ", validationLoss, ", accuracy: ", 100 * validationAcc, 
                "% \n\tTest loss: ", testLoss, ", accuracy: ", 100 * testAcc, "%")
        end
        
        return trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc
    end

    # Compute and store initial metrics
    trainingLoss, _, validationLoss, _, _, _ = calculateMetrics()

    # Best model at validation set 
    numEpochsValidation = 0
    bestValidationLoss = validationLoss
    
    if (useValidation)
        bestAnn = deepcopy(ann)
    else
        bestAnn = ann  # if no validation, we want to return the ANN that is trained in every cycle
    end
    
    # Start the training

    while (currentEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochsValidation < maxEpochsVal)
            
        # Update epoch number
        currentEpoch += 1

        # Fit the model
        Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate))

        # Compute and store metrics
        trainingLoss, _, validationLoss, _, _, _ = calculateMetrics()

        # Update validation early stopping only if validation set given
        if (useValidation)    
            if (validationLoss < bestValidationLoss)
                bestValidationLoss = validationLoss
                numEpochsValidation = 0
                bestAnn = deepcopy(ann)
            else
                numEpochsValidation += 1
            end
        end

    end
    
    # Print stop reason and final metrics

    if showText
        println("Final results for epoch $currentEpoch: 
            \n\tTraining loss: $(trainingLosses[end]), accuracy: $(100 * trainingAccs[end])%
            \n\tValidation loss: $(validationLosses[end]), accuracy: $(100 * validationAccs[end])%
            \n\tTest loss: $(testLosses[end]), accuracy: $(100 * testAccs[end])%")
    
        println("\nStopping criteria: 
            \n\tMax. epochs: $(currentEpoch >= maxEpochs)
            \n\tMin. loss: $(trainingLoss <= minLoss)
            \n\tNum. epochs validation: $(numEpochsValidation >= maxEpochsVal)")
    end
    
    return bestAnn, trainingLosses, validationLosses, testLosses, trainingAccs, validationAccs, testAccs
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