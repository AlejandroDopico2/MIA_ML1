using CSV
using DataFrames
using Random
import StatsBase: countmap
using ScikitLearn
using Flux

@sk_import svm:SVC;
@sk_import tree:DecisionTreeClassifier;
@sk_import neighbors:KNeighborsClassifier;
@sk_import ensemble:StackingClassifier;
@sk_import ensemble:VotingClassifier;
@sk_import ensemble:RandomForestClassifier;
@sk_import decomposition:PCA;


include("modules/ClassificationMetrics.jl")
include("modules/DataHandling.jl")
include("modules/DataPreprocessing.jl")
include("modules/Model.jl")

using .ClassificationMetrics
using .DataHandling
using .DataPreprocessing
using .Model

function ip_to_decimal(ip)
    # Split the IP address into octets
    octets = split(ip, '.')
    # Convert each octet to binary and combine them into a single 32-bit number
    binary =
        join([string(parse(Int, octet, base = 10), base = 2, pad = 8) for octet in octets])
    decimal = parse(Int, binary, base = 2) # Convert binary to decimal
    return decimal
end

function generate_latex_table(metrics::Dict{String,<:Any}, final::Bool)

    topology = metrics["topology"]
    accuracy = metrics["accuracy"]
    recall = metrics["recall"]
    specificity = metrics["specificity"]
    f1_score = metrics["f1_score"]

    if final
        confusion_matrix = metrics["confusion_matrix"]
        println(
            "$topology & $(round(accuracy*100, digits=2))\\%  & $(round(recall*100, digits=2))\\%  & $(round(specificity*100, digits=2))\\%  & $(round(f1_score*100, digits=2))\\% & $confusion_matrix \\\\",
        )
    else
        std_accuracy = metrics["std_accuracy"]
        std_recall = metrics["std_recall"]
        std_specificity = metrics["std_specificity"]
        std_f1_score = metrics["std_f1_score"]
        println(
            "$topology & $(round(accuracy*100, digits=2))\\% \\textit{($(round(std_accuracy, digits = 2)))} & $(round(recall*100, digits=2))\\% \\textit{($(round(std_recall, digits = 2)))} & $(round(specificity*100, digits=2))\\% \\textit{($(round(std_specificity, digits = 2)))} & $(round(f1_score*100, digits=2))\\% \\textit{($(round(std_f1_score, digits = 2)))} \\\\",
        )
    end

end

file_path = "datasets/simplified_Android_Malware.csv";
data = CSV.File(file_path, header = true) |> DataFrame;

columns_to_drop = ["Flow ID", " Timestamp"]
columns = names(data)

println("Size of dataframe before dropping columns $(size(data))")
for column = 1:size(data, 2)
    unique_values = countmap(data[:, column])

    if length(unique_values) == 1
        println("\t-Removing column $(columns[column])")
        push!(columns_to_drop, columns[column])
    end

end

select!(data, Not(columns_to_drop))

println("Size of dataframe after dropping columns $(size(data))")

dropmissing!(data)

println("Size of dataframe after dropping nulls $(size(data))")

unique_data = unique(data)

println("Size of dataframe after dropping duplicating rows $(size(data))")

println(countmap(data[:, :Label]));

source_ips = data[!, :" Source IP"];
destination_ips = data[!, :" Destination IP"];

data[!, :"Source IP Decimal"] = map(ip -> ip_to_decimal(ip), source_ips);
data[!, :"Destination IP Decimal"] = map(ip -> ip_to_decimal(ip), destination_ips);

select!(data, Not([" Source IP", " Destination IP"]));

output_data = data[!, :Label];
select!(data, Not(:Label));
input_data = Matrix(data[!, 1:size(data, 2)]);

println("Size of the input dataset is $(size(input_data))")

println("Doing first approach")
include("first_approach.jl")

println("Doing second approach")
include("second_approach.jl")

println("Doing third approach")
include("third_approach.jl")

println("Doing fourth approach")
include("fourth_approach.jl")
