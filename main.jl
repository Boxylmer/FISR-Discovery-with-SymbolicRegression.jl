using MLJ, SymbolicRegression
using Plots
using Zygote
plotly()

X = (; input=Float32.(10 .^ collect(range(-3, 3, 1000))))
y = @. Float32(1 / sqrt(big(X.input)));
dy = Float32.([Zygote.gradient(x -> 1/sqrt(x), val)[1] for val in big.(X.input)])

include("operators.jl")



value_loss(prediction, target) = prediction <= 0 ? eltype(target)(Inf) : (log(prediction) - log(target))^2
function deriv_loss(prediction, target)
    # scatter_loss = abs(log((abs(prediction)+1e-20) / (abs(target)+1e-20)))
    # sign_loss = 10 * (sign(prediction) - sign(target))^2
    # return (scatter_loss + sign_loss) / 100
    return 0 # doesn't work currently, setting to zero just doesn't include derivative information. 
    # maybe someone can utilize this eventually? 
end


function numerical_derivative(f, x::AbstractArray{T}, h=T(1e-10)) where T
    return (f(x .+ h) .- f(x)) ./ h
end

function eval_with_newton(tree, x, options, iters=1)
    pred_y, completed = eval_tree_array(tree, x, options)
    for _ in 1:iters
        pred_y .= pred_y .* (3/2 .- (pred_y .* pred_y .* @view x[1, :]) ./ 2)
    end
    return pred_y, completed
end

function loss_function(tree, dataset::Dataset{T,L}, options, idx) where {T,L}
    y = dataset.y
    dy = dataset.weights
    total_loss::L = 0
    # pred_y_improved = eval_with_newton(tree, dataset.X, options)
    pred_y, completed = eval_with_newton(tree, dataset.X, options)
    if !completed
        return L(Inf) # grossly assume that if it ran on the dataset, then it will run on perturbations of those points (±ε of each point in the dataset for 1st deriv.)
    end
   
    function eval_without_flag(x)
        return eval_with_newton(tree, x, options)[1]
    end

    pred_dy = numerical_derivative(eval_without_flag, dataset.X)
    for i in eachindex(y)
        vl = value_loss(pred_y[i], y[i])
        dl = deriv_loss(pred_dy[i], dy[i])
        total_loss += (vl + dl)
    end
    norm_loss = total_loss / length(y)
    return norm_loss
end


model = SRRegressor(
    binary_operators=[*, +, magic_add],
    unary_operators=[shift_right, magic_inverse, neg],
    complexity_of_operators=[shift_right=>1, magic_inverse=>1, magic_add=>1],
    niterations=1000,
    ncycles_per_iteration=100,
    optimizer_nrestarts=4,
    optimizer_algorithm="NelderMead",
    parsimony=0.0,
    maxsize=50,
    adaptive_parsimony_scaling=1000, # or 100, as miles mentioned. I feel I have an unclear meaning of what this does.
    # should_optimize_constants=false, # more of a hail mary than something that will definitively improve things.
    loss_function=loss_function,
    progress=true, # ??? when loss functions are provided it doesn't print by default anymore???
)

mach = machine(model, X, y, dy)
fit!(mach)
r = report(mach)
fitted_model = fitted_params(mach)

# testing various outputs
s = plot(xlabel="X", ylabel="log(1/sqrt(X)")
scatter!(s, X.input, log.(fisr.(X.input, -5.2391f0)), label="FISR")
ds = Dataset(reshape(X.input, 1, :))
options = Options(; unary_operators=[shift_right, magic_inverse, neg], binary_operators=[*, +, magic_add])
loss_cutoff = r.losses[end] * 100 # only show equations "on the order of" the best loss
complexity_cutoff = 10 # and equations with less than this many operations

for (i, eq) in enumerate(fitted_model.equations)
    if r.losses[i] <= loss_cutoff && r.complexities[i] < complexity_cutoff
        complexity = r.complexities[i]
        loss = r.losses[i]
        x = X.input
        pred_y = eval_with_newton(eq, ds.X, options)[1]
        scatter!(s, x, log.(pred_y), label=string(complexity) * "→ logloss=" * string(round(log10(loss); digits=2)))
    end
end
plot!(s, X.input, log.(y), label="data", linewidth=4)


# deviation curve
s = plot(xlabel="X", ylabel="y_approx - y_exact")
plot!(s, X.input, log.(fisr.(X.input, -5.2391f0)) .- log.(y), label="FISR", linewidth=2)
ds = Dataset(reshape(X.input, 1, :))
options = Options(; unary_operators=[shift_right, magic_inverse, neg], binary_operators=[*, +, magic_add])
loss_cutoff = r.losses[end] * 100 # only show equations "on the order of" the best loss
complexity_cutoff = 12 # and equations with less than this many operations
for (i, eq) in enumerate(fitted_model.equations)
    if r.losses[i] <= loss_cutoff && r.complexities[i] < complexity_cutoff
        complexity = r.complexities[i]
        loss = r.losses[i]
        x = X.input
        ypred = eval_with_newton(eq, ds.X, options)[1]
        plot!(s, x, log.(ypred) .- log.(y), label=string(complexity) * "→ logloss=" * string(round(log10(loss); digits=2)))
    end
end
display(s)
plot!(s, X.input, zeros(length(y)), label="data", linewidth=4)