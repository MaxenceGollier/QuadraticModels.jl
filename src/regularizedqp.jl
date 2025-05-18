export RegularizedQuadraticModel

mutable struct RegularizedQuadraticModel{T, S, M1, M2} <: AbstractQuadraticModel{T, S}
  model::QuadraticModel{T, S, M1, M2}
  σ::T
end

function RegularizedQuadraticModel(nlp::QuadraticModel{T, S, M1, M2}; σ::T = zero(T)) where{T, S, M1, M2}
  return RegularizedQuadraticModel(nlp, σ)
end

@default_counters RegularizedQuadraticModel model

function Base.getproperty(nlp::RegularizedQuadraticModel, name::Symbol)
  if name == :model || name == :σ
    return getfield(nlp, name)
  else
    return getproperty(nlp.model, name)
  end
end

function Base.setproperty!(nlp::RegularizedQuadraticModel, name::Symbol, value)
  if name == :model || name == :σ
    return setfield!(nlp, name, value)
  else
    return setproperty!(nlp.model, name, value)
  end
end

function NLPModels.obj(qp::RegularizedQuadraticModel{T, S}, x::AbstractVector) where {T, S}
  NLPModels.obj(qp.model, x) + qp.σ*dot(x, x)/2
end

function NLPModels.grad!(qp::RegularizedQuadraticModel{T, S}, x::AbstractVector, g::AbstractVector) where{T, S}
  NLPModels.grad!(qp, x, g)
  g .+= qp.σ .* x
end

