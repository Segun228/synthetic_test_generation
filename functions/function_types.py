from .functions import (
    LinearFunction, 
    QuadraticFunction, 
    CubicFunction, 
    ExponentialFunction, 
    LogarithmFunction, 
    SinusFunction, 
    CosineFunction, 
    TangentFunction, 
    CotangentFunction
)


FUNCTION_TYPES = {
    "linear":{
        "k":float,
        "b":float
    },
    "quadratic":{
        "a":float,
        "b":float,
        "c":float,
    },
    "cubic":{
        "a":float,
        "b":float,
        "c":float,
        "d":float,
    },
    "exponential":{
        "a":float,
        "_lambda":float
    },
    "logarithm":{
        "k":float,
        "a":int,
        "b":int
    },
    "sinus":{
        "a":float,
        "omega":float
    },
    "cosine":{
        "a":float,
        "omega":float
    },
    "tangent":{
        "a":float,
        "omega":float
    },
    "cotangent":{
        "a":float,
        "omega":float
    }
}


FUNCTION_CLASSES = {
    "linear":LinearFunction,
    "quadratic":QuadraticFunction,
    "cubic":CubicFunction,
    "exponential":ExponentialFunction,
    "logarithm":LogarithmFunction,
    "sinus":SinusFunction,
    "cosine":CosineFunction,
    "tangent":TangentFunction,
    "cotangent":CotangentFunction
}