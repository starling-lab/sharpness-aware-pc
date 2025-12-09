
from packages.pfc.models.eiflow import *
from packages.pfc.models.einet  import EinsumNet

_MODELS = {
    "EinsumNet"                         :   EinsumNet,
    "LinearSplineEinsumFlow"            :   LinearSplineEinsumFlow,
    "QuadraticSplineEinsumFlow"         :   QuadraticSplineEinsumFlow,
}