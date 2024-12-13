import matplotlib.pyplot as plt
import pyddm as ddm
import numpy as np
import pandas as pd
from pyddm import Model, Fittable, plot, InitialCondition
from pyddm.functions import fit_adjust_model, display_model, fit_model
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, LossRobustBIC, LossRobustLikelihood
from pyddm.functions import fit_adjust_model

# BASIC
class DriftCoherence(ddm.models.Drift):
    name = "Drift depends linearly on AMRate distance from midpoint"
    required_parameters = ["drifttone"]
    required_conditions = ["AMRate"]

    def get_drift(self, conditions, **kwargs):
        return self.drifttone * (np.abs(np.log(conditions['AMRate']) - np.log(6.25)))

model_basic = Model(name='Drift varies with AMRate distance from midpoint',
                 drift=DriftCoherence(drifttone=Fittable(minval=0, maxval=20)),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                 overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8)),
                                                OverlayPoissonMixture(pmixturecoef=.02,
                                                                      rate=1)]),
                 dx=.001, dt=0.001, T_dur=2.5, choice_names = ("Left", "Right"))

model_basic_no_poiss = Model(name='Drift varies with AMRate distance from midpoint, no poisson mixture',
                 drift=DriftCoherence(drifttone=Fittable(minval=-20, maxval=20)),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=Fittable(minval=.1, maxval=2)),
                 overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=-0.8, maxval=0.8))]),
                 dx=.001, dt=0.001, T_dur=2.5)

# BIAS
class ICPointSideBias(InitialCondition):
    name = "A starting point with a left or right bias."
    required_parameters = ["x0"]
    required_conditions = ["left_is_correct"]
    def get_IC(self, x, dx, conditions):
        start = np.round(self.x0/dx)
        # Positive bias for left choices, negative for right choices
        if not conditions['left_is_correct']:
            start = -start
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf

model_bias = Model(name='Biased model where drift varies with AMRate distance from midpoint',
                 drift=DriftCoherence(drifttone=Fittable(minval=0, maxval=20)),
                 noise=NoiseConstant(noise=1),
                 bound=BoundConstant(B=Fittable(minval=1, maxval=10)),
                 IC=ICPointSideBias(x0=Fittable(minval=0, maxval=1)),
                 overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8)),
                                                OverlayPoissonMixture(pmixturecoef=.02,
                                                                      rate=1)]),
                 dx=.001, dt=0.001, T_dur=2.5)