import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
import joblib
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# The elemental composition and proximate analysis of chosen biomass residue
fixed_features = ['C', 'H', 'O', 'N', 'FC', 'VM', 'A']
fixed_values = [42.93, 5.48, 43.83, 0.73, 15.52, 77.55, 6.93]


class HHV_EY_Problem(Problem):
    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([160, 0, 0.03]),
                         xu=np.array([300, 720, 1.00]),
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        HHV_list = []
        EY_list = []

        for row in x:
            T, RT, SL = row
            total_input = np.array(fixed_values + [T, RT, SL]).reshape(1, -1)


            classifier = joblib.load("best_rfc_model.pkl")
            cluster = classifier.predict(pd.DataFrame(total_input, columns=fixed_features + ['T', 'RT', 'SL']))[0]

            # Load the corresponding model
            scalers_X = joblib.load("H_{int(cluster + 1)}/x_scaler.pkl")
            scalers_Y = joblib.load("H_{int(cluster + 1)}/y_scaler.pkl")
            regressors = joblib.load("H_{int(cluster + 1)}/best_xgb_model.pkl")


            X_scaled = scalers_X.transform(pd.DataFrame(total_input, columns=fixed_features + ['T', 'RT', 'SL']))
            Y_pred_scaled = regressors.predict(X_scaled)
            Y_pred = scalers_Y.inverse_transform(Y_pred_scaled)[0]

            HHV, EY = Y_pred
            HHV_list.append(-HHV)
            EY_list.append(-EY)

        out["F"] = np.column_stack([HHV_list, EY_list])



problem = HHV_EY_Problem()
algorithm = NSGA2(pop_size=100)
termination = get_termination("n_gen", 100)


res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

pareto_X = []
pareto_F = []

for gen_idx, gen in enumerate(res.history):
    pop = gen.pop
    X = pop.get("X")
    F = -pop.get("F")
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pareto_X = X[nds]
    pareto_F = F[nds]

X_df = pd.DataFrame(pareto_X, columns=["T", "RT", "SL"])
F_df = pd.DataFrame(pareto_F, columns=["HHV", "EY"])

pareto_df = pd.concat([X_df, F_df], axis=1)

with pd.ExcelWriter("save path") as writer:
    pareto_df.to_excel(writer, sheet_name="Pareto Front", index=False)


print("Finish!")
