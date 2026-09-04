import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import config

class RiskAwareVAWTProblem(ElementwiseProblem):
    """
    Stage 9 & 10: Risk-Aware Elementwise Problem for NSGA-II.
    Objectives are penalized based on Gaussian Process prediction uncertainty (sigma).
    """
    def __init__(self, cp_surrogate, trf_surrogate, bounds):
        """
        :param cp_surrogate: Trained SurrogateBuilder instance for Cp
        :param trf_surrogate: Trained SurrogateBuilder instance for TRF
        :param bounds: Tuple of lists (xl, xu) representing lower and upper bounds of inputs
        """
        self.cp_surrogate = cp_surrogate
        self.trf_surrogate = trf_surrogate
        
        # 4 Variables: Thickness, Twist, Solidity, TSR
        # 2 Objectives: Penalized Cp, Penalized TRF
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=0, xl=bounds[0], xu=bounds[1])

    def _evaluate(self, x, out, *args, **kwargs):
        # Format inputs for the surrogate models
        # Cp needs all 4 vars (Thickness, Twist, Solidity, TSR)
        x_cp = np.array([x]) 
        
        # TRF only depends on the geometry, not operating speed (Thickness, Twist, Solidity)
        x_trf = np.array([x[0:3]]) 
        
        # Get Predictions and Standard Deviations (Uncertainty)
        cp_pred, cp_sigma = self.cp_surrogate.predict(x_cp)
        trf_pred, trf_sigma = self.trf_surrogate.predict(x_trf)
        
        # Extract scalar values from the arrays
        cp_val = cp_pred[0]
        cp_unc = cp_sigma[0]
        
        trf_val = trf_pred[0]
        trf_unc = trf_sigma[0]
        
        # ---------------------------------------------------------
        # RISK-AWARE OBJECTIVES (Lower Confidence Bound formulation)
        # ---------------------------------------------------------
        # Pymoo minimizes by default. 
        # To maximize Cp, we minimize -Cp. We penalize uncertainty by adding λ*σ (making it worse)
        # Therefore: Minimize -(Cp - λ*σ)
        obj1 = -(cp_val - (config.LAMBDA_PENALTY * cp_unc))
        
        # To minimize TRF, we simply minimize TRF. We penalize uncertainty by adding λ*σ
        # Therefore: Minimize (TRF + λ*σ)
        obj2 = trf_val + (config.LAMBDA_PENALTY * trf_unc)
        
        # Assign to output
        out["F"] = [obj1, obj2]

def run_optimization(cp_surrogate, trf_surrogate, df_cp):
    """
    Executes the NSGA-II optimization loop and extracts the Pareto Front.
    Requires the original dataframe to extract the physical min/max bounds.
    """
    print("\n" + "="*50)
    print("🚀 [Stage 10] Running Risk-Aware NSGA-II Optimization")
    print("="*50)
    
    # 1. Dynamically extract bounds from the training data
    xl = [
        df_cp['thickness'].min(), 
        df_cp['twist'].min(), 
        df_cp['solidity'].min(), 
        df_cp['tsr'].min()
    ]
    xu = [
        df_cp['thickness'].max(), 
        df_cp['twist'].max(), 
        df_cp['solidity'].max(), 
        df_cp['tsr'].max()
    ]
    
    # 2. Initialize Problem and Algorithm
    problem = RiskAwareVAWTProblem(cp_surrogate, trf_surrogate, bounds=(xl, xu))
    
    algorithm = NSGA2(
        pop_size=config.POPULATION_SIZE,
        n_offsprings=config.OFFSPRING_SIZE,
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", config.GENERATIONS)
    
    # 3. Execute Optimization
    print(f" -> Evolving {config.POPULATION_SIZE} designs over {config.GENERATIONS} generations...")
    print(f" -> Risk Penalty (λ) applied: {config.LAMBDA_PENALTY}")
    
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=config.RANDOM_STATE,
        save_history=False,
        verbose=False
    )
    
    if res.X is None:
        print("[Error] Optimizer failed to find a valid front.")
        return None
        
    print(f" -> ✅ Discovered {len(res.X)} optimal Pareto designs.")
    
    # 4. Compile Results into a DataFrame for easy export and analysis (Stage 11 prep)
    # We re-evaluate the raw (unpenalized) surrogate values just for the final table display,
    # so the engineer sees the actual predicted performance, not the distorted objective value.
    results = []
    for x in res.X:
        cp_pred, cp_sigma = cp_surrogate.predict(np.array([x]))
        trf_pred, trf_sigma = trf_surrogate.predict(np.array([x[0:3]]))
        
        results.append({
            'Thickness': x[0],
            'Twist': x[1],
            'Solidity': x[2],
            'TSR': x[3],
            'Pred_Cp': cp_pred[0],
            'Sigma_Cp': cp_sigma[0],
            'Pred_TRF': trf_pred[0],
            'Sigma_TRF': trf_sigma[0]
        })
        
    df_pareto = pd.DataFrame(results)
    
    # Sort by raw aerodynamic performance for a logical top-to-bottom reading
    df_pareto = df_pareto.sort_values(by='Pred_Cp', ascending=False).reset_index(drop=True)
    
    return df_pareto

if __name__ == "__main__":
    print("This module requires trained surrogates to run. Execute main.py.")