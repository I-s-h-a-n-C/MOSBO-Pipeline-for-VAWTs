import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import Hypervolume
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

import config

class GenerationProgressBar(Callback):
    """Custom callback to render a tqdm progress bar and track Hypervolume."""
    def __init__(self, total_generations):
        super().__init__()
        self.pbar = tqdm(total=total_generations, desc=" -> [NSGA-II] Evolving Generations", leave=True)
        self.hv_history = []
        # Using the exact same reference point from your original code
        self.hv_metric = Hypervolume(ref_point=np.array([1.0, 2.5])) 

    def notify(self, algorithm, **kwargs):
        self.pbar.update(1)
        # Safely track hypervolume on the fly without deepcopying the surrogates
        opt_feasible = algorithm.opt.get("F") if algorithm.opt is not None else None
        if opt_feasible is not None and len(opt_feasible) > 0:
            self.hv_history.append(self.hv_metric.do(opt_feasible))
        else:
            self.hv_history.append(0.0)

    def close(self):
        if hasattr(self, 'pbar'):
            self.pbar.close()

class RiskAwareVAWTProblem(ElementwiseProblem):
    """
    Stage 9 & 10: Risk-Aware Elementwise Problem for NSGA-II.
    Objectives are penalized based on Gaussian Process prediction uncertainty (sigma).
    Includes an inequality constraint limiting TRF to <= 2.0.
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
        # 1 Constraint: Penalized TRF must be <= 2.0
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=1, xl=bounds[0], xu=bounds[1])

    def _evaluate(self, x, out, *args, **kwargs):
        # Format inputs for the surrogate models
        x_cp = np.array([x]) 
        x_trf = np.array([x[0:3]]) 
        
        # Get Predictions and Standard Deviations (Uncertainty)
        cp_pred, cp_sigma = self.cp_surrogate.predict(x_cp)
        trf_pred, trf_sigma = self.trf_surrogate.predict(x_trf)
        
        cp_val, cp_unc = cp_pred[0], cp_sigma[0]
        trf_val, trf_unc = trf_pred[0], trf_sigma[0]
        
        # ---------------------------------------------------------
        # RISK-AWARE OBJECTIVES
        # ---------------------------------------------------------
        # Minimize -(Cp - λ*σ)
        obj1 = -(cp_val - (config.LAMBDA_PENALTY * cp_unc))
        # Minimize (TRF + λ*σ)
        penalized_trf = trf_val + (config.LAMBDA_PENALTY * trf_unc)
        obj2 = penalized_trf
        
        out["F"] = [obj1, obj2]
        
        # ---------------------------------------------------------
        # NSGA-II INEQUALITY CONSTRAINT
        # ---------------------------------------------------------
        # PyMoo expects constraints in the form G(x) <= 0.
        # FIX: Now driven by config.TRF_PARETO_LIMIT so this matches the
        # SLSQP constraint below AND the final export filter. Previously
        # this was hardcoded to 2.0 while the final export filtered at 1.5,
        # which silently discarded a large portion of the optimizer's
        # output (especially high-Cp designs) before plotting/export.
        out["G"] = [penalized_trf - config.TRF_PARETO_LIMIT]

def scalarized_objective(x, cp_surrogate, trf_surrogate, weight_cp):
    """
    Hybrid Step: Combines the Risk-Aware Cp and TRF objectives into a single normalized 
    scalar value for the SLSQP gradient-based local search.
    """
    x_cp = np.array([x]) 
    x_trf = np.array([x[0:3]]) 
    
    cp_val, cp_sigma = cp_surrogate.predict(x_cp)
    trf_val, trf_sigma = trf_surrogate.predict(x_trf)
    
    c_v, c_s = cp_val[0], cp_sigma[0]
    t_v, t_s = trf_val[0], trf_sigma[0]
    
    # 1. Calculate Penalized Values
    penalized_cp = c_v - (config.LAMBDA_PENALTY * c_s)
    penalized_trf = t_v + (config.LAMBDA_PENALTY * t_s)
    
    # 2. Min-Max Normalization to [0, 1] Loss
    # FIX: Expanded bounds to absolute physical limits (0.65 for Cp, 0.0 for TRF).
    # Used max() to ensure losses NEVER drop below 0.0, preventing SLSQP runaway gradients.
    norm_cp_loss = max(0.0, (0.65 - penalized_cp) / 0.65)
    norm_trf_loss = max(0.0, penalized_trf / 5.0)
    
    # 3. Weighted Sum (0.0 to 1.0 gradient scaler)
    return (weight_cp * norm_cp_loss) + ((1.0 - weight_cp) * norm_trf_loss)

def run_optimization(cp_surrogate, trf_surrogate, df_cp):
    """
    Executes the NSGA-II optimization loop and extracts the Pareto Front,
    followed by a Gradient-Based (SLSQP) local refinement.
    """
    print("\n" + "="*50)
    print("🚀 [Stage 10] Running Risk-Aware Memetic Optimization (NSGA-II + SLSQP)")
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
    
    # =========================================================================
    # Phase 1: Global Exploration (NSGA-II)
    # =========================================================================
    problem = RiskAwareVAWTProblem(cp_surrogate, trf_surrogate, bounds=(xl, xu))
    
    algorithm = NSGA2(
        pop_size=config.POPULATION_SIZE,
        n_offsprings=config.OFFSPRING_SIZE,
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", config.GENERATIONS)
    
    print(f" -> Phase 1/2: Evolving {config.POPULATION_SIZE} designs over {config.GENERATIONS} generations (NSGA-II)...")
    print(f" -> Risk Penalty (λ) applied: {config.LAMBDA_PENALTY}")
    
    progress_callback = GenerationProgressBar(config.GENERATIONS)
    
    # Enable save_history=False to record population state across all generations
    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        callback=progress_callback,
        seed=config.RANDOM_STATE,
        save_history=False,
        verbose=False
    )
    progress_callback.close()
    
    if res.X is None:
        print("[Error] Optimizer failed to find a valid front. Constraint might be too strict.")
        return None, []
        
    print(f" -> ✅ NSGA-II discovered {len(res.X)} optimal Pareto neighborhoods.")
    
    # Calculate Hypervolume metric history across all generations
    # Reference point set worse than maximum objective boundaries: Obj1 max ~1.0, Obj2 max ~2.5
    ref_point = np.array([1.0, 2.5])
    hv_metric = Hypervolume(ref_point=ref_point)
    
    hv_history = []
    if res.history is not None:
        for gen in res.history:
            opt_feasible = gen.opt.get("F") if gen.opt is not None else None
            if opt_feasible is not None and len(opt_feasible) > 0:
                hv_val = hv_metric.do(opt_feasible)
            else:
                hv_val = 0.0
            hv_history.append(hv_val)

    # =========================================================================
    # Phase 2: Hybrid Local Refinement (SLSQP)
    # =========================================================================
    print(f" -> Phase 2/2: Initiating Gradient-Based Local Refinement (SLSQP)...")
    
    refined_X = []
    
    # Pair X coordinates with their objective scores so we can sort them
    # res.F[:, 0] is negative Cp. Smallest value = Highest Cp.
    front_data = [{'x': x, 'f_cp': f[0]} for x, f in zip(res.X, res.F)]
    front_sorted = sorted(front_data, key=lambda item: item['f_cp'])
    n_points = len(front_sorted)
    
    for i, item in enumerate(tqdm(front_sorted, desc=" -> [SLSQP] Fine-Tuning Designs")):
        x_initial = item['x']
        
        # Dynamically scale the gradient weight based on the point's position on the Pareto front.
        # i=0 gets weight 1.0 (100% focused on Cp), i=max gets weight 0.0 (100% focused on TRF)
        weight_cp = 1.0 - (i / max(1, n_points - 1))
        
        # FIX: Create a local bounding box (+/- 5% of global range) around the NSGA-II 
        # point to force SLSQP to act as a *local* refiner, strictly preserving Pareto diversity.
        local_bounds = []
        for j in range(4):
            var_range = xu[j] - xl[j]
            l_min = max(xl[j], x_initial[j] - 0.05 * var_range)
            l_max = min(xu[j], x_initial[j] + 0.05 * var_range)
            local_bounds.append((l_min, l_max))
        
        # SLSQP Inequality Constraint is formulated as fun(x) >= 0.
        # FIX: Now driven by config.TRF_PARETO_LIMIT to match the NSGA-II
        # constraint and the final export filter (previously hardcoded to
        # 2.0, which let SLSQP refine points that would later be dropped).
        def trf_constraint(x_val):
            t_pred, t_sig = trf_surrogate.predict(np.array([x_val[0:3]]))
            return config.TRF_PARETO_LIMIT - (t_pred[0] + config.LAMBDA_PENALTY * t_sig[0])
        
        opt_res = scipy_minimize(
            scalarized_objective,
            x0=x_initial,
            args=(cp_surrogate, trf_surrogate, weight_cp),
            method='SLSQP',
            bounds=local_bounds,
            constraints=[{'type': 'ineq', 'fun': trf_constraint}],
            options={'ftol': 1e-5, 'maxiter': 100} # Tight tolerance for smooth GP models
        )
        
        refined_X.append(opt_res.x)

    # =========================================================================
    # Final Compilation
    # =========================================================================
    results = []
    for x in tqdm(refined_X, desc=" -> [Post-Processing] Evaluating Final Uncertainties"):
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
    
    # HARD FILTER: Drop any designs where SLSQP slightly overstepped the boundary.
    # FIX: Now uses config.TRF_PARETO_LIMIT (same value enforced during NSGA-II
    # and SLSQP above), so this only trims genuine SLSQP overshoot rather than
    # discarding a whole band of legitimately-optimized designs.
    df_pareto = df_pareto[(df_pareto['Pred_TRF'] <= config.TRF_PARETO_LIMIT) & (df_pareto['Pred_Cp'] >= 0.35)]
    
    # Sort by raw aerodynamic performance for a logical top-to-bottom export
    df_pareto = df_pareto.sort_values(by='Pred_Cp', ascending=False).reset_index(drop=True)
    
    return df_pareto, hv_history

if __name__ == "__main__":
    print("This module requires trained surrogates to run. Execute main.py.")