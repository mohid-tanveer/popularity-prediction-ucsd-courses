import itertools
import copy

def grid_search_lfm(model_class, train_data, valid_data, pretrained_params=None):
    """
    Performs a grid search with Warm Start capability.
    
    Args:
        model_class: The class constructor for latent_factor_model.
        train_data: List of training samples.
        valid_data: List of validation samples.
        pretrained_params: Dict containing {'alpha': ..., 'beta_U': ..., 'beta_I': ...} 
                           from a previously trained baseline model.
        
    Returns:
        dict: The best hyperparameters found.
    """
    
    # 1. Parameter Grid
    param_grid = {
        'n_factors': [5, 10, 20],
        'lr': [0.001, 0.01, 0.02],
        'reg': [0.01, 0.02, 0.1]
    }
    
    # Fixed parameters
    FIXED_PARAMS = {
        'n_epochs': 10,
        'shuffle': True,
        'seed': 42
    }

    best_mse = float('inf')
    best_params = {}
    
    combinations = list(itertools.product(*param_grid.values()))
    keys = param_grid.keys()
    
    print(f"Starting Warm-Start Grid Search with {len(combinations)} combinations...")
    print("-" * 75)
    print(f"{'n_factors':<10} {'lr':<10} {'reg':<10} | {'Valid MSE':<10}")
    print("-" * 75)

    # 2. Iterate
    for values in combinations:
        current_grid_params = dict(zip(keys, values))
        full_params = {**current_grid_params, **FIXED_PARAMS}
        
        # Instantiate model
        model = model_class(**full_params)
        
        # --- WARM START INJECTION ---
        if pretrained_params:
            # Inject alpha if provided
            if 'alpha' in pretrained_params:
                model.alpha = pretrained_params['alpha']
            
            # Inject deep copies of beta_U and beta_I to prevent mutation across loops
            if 'beta_U' in pretrained_params:
                model.beta_U = copy.deepcopy(pretrained_params['beta_U'])
            if 'beta_I' in pretrained_params:
                model.beta_I = copy.deepcopy(pretrained_params['beta_I'])
        # ----------------------------
        
        # Train
        model.fit(train_data)
        
        # Evaluate
        mse = 0.0
        for u, i, r in valid_data:
            prediction = model.predict(u, i)
            mse += (r - prediction) ** 2
        mse /= len(valid_data)
        
        print(f"{current_grid_params['n_factors']:<10} {current_grid_params['lr']:<10} {current_grid_params['reg']:<10} | {mse:.5f}")
        
        if mse < best_mse:
            best_mse = mse
            best_params = current_grid_params

    print("-" * 75)
    print(f"Best MSE: {best_mse:.5f}")
    
    return {**best_params, **FIXED_PARAMS}