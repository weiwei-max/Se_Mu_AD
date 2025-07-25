# Second-Order Multi-Modal Fusion and Dynamic Optimization for Robust Multi-Task Autonomous Driving

A structured multi-task learning framework for autonomous driving tasks, supporting task affinity estimation, dynamic grouping, and end-to-end control strategy training and evaluation in the CARLA environment. The framework integrates modules such as trajectory generation and is suitable for multi-task optimization scenarios in the CARLA simulator and autonomous driving research.

## 🔧 Environment Setup

Please use `conda` to create the required virtual environment:

```bash
conda env create -f environment.yml
conda activate semuad
```

Also, install CARLA and its dependencies:
```
bash setup_carla.sh
```

##  Training & Evaluation
1. Generate training data
 ```
python generate_dataset_slurm.py
```

2. Run evaluation
```
python evaluate_routes_slurm.py
```

## Project Structure
```
Se_Mu_AD/
│
├── team_code/                  
├── leaderboard/                
├── scenario_runner/           
├── tools/                      
├── docs/                       
├── assets/                    
│
├── generate_dataset_slurm.py  
├── evaluate_routes_slurm.py   
├── evaluate_routes_slurm_dw.py
├── environment.yml            
├── setup_carla.sh             
├── LICENSE                    
```

