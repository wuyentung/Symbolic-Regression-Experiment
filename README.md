# Experiments for symbolic regression
This repository is the research experiments for symbolic regression (SR). Where the goal of the research is to use SR to find out a function form of a unknown production function. This project is suspending in this point due to the problem complexity. You can run *experiment.py* to see the most resent version of experiment.
## Version Description
### v1 --> without elastic net (EN) 
- v1.1 --> PonyGE2( i.e. crossover and mutation after selection)
- v1.2 --> crossover and mutation before selection
### v2 --> without EN, with production function penalty in fitness
### v3 --> with EN
### v4 --> refactor tree.evaluation() to use ne.evaluate()
### v5 --> Closed_NonLinear_Regression, scipy closed-form non-linear regression, for comparing used
- v5.5 --> 20 times no change to exit
### v6 --> refactor tree and experiment method to more precise fitness
- v6.1 --> SCAD in tree.fitness()
- v6.2 --> MCP in tree.fitness()
### v7 --> refactor coefficient into continuous encoding
### vX --> efficiency update, OCBA in experiment