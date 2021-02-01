# Production function properties (according to *Cobb-Douglas* production function)
### Q = A * L^α * K^β
### Give a penalty if expression doesn't follow **one** of it or **all**
- Parameters:
    1. A, α and β is positive constant
        1. ~~A > 0:~~
            - ~~bottom up, len(["minus"]+["-" and rightsub]) //2 == 0~~
        1. α, β is consant:
            - ~~bottom up, encounter "**" and leftsub~~
            - and which "**" rightsub is Numbers (len(get_leaves) == 0)
            1. α, β > 0: (top-down)
                - ~~"**" rightsub > 0~~
            1. α, β ≤ 1:
                - sum("**" rightsub) ≤ 1
    1. we define α + β ≤ 1
- For **INPUTS**
    1. **positive effect** for each input
    1. multipling relationship for each other
    1. sum of the expoential ≤ 1 
- For **OUTPUTS**
    1. **trade-off effect** for each output

https://www.economicsdiscussion.net/production-function/cobb-douglas-production-function-and-its-properties/23407

# Production Function Properties (Hard Constraint)
## use minus trade-off for output first
1. [ ] Function: coe = num_tree(var=None), well it should be built in scientific expression
1. [x] Function: coe * var ^ coe, s.t. coe > 0, for x: ≥ 0
1. [x] structure of Cobb-Douglas
1. [ ] short express for coe
1. [ ] mutation: Fix point mutation for certain node