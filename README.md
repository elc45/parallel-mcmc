## Usage
Activate environment, then run:
```bash
python examples/run_hmc.py --target banana
python examples/run_hmc.py --target ill_conditioned_gaussian --deer-config examples/configs/deer/default.json
```

## Repository Structure 
The structure of the primary source code and examples is:
```
src/                    Source code for DEER algorithms and samplers.
├── samplers.py             Defines parallel MALA and HMC samplers.
├── qdeer.py                Optimized stochastic quasi DEER implementation.
├── deer.py                 DEER implementation.
├── elk.py                  Quasi ELK implementation.
├── qdeer_leapfrog.py       Block quasi-DEER for parallel leapfrog. 
├── windowed_qdeer.py       Quasi DEER implementation with windowing.
examples/               Example scripts
```

## Examples

- **`examples/run_mala_german_credit.py`** - Runs parallel MALA using stochastic quasi DEER targeting a logistic regression model of the German Credit dataset. 
  ```
  python examples/run_mala_german_credit.py
  ```
- **`examples/run_hmc_rosenbrock.py`** - Runs parallel HMC using DEER with damping (ELK) targeting the Rosenbrock distribution.
  ```
  python examples/run_hmc_rosenbrock.py
  ```

## Citation

```
@article{zoltowski2025parallelmcmc,
      title={Parallelizing MCMC Across the Sequence Length}, 
      author={David M. Zoltowski and Skyler Wu and Xavier Gonzalez and Leo Kozachkov and Scott W. Linderman},
      year={2025},
      eprint={2508.18413},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
}
```
