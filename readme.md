# Machine Learning Optimized Univariate Piecewise Polynomial Approximation for Use in Cam Approximation 

**Experimental** Python code developed for research on:  
  
_H. Waclawek and S. Huber, “Machine Learning Optimized Orthogonal
Basis Piecewise Polynomial Approximation,” in Learning and Intelligent Op-
timization, Cham: Springer Nature Switzerland, 2025, pp. 427–441. DOI: 10.
1007/978-3-031-75623-8_33_

See _citation.bib_ for details on citation.  
This project is licensed under the terms of the MIT license.  

 <!-- - https://doi.org/10.1007/978-3-031-75623-8_33
 - https://doi.org/10.48550/arXiv.2403.08579
 - https://doi.org/10.1007/978-3-031-25312-6_68 -->

Allows fitting of univariate piecewise polynomials of arbitrary degree.  
$C^k$-continuity requires degree $2k+1$.  
Supports $2$ Bases:  
 - Power Basis (non-orthogonal)  
 - Chebyshev Basis (orthogonal)  

![OptimizeVarSin](/fig/chebyshev_sin_var_with_loss.gif)

## Usage

### Basics

1. Fitting using 'model' module:

```py
import model
import numpy as np
from tensorflow import keras

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

pp = model.PP(polydegree=7, polynum=3, ck=3, basis='chebyshev')
opt = keras.optimizers.Adam(amsgrad=True, learning_rate=0.1)

alpha = 0.01 # optimization: how much emphasis do we want to put on continuity?
pp.fit(x, y, optimizer=opt, n_epochs=600, factor_approximation_quality=1-alpha, factor_ck_pressure=alpha, early_stopping=True, patience=100)
```

Note: x-data will be rescaled so that every polynomial segment spans a range of $2$.  
We can evaluate generated PPs for specific x-ranges or individual x-values:  

```py
import matplotlib.pyplot as plt

x_range = np.linspace(2, 4, 50)
y = pp.evaluate_pp_at_x(x_range, deriv=0)
plt.plot(x_range,y)
```


2. Plotting using 'plot' module:

```py
import plot

plot.plot_pp(pp)
plot.plot_loss(pp)
```

We can plot specific derivatives or losses of specific optimization targets:

```py
plot.plot_pp(pp, deriv=1)
plot.plot_loss(pp, type='continuity-total')
plot.plot_loss(pp, type='continuity-derivatives')
plot.plot_loss(pp, type='approximation')
```

3. Initialization from coefficients

```py
pp_new = model.get_pp_from_coeffs(pp.coeffs, x, y, basis='chebyshev', ck=3)
plot.plot_pp(pp_new)
```

### Useful stuff

1. Parallel execution

The 'parallel' module offers functions for parallel experiment execution.  
All return values are pickleable for execution on Windows.  
E.g. comparing performance of different optimizers:  

```py
import parallel
import multiprocessing as mp
from itertools import repeat

optimizers = ['sgd', 'sgd-momentum', 'sgd-momentum-nesterov', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax', 'nadam', 'adam-amsgrad', 'adafactor', 'adamw', 'ftrl', 'lion']

kwargs = {'data_x': x, 'data_y': y, 'polynum': 3, 'ck': 3, 'degree': 7,
        'n_epochs': 600, 'learning_rate': 0.01 , 'mode': 'optimizers',
        'factor_approximation_quality': 1-alpha, 'factor_ck_pressure': alpha,
        'basis': 'chebyshev'}

pool = mp.Pool(mp.cpu_count())
results = pool.starmap(parallel.job, zip(optimizers, repeat(kwargs)))

losses = []

for i in range(len(results)):
    losses.append(results[i][1])

fig, axes = plt.subplots(4, (len(optimizers)+2)//4)
axes = axes.flatten()
fig.set_figwidth(len(optimizers)*3)
fig.set_figheight(20)
fig.suptitle(f'Losses over epochs with different optimizers')

for i, opt in enumerate(optimizers):
    ax = axes[i]
    ax.set_title("%s" % opt)
    ax.semilogy(losses[i])
```

2. Creating animations

```py
import animate

animate.create_animation(filepath='pp_animation', pp=pp, basis='chebyshev', shift_polynomial_centers='mean', plot_loss=True)
```
