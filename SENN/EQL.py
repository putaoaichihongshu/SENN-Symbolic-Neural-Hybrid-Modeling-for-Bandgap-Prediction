from DySymNet import SymbolicRegression
from DySymNet.scripts.params import Params
from DySymNet.scripts.functions import *
import numpy as np

config = Params()

funcs = [Identity(), Sin(), Exp(), Square(), Pow(2), Plus(), Sub(), Product(), Div()]
n_layers = [2,3,4,5]
num_func_layer = [3,4,5,6,7]
reg_weight = 5e-3

config.funcs_avail = funcs
config.n_layers = n_layers
config.num_func_layer = num_func_layer
config.reg_weight = reg_weight


data_path = 'data/data_withoutA.csv'
SR = SymbolicRegression.SymboliRegression(config=config, func_name='Perovskite', data_path=data_path)
eq, R2, error, relative_error = SR.solve_environment()

print('Expression: ', eq)
print('R2: ', R2)
print('error: ', error)
print('relative_error: ', relative_error)
print('log(1 + MSE): ', np.log(1 + error))