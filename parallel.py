from tensorflow import keras
import numpy as np
import model
import sampledata

# https://docs.python.org/2/library/multiprocessing.html#windows

def job(param, kwargs):
    try:
        if 'data_x' in kwargs:
            data_x = kwargs['data_x']
        else:
            raise Exception("Missing x-data")
        if 'data_y' in kwargs:
            data_y = kwargs['data_y']
        else:
            raise Exception("Missing y-data")
        if 'degree' in kwargs:
            degree = kwargs['degree']
        else:
            degree = model.DEGREE_DEFAULT
        if 'polynum' in kwargs:
            polynum = kwargs['polynum']
        else:
            polynum = 1
        if 'learning_rate' in kwargs:
            learning_rate = kwargs['learning_rate']
        else:
            learning_rate = 0.1
        if 'enforce_continuity' in kwargs:
            continuity = kwargs['enforce_continuity']
        else:
            continuity = False
        if 'optimizer' in kwargs: # all arguments to Process.__init__() need to be picklable. This is not the case for keras optimizers, which is why we just pass a string
            optimizer = kwargs['optimizer']
        else:
            optimizer = 'AMSGrad'
        if 'n_epochs' in kwargs:
            n_epochs = kwargs['n_epochs']
        else:
            raise Exception('Missing epochs parameter')
        if 'factor_ck_pressure' in kwargs:
            factor_ck_pressure = kwargs['factor_ck_pressure']
        else:
            factor_ck_pressure = 0
        if 'factor_approximation_quality' in kwargs:
            factor_approximation_quality = kwargs['factor_approximation_quality']
        else:
            factor_approximation_quality = 1
        if 'factor_curvature' in kwargs:
            factor_curvature = kwargs['factor_curvature']
        else:
            factor_curvature = 0
        if 'seg_overlap' in kwargs:
            seg_overlap = kwargs['seg_overlap']
        else:
            seg_overlap = 0
        if 'gradient_regularization' in kwargs:
            gradient_regularization = kwargs['gradient_regularization']
        else:
            gradient_regularization = False
        if 'ck' in kwargs:
            ck = kwargs['ck']
        else:
            ck = 2
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            raise Exception('Missing mode!')
        if 'initialization' in kwargs:
            initialization = kwargs['initialization']
        else:
            initialization = 'zero'
        if 'seed' in kwargs:
            seed = kwargs['seed']
        else:
            seed = 0
        if 'shift_polynomial_centers' in kwargs:
            shift_polynomial_centers = kwargs['shift_polynomial_centers']
        else:
            shift_polynomial_centers = 'default'
        if 'split_uniform' in kwargs:
            split_uniform = kwargs['split_uniform']
        else:
            split_uniform = False
        if 'basis' in kwargs:
            basis = kwargs['basis']
        else:
            basis = 'power'
        if 'rescale_x_data' in kwargs:
            rescale_x_data = kwargs['rescale_x_data']
        else:
            rescale_x_data = True
        if 'ck_regularization' in kwargs:
            ck_regularization = kwargs['ck_regularization']
        else:
            ck_regularization = True
        if 'early_stopping' in kwargs:
            early_stopping = kwargs['early_stopping']
        else:
            early_stopping = False
        if 'patience' in kwargs:
            patience = kwargs['patience']
        else:
            patience = 100

        if optimizer.upper() == 'ADAM':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
        elif optimizer.upper() == 'AMSGRAD':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
        elif optimizer.upper() == 'SGD':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.upper() == 'SGD-MOMENTUM':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95)
        elif optimizer.upper() == 'SGD_POWER_BASIS':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.98)
        elif optimizer.upper() == 'FTRL':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, learning_rate_power=-1.25)
        elif optimizer.upper() == 'FTRL-EMA':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, use_ema=True, ema_overwrite_frequency=100)
        else:
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True)

        if mode == 'approx_ck':
            if param < 0 or param > 1:
                raise Exception("Factor_approximation_quality must be between 0 and 1")
            factor_approximation_quality = param
            factor_ck_pressure = 1 - param
            factor_curvature = 0
        elif mode == 'learning_rate':
            learning_rate = param
            opt.learning_rate = learning_rate
        elif mode == 'degree':
            degree = param
        elif mode == 'degree_and_learning_rate':
            degree = param[0]
            learning_rate = param[1]
            opt.learning_rate = learning_rate
            #print(f'learning rate:{learning_rate}, degree:{degree}')
        elif mode == 'numpoints_and_learning_rate':
            numpoints = param[0]
            data_x = np.linspace(0, 0.5 * np.pi, int(numpoints))
            data_y = np.sin(data_x)

            data_x = sampledata.rescale_input_data(data_x, 1)

            learning_rate = param[1]
            opt.learning_rate = learning_rate
        elif mode == 'optimizers':
            if param == "sgd":
                opt = keras.optimizers.SGD(learning_rate=learning_rate)
            elif param == "sgd-momentum":
                opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95)
            elif param == "sgd-momentum-nesterov":
                opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True)
            elif param == "adam":
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif param == "adam-amsgrad":
                opt = keras.optimizers.Adam(amsgrad=True, learning_rate=learning_rate)
            elif param == "adamax":
                opt = keras.optimizers.Adamax(learning_rate=learning_rate)
            elif param == "nadam":
                opt = keras.optimizers.Nadam(learning_rate=learning_rate)
            elif param == "ftrl":
                opt = keras.optimizers.Ftrl(learning_rate=learning_rate)
            elif param == "ftrl-ema":
                opt = keras.optimizers.Ftrl(learning_rate=learning_rate, use_ema=True, ema_overwrite_frequency=100)
            elif param == "rmsprop":
                opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
            elif param == "adafactor":
                opt = keras.optimizers.Adafactor(learning_rate=learning_rate)
            elif param == "adamw":
                opt = keras.optimizers.AdamW(learning_rate=learning_rate)
            elif param == "lion":
                opt = keras.optimizers.Lion(learning_rate=learning_rate)
            elif param == "adadelta":
                opt = keras.optimizers.Adadelta(learning_rate=learning_rate)
            elif param == "adagrad":
                opt = keras.optimizers.Adagrad(learning_rate=learning_rate)
            else:
                raise Exception(f'Invalid optimizer {param}')

        elif mode == 'sgd-momentum':
            opt = keras.optimizers.SGD(momentum=param, learning_rate=learning_rate)
        elif mode == 'sgd-nesterov-momentum':
            opt = keras.optimizers.SGD(momentum=param, nesterov=True, learning_rate=learning_rate)
        elif mode == 'amsgrad-beta1':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True, beta_1=param)
        elif mode == 'amsgrad-beta2':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True, beta_2=param)
        elif mode == 'amsgrad-weight_decay':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True, weight_decay=True)
        elif mode == 'amsgrad-ema_overwrite_frequency':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True, use_ema=True, ema_overwrite_frequency=param)
        elif mode == 'amsgrad-ema_momentum':
            opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True, ema_momentum=param, use_ema=True, ema_overwrite_frequency=100)
        elif mode == 'ftrl-lr-power':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, learning_rate_power=param)
        elif mode == 'ftrl-initial_accumulator_value':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, initial_accumulator_value=param)
        elif mode == 'ftrl-l1_regularization_strength':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, l1_regularization_strength=param)
        elif mode == 'ftrl-l2_regularization_strength':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, l2_regularization_strength=param)
        elif mode == 'ftrl-l2_shrinkage_regularization_strength':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, l2_shrinkage_regularization_strength=param)
        elif mode == 'ftrl-beta':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, beta=param)
        elif mode == 'ftrl-weight_decay':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, weight_decay=param)
        elif mode == 'ftrl-ema_overwrite_frequency':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, ema_overwrite_frequency=param, use_ema=True)
        elif mode == 'ftrl-ema_momentum':
            opt = keras.optimizers.Ftrl(learning_rate=learning_rate, ema_momentum=param, use_ema=True, ema_overwrite_frequency=100)

        elif mode == 'lambda':
            factor_approximation_quality = 1-param
            factor_ck_pressure = param
            factor_curvature = 0

        elif mode == 'lambda_enforced_continuity':
            factor_approximation_quality = 1-param
            factor_ck_pressure = param
            factor_curvature = 0
            continuity = True

        elif mode == 'lambda_and_ck_regularization_mode':
            ck_regularization = param[0]
            factor_approximation_quality = 1-param[1]
            factor_ck_pressure = param[1]
            factor_curvature = 0

        elif mode == 'lambda_and_basis':
            basis = param[0]
            factor_approximation_quality = 1-param[1]
            factor_ck_pressure = param[1]
            factor_curvature = 0

        elif mode == 'segment_size':
            data_x = sampledata.rescale_input_data(data_x, polynum*param)
            rescale_x_data = False

        elif mode == 'data_y':
            data_y = param

        elif mode == 'data_y_and_lambda':
            data_y = param[0]
            factor_approximation_quality = 1-param[1]
            factor_ck_pressure = param[1]
            factor_curvature = 0

        elif mode == 'random_init_seed':
            if param == -2:
                initialization = 'l2'
            elif param == -1:
                initialization = 'zero'
            else:
                initialization = 'random'
                seed = param

        else:
            raise Exception("Invalid mode.")

        pp = model.PP(polydegree=degree, polynum=polynum, ck=ck, shift_polynomial_centers=shift_polynomial_centers, basis=basis)

        print(".", end="")
        pp.fit(data_x, data_y, optimizer=opt, n_epochs=n_epochs, factor_approximation_quality=factor_approximation_quality,
            factor_ck_pressure=factor_ck_pressure, factor_curvature=factor_curvature, gradient_regularization=gradient_regularization, overlap_segments=seg_overlap,
            initialization=initialization, seed=seed, uniform_split=split_uniform, rescale_x_data=rescale_x_data, ck_regularization=ck_regularization,
                   early_stopping=early_stopping, patience=patience, enforce_continuity=continuity)

        print("#", end="")

        return [{'optimizer': optimizer, 'param_value': param}, pp.total_loss_values, pp.e_loss_values, pp.D_loss_values, pp.d_loss_values, pp.coeffs]

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return

if __name__ == '__main__':
    p = Process(target=sweep_param)
    p.start()
