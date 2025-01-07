# !/usr/bin/env python

"""model.py: Fitting piecewise polynomials of arbitrary degree and C^k-continuity.
             Perform optimization using gradient descent optimization.
"""
import numpy
import numpy as np
import sklearn.preprocessing as preprocessing
import tensorflow as tf
import numpy.polynomial.polynomial as poly
import numpy.polynomial.chebyshev as cheby
import math
import statistics
import time
import copy
import sys

if hasattr(sys.modules["__main__"], "get_ipython"):
    from tqdm import notebook as tqdm
else:
    import tqdm


__author__ = "Hannes Waclawek"
__version__ = "3.1"
__email__ = "hannes.waclawek@fh-salzburg.ac.at"


# Maximum values
POLY_DEGREE_MAX = 19
POLY_NUM_MAX = 128

# Default constructor values
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS = 500
DEFAULT_OVERLAP = 0
DEFAULT_CURVATURE_FACTOR = 0
DEFAULT_CK_PRESSURE_FACTOR = 0.05
DEFAULT_APPROXIMATION_QUALITY_FACTOR = 0.95
DEFAULT_PATIENCE = 100

# Other Constants
SHIFT_POLYNOMIAL_CENTER_MEAN = 0
SHIFT_POLYNOMIAL_CENTER_BOUNDARY = 1
SHIFT_POLYNOMIAL_CENTER_OFF = 2

# Polynomial Basis
POWER_BASIS = 0
CHEBYSHEV = 1
SHIFTED_CHEBYSHEV = 2
LEGENDRE = 3
SHIFTED_LEGENDRE = 4

# Basis maximum segment size
MAX_SEGMENT_SIZE_POWER = 2.1
MAX_SEGMENT_SIZE_CHEBYSHEV = 2.1
MAX_SEGMENT_SIZE_SHIFTED_CHEBYSHEV = 1.1
MAX_SEGMENT_SIZE_LEGENDRE = 2.1
MAX_SEGMENT_SIZE_SHIFTED_LEGENDRE = 1.1

# Basis degrees
DEGREE_DEFAULT = 0
DEGREE_POWER = 5
DEGREE_CHEBYSHEV = 6
DEGREE_SHIFTED_CHEBYSHEV = 5

# Basis ideal segment size
SEGMENT_SIZE_POWER = 2.0
SEGMENT_SIZE_CHEBYSHEV = 2.0
SEGMENT_SIZE_SHIFTED_CHEBYSHEV = 1.0 # 1.0
SEGMENT_SIZE_LEGENDRE = 2.0
SEGMENT_SIZE_SHIFTED_LEGENDRE = 1.0


def get_pp_from_coeffs(coeffs, data_x, data_y, basis, ck, shift_polynomial_centers='default',
                           total_loss_values = None, e_loss_values = None, D_loss_values = None, d_loss_values = None,
                           rescale_x_data = True):
    '''Generate pp from existing pp coefficients.
    Expected polynomial coefficients order is from lowest (index 0) to highest.
    See pp class for parameter description.'''
    s = PP(polydegree=coeffs[0].get_shape()[0] - 1, polynum=len(coeffs), ck=ck, basis=basis,
               shift_polynomial_centers=shift_polynomial_centers)

    s.rescale_x_data = rescale_x_data

    s.data_x = data_x
    s.data_y = data_y

    s._initialize_pp_data()

    s.coeffs = coeffs
    s.performedfit = True

    if total_loss_values is not None:
        s.total_loss_values = total_loss_values
        s.epochs = len(total_loss_values)
    s.e_loss_values = e_loss_values
    s.D_loss_values = D_loss_values
    s.d_loss_values = d_loss_values

    return s


def get_pp_from_numpy_coeffs(coeffs, data_x, data_y, ck, basis='power', rescale_x_data = True):
    '''Generate pp from existing pp coefficients.
    Expected polynomial coefficients order is from lowest (index 0) to highest.
    See pp class for parameter description.'''
    s = PP(polydegree=len(coeffs[0]) - 1, polynum=len(coeffs), ck=ck, basis=basis)

    s.rescale_x_data = rescale_x_data

    s.data_x = data_x
    s.data_y = data_y

    s._initialize_pp_data()

    s.coeffs = [0.0] * len(coeffs)

    for i, c in enumerate(coeffs):
        s.coeffs[i] = tf.Variable(c, dtype='float64', trainable=True)

    s.performedfit = True

    return s


class PP():
    """Main class containing coefficients and methods to fit and evaluate pps
    """
    def __init__(self, polydegree=DEGREE_DEFAULT, polynum=1, ck=2, basis='chebyshev', shift_polynomial_centers='default'):
        """
        :param polydegree: Polynomial order of individual functional pieces
        :param polynum: Number of polynomial pieces
        :param ck: C^k-continuity, e.g. 2 = C^2-continuous pp. Requires minimum polydegree 2k + 1
        :param basis: Determines the polynomial base. Options are 'power' for Power Basis, 'chebyshev' for Chebyshev Polynomials
        :param shift_polynomial_centers: Shifts polynomial centers to the range of their respective segment.
        This enables convergence especially for segments with higher x-values. "mean" shifts polynomials to the mean of their respective segment.
        This is the default for Power Basis polynomials, as it shows the best results. "boundary" shifts polynomials to the left boundary of their respective segment.
        This is the default for Chebyshev and Legendre Basis, providing the best results with those.
        With "default", the default for the respective basis is chosen.
        Example: Shifting to boundary --> Segment [4, 7] --> evaluation call will be done with f(x-4)
        """
        # pp parameters
        if basis is None:
            self.polynomial_base = POWER_BASIS
        elif basis == 'chebyshev':
            self.polynomial_base = CHEBYSHEV
        elif basis == 'shifted_chebyshev':
            self.polynomial_base = SHIFTED_CHEBYSHEV
        elif basis == 'legendre':
            self.polynomial_base = LEGENDRE
        elif basis == 'shifted_legendre':
            self.polynomial_base = SHIFTED_LEGENDRE
        else:
            self.polynomial_base = POWER_BASIS

        self._initialize_polynomial_center()

        # Polynomial order of individual functional pieces
        if polydegree == DEGREE_DEFAULT:
            self.polydegree = self._get_default_degree()
        else:
            self.polydegree = polydegree

        # Number of polynomial pieces
        self.polynum = polynum
        self.ck = ck
        self.clamped = False
        self.periodic = False

        if self.polynum <= 0 or self.polynum > POLY_NUM_MAX:
            raise Exception("Invalid polynomial count - Must be 1 <= polynum <= " + str(POLY_NUM_MAX))

        if self.polydegree <= 0 or self.polydegree > POLY_DEGREE_MAX:
            raise Exception("Invalid polynomial degree - Must be 1 <= polydegree <= " + str(POLY_DEGREE_MAX))

        # Parameters
        self.epochs = DEFAULT_EPOCHS
        self.overlap_segments = DEFAULT_OVERLAP
        self.rescale_x_data = False
        self.gradient_regularization = False
        self.ck_regularization = True
        self.factor_ck_pressure = DEFAULT_CK_PRESSURE_FACTOR
        self.factor_approximation_quality = DEFAULT_APPROXIMATION_QUALITY_FACTOR
        self.factor_curvature = DEFAULT_CURVATURE_FACTOR

        # Early stopping and restoring of model parameters
        self.early_stopping = False
        self.patience = DEFAULT_PATIENCE
        self.current_patience = 0
        self.best_loss_index = 1 # No stop at initialization values

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE, amsgrad=True)

        self.initialization = 'zero'

        self.enforce_continuity = False
        self.verbose = False

        # Data arrays
        self.data_x = []
        self.data_y = []
        self.data_x_split = [[]]
        self.data_y_split = [[]]
        self.data_x_split_overlap = [[]]
        self.data_y_split_overlap = [[]]
        self.boundary_points = []

        # Internal variables
        self.performedfit = False
        self.initial_data_array_length = 0

        # Power basis Ck-loss regularization by highest order term factor for each derivative
        self.ck_regularization_power_basis = [math.factorial(self.polydegree)/math.factorial(self.polydegree-k) for k in range(self.polydegree)]

        # TF oefficients array = TF training vars, self.coeffs[1][0] = polynomial segment 1, lowest order coefficient.
        # Polynomial coefficients are ordered from low to high.
        self.coeffs = [None for _ in range(self.polynum)]
        # Record of coeffs of each training epoch for later reconstruction
        self.recorded_coeffs = [[]]
        self.record_coefficients = False

        # Derivatives of chebyshev polynomials
        # self.chebyshev_polynomials[1][2] = 1st Chebyshev polynomial, 2nd derivative
        self.chebyshev_polynomials = []

        # Initialize loss arrays (For plotting loss functions after optimization)
        self.total_loss_values = []  # combined loss
        self.I_loss_values = []  # loss for integrate_squared_pp_acceleration()
        self.D_loss_values = []  # loss for ck_pressure() - total
        self.d_loss_values = [[]]  # loss for ck_pressure() - loss_d
        self.e_loss_values = []  # loss for sum of squared approximation errors

        # Set polynomial center for segments
        if shift_polynomial_centers is None:
            pass
        elif shift_polynomial_centers == 'default':
            pass
        elif shift_polynomial_centers == 'off':
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_OFF
        elif shift_polynomial_centers == 'mean':
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_MEAN
        elif shift_polynomial_centers == 'boundary':
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_BOUNDARY

    def properties(self):
        print(f'Polynomial degree: {self.polydegree}')
        print(f'Number of polynomial segments: {self.polynum}')
        print(f'Clamped: {self.clamped}')
        print(f'Segment overlap: {self.overlap_segments}')
        print(f'C{self.ck}-continuous')
        print(f'Optimization epochs: {self.epochs}')

    def _sum_squared_errors(self):
        """Calculate sum of squared errors (derivative 0) with regards to self.data_y.
        Invariant to length of data points array.       
        :return: derivative 0 squared error
        """
        total_cost = 0

        for i in range(self.polynum):
            y1 = self._evaluate_polynomial_at_x(i, 0, self.data_x_split[i], self._polynomial_center(i))
            #y1 = self._evaluate_polynomial_at_x(i, 0, self.data_x_split_overlap[i], self._polynomial_center(i))
            y2 = self.data_y_split[i]
            #y2 = self.data_y_split_overlap[i]
            e = tf.subtract(y1, y2)
            total_cost += tf.reduce_sum(tf.multiply(e,e))

        return tf.divide(total_cost, len(self.data_x))

    def ck_pressure(self):
        """Cost function: "C^k-pressure"
        (= distance between segment endpoints in continuous derivatives (see self.ck))       
        :return: Sum of total C^k error values (Invariant to number of boundary points), C^k error array for self.ck derivatives"""
        if not self.performedfit:
            raise Exception("No pp data - Perform fit() first")

        if self.polynum < 2:
            return tf.constant(0.0, dtype=tf.dtypes.float64), [tf.constant(0.0, dtype=tf.dtypes.float64)] * (self.ck + 1), tf.constant(0.0, dtype=tf.dtypes.float64)

        total_cost = 0
        regularization_free_total_cost = 0
        cost_d = [0.0] * (self.ck + 1)

        for k in range(self.ck + 1):
            for i in range(self.polynum - 1):
                y1 = self._evaluate_polynomial_at_x(i, k, self.boundary_points[i+1],
                                                    self._polynomial_center(i))
                y2 = self._evaluate_polynomial_at_x(i + 1, k, self.boundary_points[i+1],
                                                    self._polynomial_center(i + 1))

                # loss according to regularization setting
                if self.ck_regularization:
                    if self.polynomial_base == POWER_BASIS:
                        total_cost += tf.square((y1 - y2)/self.ck_regularization_power_basis[k])
                    # Shifted Chebyshev actually misses symmetry properties due to the interval [0, 1] and would need a separate factor for left and right evaluation
                    elif self.polynomial_base == CHEBYSHEV or self.polynomial_base == SHIFTED_CHEBYSHEV:
                        #total_cost += tf.square((y1 - y2)/self.chebyshev_polynomials[self.polydegree][k](1))
                        total_cost += tf.square((y1 - y2) / self.ck_regularization_power_basis[k]) # Experiments show that this regularization method works better also for Chebyshev basis. However, SGD will not converge.
                    else:
                        total_cost += tf.square((y1 - y2))/(math.factorial(k+1))
                else:
                    total_cost += tf.square((y1 - y2))

                cost_d[k] += tf.divide(tf.square((y1 - y2)), (self.polynum - 1)) # regularization-free derivative specific loss
                regularization_free_total_cost += tf.square((y1 - y2)) # regularization-free total loss

        return tf.divide(total_cost, (self.polynum - 1)), cost_d, tf.divide(regularization_free_total_cost, (self.polynum - 1))  # divide with number of segments

    def integrate_squared_pp_acceleration(self):
        """ Perform integration of stored pp coefficients' 2nd derivative
            :return: total curvature value of pp
        """
        if not self.performedfit:
            raise Exception("No pp data - Perform fit() first")

        ret = 0

        for i in range(0, self.polynum):
            x0 = self._polynomial_center(i)
            for j in range(2, self.polydegree + 1):
                for k in range(2, self.polydegree + 1):
                    ret += self.coeffs[i][j] * self.coeffs[i][k] * ((j * k * (j - 1) * (k - 1)) / (j + k - 3)) * (
                                (self.data_x_split[i][-2] - x0) ** (j + k - 3) - (self.data_x_split[i][0] - x0) ** (
                                    j + k - 3))

        return ret

    def _initialize_polynomial_center(self):
        if self.polynomial_base is None:
            self.polynomial_base = POWER_BASIS
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_MEAN
        elif self.polynomial_base == CHEBYSHEV:
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_MEAN
        elif self.polynomial_base == SHIFTED_CHEBYSHEV:
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_BOUNDARY
        elif self.polynomial_base == LEGENDRE:
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_MEAN
        elif self.polynomial_base == SHIFTED_LEGENDRE:
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_BOUNDARY
        else:
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_MEAN

    def _get_boundary_points_uniform(self):
        """Equally divides input space
        """
        self.boundary_points = np.linspace(self.data_x[0], self.data_x[-1], self.polynum + 1)

    def _get_boundary_points_non_uniform(self):
        """Boundary points lie on input points
        """
        if self.polynum <= 0:
            raise Exception("Invalid polynomial count")

        if self.polynum == 1:
            self.boundary_points = [self.data_x[0], self.data_x[-1]]
            return

        data_x_split = np.array_split(self.data_x, self.polynum)

        self.boundary_points.append(data_x_split[0][0])

        for i in range(len(data_x_split)):
            self.boundary_points.append(data_x_split[i][-1])

    def _split_input_arrays_with_boundary_points(self, input_arr):
        output_arr = [[]]
        i = 1
        j = 0
        while i < len(self.boundary_points):
            # add left boundary point
            if j <= (len(input_arr) - 1) and j != 0:
                output_arr[i - 1].append(input_arr[j-1])

            while (self.data_x[j] < self.boundary_points[i]) or (math.isclose(self.data_x[j], self.boundary_points[i])):
                output_arr[i-1].append(input_arr[j])
                j += 1
                if j > (len(input_arr) - 1):
                    break
            i += 1
            if i < len(self.boundary_points):
                output_arr.append([])

        # create list (since length of arrays may vary and numpy array of arrays requires uniform length) of numpy arrays
        # we require numpy arrays in evaluate_polynomial_at_x()
        ret = []
        for segment in output_arr:
            ret.append(np.array(segment, dtype=segment[0]))

        return ret

    def _rescale_input_data(self, data_x, max_value):
        """Scale 2D input data axes to [0, max_value]."""
        min_max_scaler = preprocessing.MinMaxScaler((0, max_value))

        if not isinstance(data_x, (np.ndarray)):
            data_x = np.array(data_x)

        x = data_x.reshape(-1, 1)
        data_x = min_max_scaler.fit_transform(x)
        data_x = data_x.flatten()

        return data_x

    def _rescale_x_data(self):
        segment_size = 0.0
        if self.polynomial_base == POWER_BASIS:
            segment_size = SEGMENT_SIZE_POWER
        elif self.polynomial_base == CHEBYSHEV:
            segment_size = SEGMENT_SIZE_CHEBYSHEV
        elif self.polynomial_base == SHIFTED_CHEBYSHEV:
            segment_size = SEGMENT_SIZE_SHIFTED_CHEBYSHEV
        elif self.polynomial_base == LEGENDRE:
            segment_size = SEGMENT_SIZE_LEGENDRE
        elif self.polynomial_base == SHIFTED_LEGENDRE:
            segment_size = SEGMENT_SIZE_SHIFTED_LEGENDRE

        self.data_x = self._rescale_input_data(self.data_x, self.polynum*segment_size)

    def _split_data(self, overlap=0):
        """Split data into POLY_NUM segments. Only used for overlapping segments.
        :param overlap: Specify number of points of segment n to be contained in adjacent segments n-1 and n+1
                        in order to better align the curve's derivatives at transition points.
                        (Reduce "C^k-pressure")
                        Must be in the range 0 < overlap <= 1.

                        Examples:
                        overlap = 1: All points of segment n are contained in adjacent segments n-1 and n+1
                        overlap = 0.5: 50% of points of segment n are contained in adjacent segments n-1 and n+1
                        overlap = 0.1: 10% of points of segment n are contained in adjacent segments n-1 and n+1
                        overlap = 0: Segments only share boundary points
                        Result is rounded up
        """
        self._get_boundary_points_non_uniform()

        if overlap < 0 or overlap > 1:
            raise Exception("Overlap must be between 0 and 1")

        if self.polynum <= 0:
            raise Exception("Invalid polynomial count")

        if self.polynum == 1:
            self.data_x_split[0] = self.data_x
            self.data_y_split[0] = self.data_y
            self.data_x_split_overlap = self.data_x_split.copy()
            self.data_y_split_overlap = self.data_y_split.copy()
            return

        # 2 Data points per segment required
        if (len(self.data_x) / self.polynum) < 2:
            raise Exception("Not enough data points for polynomial count")

        # boundary points initialization required
        if len(self.boundary_points) < 2:
            raise Exception("Internal error: Boundary points not defined")

        # split into chunks
        self.data_x_split = self._split_input_arrays_with_boundary_points(self.data_x)
        self.data_y_split = self._split_input_arrays_with_boundary_points(self.data_y)

        #self._check_max_segment_size()

        # # split into chunks
        # self.data_x_split = np.array_split(self.data_x, self.polynum)
        # self.data_y_split = np.array_split(self.data_y, self.polynum)
        #
        # self.initial_data_array_length = len(self.data_x_split[0])

        overlapping_points = int(math.ceil(len(self.data_x_split[0]) * overlap))

        if overlapping_points > 0:
            # add segment 1 with overlapping points to the right
            self.data_x_split_overlap = [np.append(self.data_x_split[0],
                                                             self.data_x_split[1][0: overlapping_points], axis=0)]
            self.data_y_split_overlap = [np.append(self.data_y_split[0],
                                                             self.data_y_split[1][0: overlapping_points], axis=0)]

            for i in range(1, len(self.data_x_split)):
                # add last segment with overlapping points to the left
                if i == (len(self.data_x_split) - 1):
                    self.data_x_split_overlap.append(np.insert(self.data_x_split[i], 0,
                                                             self.data_x_split[i - 1][-1 * overlapping_points:], axis=0))
                    self.data_y_split_overlap.append(np.insert(self.data_y_split[i], 0,
                                                             self.data_y_split[i - 1][-1 * overlapping_points:], axis=0))
                # add segments with overlapping points on both sides
                else:
                    add_x = np.append(self.data_x_split[i], self.data_x_split[i + 1][0: overlapping_points], axis=0)
                    add_x = np.insert(add_x, 0, self.data_x_split[i - 1][-1 * overlapping_points:], axis=0)
                    self.data_x_split_overlap.append(add_x)

                    add_y = np.append(self.data_y_split[i], self.data_y_split[i + 1][0: overlapping_points], axis=0)
                    add_y = np.insert(add_y, 0, self.data_y_split[i - 1][-1 * overlapping_points:], axis=0)
                    self.data_y_split_overlap.append(add_y)

        else:
            # create overlapping arrays
            self.data_x_split_overlap = self.data_x_split.copy()
            self.data_y_split_overlap = self.data_y_split.copy()

    def _check_max_segment_size(self):
        max_size = max([d[-1] - d[0] for d in self.data_x_split])
        epsilon = 0.1

        if self.polynomial_base == POWER_BASIS:
            pass
        elif self.polynomial_base == CHEBYSHEV:
            if max_size > MAX_SEGMENT_SIZE_CHEBYSHEV + epsilon:
                raise Exception(f'Chebyshev Basis: Expected maximum segment x-size to be <= {MAX_SEGMENT_SIZE_CHEBYSHEV}, is {max_size:.2f}')
        elif self.polynomial_base == SHIFTED_CHEBYSHEV:
            if max_size > MAX_SEGMENT_SIZE_SHIFTED_CHEBYSHEV + epsilon:
                raise Exception(f'Shifted Chebyshev Basis: Expected maximum segment x-size to be <= {MAX_SEGMENT_SIZE_SHIFTED_CHEBYSHEV}, is {max_size:.2f}')
        elif self.polynomial_base == LEGENDRE:
            if max_size > MAX_SEGMENT_SIZE_LEGENDRE + epsilon:
                raise Exception(f'Legendre Basis: Expected maximum segment x-size to be <= {MAX_SEGMENT_SIZE_LEGENDRE}, is {max_size:.2f}')
        elif self.polynomial_base == SHIFTED_LEGENDRE:
            if max_size > MAX_SEGMENT_SIZE_SHIFTED_LEGENDRE + epsilon:
                raise Exception(f'Shifted Legendre Basis: Expected maximum segment x-size to be <= {MAX_SEGMENT_SIZE_SHIFTED_LEGENDRE}, is {max_size:.2f}')
        else:
            pass

    def _get_default_degree(self):
        if self.polynomial_base == CHEBYSHEV:
            return DEGREE_CHEBYSHEV
        elif self.polynomial_base == SHIFTED_CHEBYSHEV:
            return DEGREE_SHIFTED_CHEBYSHEV
        else:
            return DEGREE_POWER

    def _linear_equationsystem_x_values(self, x, x0, degree, deriv):
        """Retrieve left hand side vector for polynomial of given degree
        :param x: Single x- value for which the return value should be calculated
        :param x0: Mean of given x-interval = center of polynomial piece
        :param degree: degree of the polynomial
        :param deriv: derivative of the polynomial
        :return: As an example, degree 3, derivative 0 will return [1, x, x**2, x**3]
        """
        if deriv < 0:
            raise Exception("Negative derivative")

        if deriv == 0:
            return [(x - x0) ** i for i in range(degree + 1)]

        result = []

        for i in range(degree + 1):
            if (i - deriv >= 0):
                result.append(math.factorial(i) / math.factorial(i - deriv) * (x - x0) ** (i - deriv))
            else:
                result.append(0)

        return result

    def _evaluate_polynomial_at_x(self, polynum, deriv, x, x0):
        """Return evaluation of a polynomial with given coefficients at location x.
        Method of evaluation depends on parameter self.polynomial_base.
        :param polynum: number of polynomial piece to evaluate. e.g. 0 will use self.coeffs[0] 
        :param deriv: derivative of the polynomial (deriv 0 = y, deriv 1 = y', ...))
        :param x: Single x value or x-value ndarray. Has to be within limits of polynomial segment
        :param x0: center of polynomial piece
        :return: value of a0 + (x-x0)*(a1 + (x-x0)*(a2+(x-x0)*(a3+...+(x-x0)(a_{n-1}+x*a_n))))
        """
        if type(x) != numpy.ndarray and type(x) != numpy.float64 and type(x) != numpy.float32:
            raise Exception(f'Expected input to be of numpy.array or numpy.float but is {type(x)}')

        coeffs = self._derive(deriv, self.coeffs[polynum])

        res = 0.0

        if self.polynomial_base == None:
            res = self._evaluate_horner(coeffs, x, x0)
        elif self.polynomial_base == CHEBYSHEV or self.polynomial_base == SHIFTED_CHEBYSHEV:
            for i, c in enumerate(self.coeffs[polynum]):
                res = self.chebyshev_polynomials[i][deriv](x-x0) * c + res
        elif self.polynomial_base == LEGENDRE:
            raise Exception('Not implemented.')
        else:
            res = self._evaluate_horner(coeffs, x, x0)

        return res

    def _evaluate_horner(self, coeffs, x, x0):
        """ Power basis evaluation for given coefficients.
        """
        res = 0.0

        for c in coeffs[::-1]:
            res = (x - x0) * res + c

        return res

    def _generate_chebyshev_polynomial(self, n, x, x0):
        """ Generate n-th Chebyshev polynomial (Basis vector for orthogonal basis for interval [-1,1]).
        """
        if n == 0:
            #return [1] * len(x)
            return np.poly1d([0, 0, 1])
        elif n == 1:
            return (x - x0)
        else:
            return 2 * (x - x0) * self._generate_chebyshev_polynomial(n - 1, x, x0) - self._generate_chebyshev_polynomial(n - 2, x, x0)

    def _generate_shifted_chebyshev_polynomial(self, n, x, x0):
        """ Generate n-th Chebyshev polynomial shifted by (2(x-x0)-1)
        (Basis vector for orthogonal basis for interval [0,1]).
        """
        if n == 0:
            # return [1] * len(x)
            return np.poly1d([0, 0, 1])
        elif n == 1:
            return 2 * (x - x0) - 1
        else:
            return 2 * (2 * (x - x0) - 1) * self._generate_shifted_chebyshev_polynomial(n - 1, x, x0) - self._generate_shifted_chebyshev_polynomial(n - 2, x, x0)

    def _derive(self, deriv, coeffs):
        """Returns derivative for given coefficient. E.g. deriv=1 is first derivative.
        """
        if deriv == 0:
            pass
        elif deriv < 0 or deriv > self.polydegree:
            raise Exception("Invalid derivative parameter")
        else:
            for i in range(deriv):
                coeffs = self._derive_polynomial(coeffs)
        return coeffs

    def _derive_polynomial(self, coeffs):
        """Returns the derivative of the polynomial given by the cofficients.
        :param coeffs: polynomial coefficients
        :return: If coeffs is [a, b, c, d, …] then return [b, 2c, 3d, …]
        """
        return np.arange(1, coeffs.get_shape()[0]) * coeffs[1:]

    def _pregenerate_chebyshev_polynomials(self):
        """Generate Chebyshev polynomials for later use.
        self.chebyshev_polynomials[1][2] = 1st Chebyshev polynomial, 2nd derivative
        """
        # boundary points initialization required
        if len(self.boundary_points) < 2:
            raise Exception("Internal error: Boundary points not defined")

        # 2D array: self.chebyshev_polynomials[1][2] = 1st Chebyshev polynomial, 2nd derivative
        # polydegree + 1 derivatives for coeffs[0].shape[0] Chebyshev functions
        self.chebyshev_polynomials = [[None] * (self.polydegree + 1) for i in range(self.coeffs[0].shape[0])]

        # Initialize to 0 polynomials
        for j in range(0, self.polydegree + 1):
            for i in range(0, self.polydegree + 1):
                self.chebyshev_polynomials[j][i] = np.poly1d([0, 0, 0])

        p = np.poly1d([0, 1, 0]) # generate monomial x and pass onto generating function to pre-generate non-zero derivatives

        if self.polynomial_base == CHEBYSHEV:
            for i in range(0, len(self.chebyshev_polynomials[0])):
                for j in range(0, self.polydegree + 1):
                    self.chebyshev_polynomials[i][j] = self._generate_chebyshev_polynomial(i, p, 0).deriv(m=j)
        elif self.polynomial_base == SHIFTED_CHEBYSHEV:
                for i in range(0, len(self.chebyshev_polynomials[0])):
                    for j in range(0, self.polydegree + 1):
                        self.chebyshev_polynomials[i][j] = self._generate_shifted_chebyshev_polynomial(i, p, 0).deriv(m=j)
        else:
            raise Exception(f'Invalid polynomial base parameter {self.polynomial_base}')

    def evaluate_pp_at_x(self, x, deriv=0):
        """Evaluate pp derivate deriv at position x
        :param x: Single x value or x-value array. Has to be within limits of self.data_x
        :param deriv: derivative of the pp (deriv 0 = y, deriv 1 = y', ...))
        :return: value of a0 + (x-x0)*(a1 + (x-x0)*(a2+(x-x0)*(a3+...+(x-x0)(a_{n-1}+x*a_n))))
        """
        if deriv < 0:
            raise Exception("Negative derivative")

        if len(self.coeffs) == 0:
            raise Exception("No pp data - Perform fit() first")

        # Evaluate singe x value
        if np.size(x) == 1:
            segment = 0

            if x < self.data_x[0] or x > self.data_x[-1]:
                raise Exception("Provided x-value not in range [" + str(self.data_x[0]) + "," + str(
                    self.data_x[-1]) + "]")

            for i in range(self.polynum):
                if x < self.data_x_split[i][0]:
                    break
                segment = i

            y = self._evaluate_polynomial_at_x(segment, deriv, x, self._polynomial_center(segment))

        # Evaluate x - array
        elif np.size(x) > 1:
            if not self._strictly_increasing(x):
                raise Exception("x - vector not strictly increasing")

            if x[0] < self.data_x_split[0][0] or x[-1] > self.data_x_split[self.polynum - 1][-1]:
                raise Exception(f'Provided x-values in range [{x[0]}, {x[-1]}], expected range [{self.data_x_split[0][0]}, {self.data_x_split[self.polynum - 1][-1]}]')

            begin_segment = 0
            end_segment = 0

            x_split = [[]] * self.polynum
            i = 0
            j = 0

            # Assign given x-values to pp segments
            while i < self.polynum:
                x_split[j] = []
                endpoint = self.boundary_points[i+1]
                while x[end_segment] <= endpoint:
                    end_segment += 1
                    if end_segment >= len(x):
                        break

                if begin_segment != end_segment:
                    x_split[j] = x[begin_segment:end_segment]
                    begin_segment = end_segment

                if end_segment >= len(x) - 1:
                    # if j == self.polynum - 2:  # if last segment only contains one element
                    #     x_split[j + 1] = [x[-1]]
                    break

                j += 1
                i += 1

            y = []

            # Evaluate x segments and merge to result vector
            for segment in range(self.polynum):
                if len(x_split[segment]) != 0:
                    data = self._evaluate_polynomial_at_x(segment, deriv, x_split[segment],
                                                          self._polynomial_center(segment))
                    y.extend(data)
        else:
            raise Exception("No x-value(s) provided")

        return y

    def _polynomial_center(self, polynomial_index):
        """Internal function to retrieve x-center of segment
        """
        if self.shift_polynomial_centers is None:
            return 0
        elif self.shift_polynomial_centers == SHIFT_POLYNOMIAL_CENTER_MEAN: # mean
            return statistics.mean([self.boundary_points[polynomial_index], self.boundary_points[polynomial_index + 1]])
        elif self.shift_polynomial_centers == SHIFT_POLYNOMIAL_CENTER_BOUNDARY: # left boundary point
            return self.boundary_points[polynomial_index]
        else:
            return 0

    def _strictly_increasing(self, L):
        """Check if elements in L are strictly increasing
        :param: L: vector of elements
        :return: true if elements in L are strictly increasing
        """
        return all(x < y for x, y in zip(L, L[1:]))

    def _establish_continuity(self):
        """Align polynomial ends with "self.ck"-continuity.
        C^k-continuity, e.g. 2 = C^2-continuous pp
        Requires minimum polydegree 2k + 1 (needed_coeffs = 2(k+1), needed_degree=needed_coeffs-1)
        """

        if self.polynum < 2:
            return

        if self.polydegree < 2 * self.ck + 1:
            raise Exception("C^" + str(self.ck) + "-continuous pp requires minimum polydegree " + str(2 * self.ck + 1))

        original_basis = None

        if self.polynomial_base != POWER_BASIS:
            original_basis = self.polynomial_base
            self._basis_conversion(target=POWER_BASIS)

        i = 0

        no_equations_per_boundary_point = self.ck + 1
        corr_poly_degree = 2 * self.ck + 1

        # Build from left to right
        while i < self.polynum:

            # Endpoints for polynomial
            x1 = self.data_x_split[i][0]
            y_1 = self.data_y_split[i][0]
            x2 = self.data_x_split[i][-1]
            y_2 = self.data_y_split[i][-1]
            x0 = self._polynomial_center(i)

            # Arrays holding derivatives at point x1 and x2 = equation system right hand side
            y1 = []
            y2 = []

            # Equation system left hand side
            a = []

            # Construct corrective polynomial points
            # Derivatives and equation system for left boundary point
            for k in range(no_equations_per_boundary_point):
                if i == 0:
                    if self.periodic:
                        if self.clamped and k == 0:  # Overtake y-value of derivative 0 if clamped
                            target = self.data_y_split[0][0]
                            diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            target = self._evaluate_polynomial_at_x(self.polynum - 1, k,
                                                                    self.data_x_split[self.polynum - 1][-1],
                                                                    self._polynomial_center(self.polynum - 1)).numpy()  # Overtake boundary derivative value of last segment
                            diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                    elif self.clamped:
                        if (k == 0):
                            target = self.data_y_split[0][0]  # Overtake first y-value
                            diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            diff = 0  # Leave all other derivative values
                    else:
                        diff = 0  # Leave current derivative values
                else:
                    target = self._evaluate_polynomial_at_x(i - 1, k, x1, self._polynomial_center(i - 1)).numpy()  # Overtake boundary derivative value of previous segment
                    diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point

                y1.append(diff)
                a.append(self._linear_equationsystem_x_values(x1, x0, corr_poly_degree, k))

            # Derivatives and equation system for right boundary point
            for k in range(no_equations_per_boundary_point):
                if i == self.polynum - 1:
                    if self.periodic:
                        if self.clamped and k == 0:  # Overtake y-value of derivative 0 if clamped
                            target = self.data_y_split[-1][-1]
                            diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            target = self._evaluate_polynomial_at_x(0, k, self.data_x_split[0][0], self._polynomial_center(0)).numpy()  # Overtake boundary derivative value of first segment
                            diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                    elif self.clamped:
                        if k == 0:
                            target = self.data_y_split[-1][-1]  # Overtake last y-value
                            diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            diff = 0  # Leave all other derivative values
                    else:
                        diff = 0  # Leave current derivative values
                else:
                    target = ((self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i))
                               + self._evaluate_polynomial_at_x(i + 1, k, x2, self._polynomial_center(i + 1))) / 2).numpy()  # Calculate mean of boundary derivative values
                    diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) \
                           - target  # Calculate difference to current value of polynomial at boundary point

                y2.append(diff)
                a.append(self._linear_equationsystem_x_values(x2, x0, corr_poly_degree, k))

            # Solve resulting equation system
            b = y1 + y2
            t = np.linalg.solve(a, b)

            p1 = poly.Polynomial(t)  # corrective polynomial
            p2 = poly.Polynomial(self.coeffs[i].numpy())  # current polynomial
            p3 = poly.polysub(p2, p1)[0]  # difference
            c = p3.coef

            if len(c) < (self.polydegree + 1):
                c = np.pad(c, (0, self.polydegree + 1 - len(c)), constant_values=(0))

            self.coeffs[i] = tf.Variable(c, dtype='float64', trainable=True)  # Update coefficients

            i += 1

        if original_basis != None:
            self._basis_conversion(target=original_basis)

    def _trans_probabilities(self, xs):
        """Takes a sequence of numbers and makes it sum up to one.
        :param: xs: Input x-vector.
        :return: normalized x-vector"""
        xs = np.array(xs)
        return xs / sum(xs)

    def _initialize_pp_data(self):
        """Split input data into POLY_NUM segments and initialize coefficients"""
        if self.rescale_x_data:
            self._rescale_x_data()
        self._split_data(self.overlap_segments)
        self._initialize_polynomial_coefficients()
        # Pregenerate orthogonal basis derivatives
        if self.polynomial_base == CHEBYSHEV or self.polynomial_base == SHIFTED_CHEBYSHEV:
            self._pregenerate_chebyshev_polynomials()

    def _initialize_polynomial_coefficients(self):
        """Initialize polynomial coefficients as tf.Variables"""
        for i in range(self.polynum):
            if self.initialization == 'l2':
                x0 = self._polynomial_center(i)
                if self.polynomial_base == POWER_BASIS:
                    coeff = poly.polyfit(self.data_x_split_overlap[i] - x0, self.data_y_split_overlap[i], self.polydegree)
                    self.coeffs[i] = tf.Variable(coeff, dtype='float64', trainable=True)
                if self.polynomial_base == CHEBYSHEV:
                    coeff = cheby.chebfit(self.data_x_split_overlap[i] - x0, self.data_y_split_overlap[i], self.polydegree)
                    self.coeffs[i] = tf.Variable(coeff, dtype='float64', trainable=True)
            elif self.initialization == 'random':
                coeff = np.random.random(self.polydegree + 1)
                coeff = coeff / sum(coeff)
                self.coeffs[i] = tf.Variable(coeff, dtype='float64', trainable=True)
            else:
                self.coeffs[i] = tf.Variable([0.0 for _ in range(self.polydegree + 1)], dtype='float64', trainable=True)

    def _basis_conversion(self, target=POWER_BASIS):
        if self.polynomial_base == POWER_BASIS and target == CHEBYSHEV:
            for i, _ in enumerate(self.coeffs):
                pol = poly.Polynomial(self.coeffs[i])
                self.coeffs[i] = tf.Variable(pol.convert(kind=cheby.Chebyshev).coef, dtype='float64', trainable=True)
                self.polynomial_base = CHEBYSHEV
                self._initialize_polynomial_center()
                self._pregenerate_chebyshev_polynomials()
        elif self.polynomial_base == CHEBYSHEV and target == POWER_BASIS:
            for i, _ in enumerate(self.coeffs):
                cheb = cheby.Chebyshev(self.coeffs[i])
                self.coeffs[i] = tf.Variable(cheb.convert(kind=poly.Polynomial).coef, dtype='float64', trainable=True)
                self.polynomial_base = POWER_BASIS
                self._initialize_polynomial_center()
        else:
            raise Exception('Not implemented.')

    def _early_stop(self, current_epoch):
        '''https://keras.io/api/callbacks/early_stopping/'''
        if current_epoch == 0:
            self.current_patience = 0
            self.best_loss_index = 1

        if not self.early_stopping:
            return False

        if not self.record_coefficients:
            self.record_coefficients = True
            self.recorded_coeffs = [0.0] * (self.epochs)
            self.recorded_coeffs[0] = copy.deepcopy(self.coeffs)
            return False

        if self.total_loss_values[current_epoch] < self.total_loss_values[self.best_loss_index]:
            self.current_patience = 0
            self.best_loss_index = current_epoch
            return False
        else:
            self.current_patience += 1

        if self.current_patience >= self.patience:
            return True

    def _revert_to_best_epoch(self):
            self.coeffs = copy.deepcopy(self.recorded_coeffs[self.best_loss_index])

            del self.recorded_coeffs[self.best_loss_index+1:]
            del self.total_loss_values[self.best_loss_index+1:]
            del self.e_loss_values[self.best_loss_index+1:]
            del self.D_loss_values[self.best_loss_index+1:]
            del self.d_loss_values[self.best_loss_index+1:]

            self.epochs = self.best_loss_index

    def _append_current_losses(self):
        cost_I = tf.constant(0.0, dtype='float64')
        if self.factor_curvature > 0:
            cost_I = self.integrate_squared_pp_acceleration()
            if self.I_loss_values is not None:
                self.I_loss_values.append(cost_I)
        cost_D, cost_d, regularization_free_cost_D = self.ck_pressure()
        cost_e = self._sum_squared_errors()
        if self.D_loss_values is not None:
            self.D_loss_values.append(regularization_free_cost_D)
        if self.d_loss_values is not None:
            self.d_loss_values.append(cost_d)
        if self.e_loss_values is not None:
            self.e_loss_values.append(cost_e)
        if self.total_loss_values is not None:
            self.total_loss_values.append(tf.add(cost_I, tf.add(regularization_free_cost_D, cost_e)))

    def _optimize_PP(self):
        """Optimize cost function
        self.factor_curvature * self.integrate_squared_pp_acceleration() + self.factor_ck_pressure * self.ck_pressure()
        +  self.factor_approximation_quality * self._sum_squared_errors()
        Factors have to sum up to 1, otherwise the configured learning rate is increased effectively."""
        total_cost_value = 0.0
        cost_I = 0.0
        cost_D = 0.0
        cost_d = []
        reg_grads = [0] * (self.polynum)
        gradients = [0] * (self.polynum)
        self.total_loss_values = [0.0] * (self.epochs)
        self.D_loss_values = [0.0] * (self.epochs)
        self.d_loss_values = [0.0] * (self.epochs)
        self.I_loss_values = [0.0] * (self.epochs)
        self.e_loss_values = [0.0] * (self.epochs)

        if self.record_coefficients:
            self.recorded_coeffs = [0.0] * (self.epochs)
            self.recorded_coeffs[0] = copy.deepcopy(self.coeffs)

        if self.verbose:
            print("TensorFlow: Number of recognized GPUs: ", len(tf.config.list_physical_devices('GPU')))

        # Gradient regularization depending on degree of coefficient
        for j in range(self.polynum):
            if self.gradient_regularization:
                reg_grads[j] = [1.0 / (1 + i) for i in range(self.polydegree + 1)]
            else:
                reg_grads[j] = np.ones(self.polydegree + 1)
            # Make gradient regularization coefficients a probability distribution.
            # This makes the sum of gradients independent of degree.
            reg_grads[j] = self._trans_probabilities(reg_grads[j])

        # Fitting optimization routine
        for epoch in tqdm.trange(self.epochs):
            # Gradient Tape
            with tf.GradientTape(persistent=True) as tape:
                cost_I = tf.constant(0.0, dtype='float64')
                cost_D = tf.constant(0.0, dtype='float64')
                cost_e = tf.constant(0.0, dtype='float64')
                if self.factor_curvature > 0:
                    cost_I = self.integrate_squared_pp_acceleration()
                cost_D, cost_d, regularization_free_cost_D = self.ck_pressure()
                cost_e = self._sum_squared_errors()

                # Save unweighted, non-regularized loss values for loss plots
                self.D_loss_values[epoch] = regularization_free_cost_D
                self.d_loss_values[epoch] = cost_d
                self.I_loss_values[epoch] = cost_I
                self.e_loss_values[epoch] = cost_e
                self.total_loss_values[epoch] = tf.add(cost_I, tf.add(regularization_free_cost_D, cost_e))

                # weighted losses for gradient calculation
                cost_I = tf.multiply(cost_I, self.factor_curvature)
                cost_D = tf.multiply(cost_D, self.factor_ck_pressure)
                cost_e = tf.multiply(cost_e, self.factor_approximation_quality)
                total_cost_value = tf.add(cost_I, tf.add(cost_D, cost_e))

            if self._early_stop(epoch):
                self._revert_to_best_epoch()
                print(f'Early stopping: No improvement for {self.patience} epochs. Reverting to epoch {self.best_loss_index}. Exiting.')
                return

            # Calculate Gradients
            gradients = tape.gradient(total_cost_value, self.coeffs)

            # Apply regularization
            for i in range(self.polynum):
                gradients[i] = gradients[i] * reg_grads[i]

            # Apply Gradients
            self.optimizer.apply_gradients(zip(gradients, self.coeffs))

            if self.record_coefficients:
                self.recorded_coeffs[epoch] = copy.deepcopy(self.coeffs)

            if self.verbose and epoch % 10 == 0:
                # print("Gradients epoch ", epoch, ": ", gradients, "\n")
                print("epoch=%d, loss=%4g\r" % (epoch, total_cost_value), end="")

    def fit(self, data_x, data_y, initialization='zero', overlap_segments=0,
            rescale_x_data=True, ck_regularization=True, early_stopping=True, patience=100, **kwargs):
        """Fits pp to data_x / data_y using specified Hyperparameters and returns cost value
        :param: Data x-values. Have to be increasing.
        :param: Data y-values. 
        :param: overlap_segments: Only relevant if initialization == l2.
                                  Percentage of adjacent points of segment n to be included in
                                  adjacent segments n-1 and n+1 for initial l2 fit
                                  in order to better align the curve's derivatives at transition points.
        :param: initialization: zero: Initialize all segments to zero pp.
                                random: initialize coefficients to a random convex combination.
                                l2: Perform least squares fitting initialization for each segment.
        :param: rescale_x_data: Experiments have shown that each basis has an ideal segment size with
        favourable optimization results. If this parameter is set to True, data_x will be rescaled
        to enable these properties.
        :param: ck_regularization: Regularizes the Ck-loss of each derivative to minimize oscillating behaviour
        :param: early_stopping: Stop training when loss has stopped improving for...
        :param: patience: ...number of epochs.
        :param enforce_continuity: Specifies whether continuity should be strictly established after optimization
        :param periodic: "True" --> Align derivatives at position xn with derivatives at position x0. If clamped = true, derivative 0 will not be aligned
        :param clamped: "True" --> Clamp pp result (derivative 0) to first and last data point of input space
        **kwargs: Parameters for optimization. See method optimization() for details
        :return: Loss value at the end of optimization
        """
        self.data_x = data_x
        self.data_y = data_y
        self.overlap_segments = overlap_segments
        self.initialization = initialization
        self.rescale_x_data = rescale_x_data
        self.early_stopping = early_stopping
        self.patience = patience

        if overlap_segments < 0 or overlap_segments > 1:
            raise Exception("Invalid overlap_segments parameter - Must be 0 <= overlap_segments <= 1")

        if len(self.data_x) != len(self.data_y):
            raise Exception(f'x and y data must have the same length (x:{len(self.data_x)}, y:{len(self.data_y)})')

        if 'optimizer' in kwargs:
            optimizer = kwargs['optimizer']
        else:
            optimizer = self.optimizer
        if 'n_epochs' in kwargs:
            n_epochs = kwargs['n_epochs']
        else:
            n_epochs = DEFAULT_EPOCHS
        if 'learning_rate' in kwargs:
            learning_rate = kwargs['learning_rate']
        else:
            learning_rate = DEFAULT_LEARNING_RATE
        if 'factor_ck_pressure' in kwargs:
            factor_ck_pressure = kwargs['factor_ck_pressure']
        else:
            factor_ck_pressure = DEFAULT_CK_PRESSURE_FACTOR
        if 'factor_approximation_quality' in kwargs:
            factor_approximation_quality = kwargs['factor_approximation_quality']
        else:
            factor_approximation_quality = DEFAULT_APPROXIMATION_QUALITY_FACTOR
        if 'factor_curvature' in kwargs:
            factor_curvature = kwargs['factor_curvature']
        else:
            factor_curvature = DEFAULT_CURVATURE_FACTOR
        if 'gradient_regularization' in kwargs:
            gradient_regularization = kwargs['gradient_regularization']
        else:
            gradient_regularization = False
        if 'record_coefficients' in kwargs:
            self.record_coefficients = kwargs['record_coefficients']
        else:
            self.record_coefficients = False
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])  # seed for random coefficient initialization
        if 'enforce_continuity' in kwargs:
            self.enforce_continuity = kwargs['enforce_continuity']
        else:
            self.enforce_continuity = False
        if 'clamped' in kwargs:
            self.clamped = kwargs['clamped']
        else:
            self.clamped = False
        if 'periodic' in kwargs:
            self.periodic = kwargs['periodic']
        else:
            self.periodic = False

        self._initialize_pp_data()
        self.performedfit = True

        if n_epochs > 0:
            self.optimize(optimizer=optimizer, n_epochs=n_epochs, learning_rate=learning_rate,
                          factor_ck_pressure=factor_ck_pressure,
                          factor_approximation_quality=factor_approximation_quality, factor_curvature=factor_curvature,
                          gradient_regularization=gradient_regularization, ck_regularization=ck_regularization)

            return self.total_loss_values[-1]
        else:
            return None

    def optimize(self, optimizer, n_epochs=DEFAULT_EPOCHS, **kwargs):
        """Optimizes pp from previous fit(data_x, data_y) using TensorFlow Gradient Tapes
        Cost function:
        self.factor_curvature * self.integrate_squared_pp_acceleration() + self.factor_ck_pressure * self.ck_pressure()
        +  self.factor_approximation_quality * self._sum_squared_errors()
        Factors have to sum up to 1, otherwise the configured learning rate is increased effectively.
        :param: optimizer: Optimizer to use (keras.optimizers object)
        :param: n_epochs: Number of optimization cycles.
        :param: factor_ck_pressure: regularization factor for Ck-pressure
        :param: factor_approximation_quality: regularization factor for approximation quality
        :param: factor_curvature: regularization factor for curvature penalization
        :param: gradient_regularization: True --> Apply Gradient regularization depending on degree of coefficient
        :return: Loss value at the end of optimization        
        """
        self.optimizer = optimizer
        self.epochs = n_epochs

        if 'factor_ck_pressure' in kwargs:
            self.factor_ck_pressure = kwargs['factor_ck_pressure']
        if 'factor_approximation_quality' in kwargs:
            self.factor_approximation_quality = kwargs['factor_approximation_quality']
        if 'factor_curvature' in kwargs:
            self.factor_curvature = kwargs['factor_curvature']
        if 'gradient_regularization' in kwargs:
            self.gradient_regularization = kwargs['gradient_regularization']
        else:
            self.gradient_regularization = False
        if 'ck_regularization' in kwargs:
            self.ck_regularization = kwargs['ck_regularization']
        else:
            self.ck_regularization = True

        if not self.performedfit:
            raise Exception("No pp data - Perform fit() first")

        if not optimizer:
            raise Exception("No optimizer specified")

        start_time = time.time()
        self._optimize_PP()

        if self.verbose:
            print("Fitting took %s seconds" % (time.time() - start_time))

        if self.enforce_continuity:
            self._establish_continuity()
            self._append_current_losses()


        return self.total_loss_values[-1]
