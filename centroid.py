import numpy as np
import scipy.signal as sg
import scipy.interpolate as interp

#acknowledgements:
#https://ws680.nist.gov/publication/get_pdf.cfm?pub_id=901379

def get_centroid(t_series, data_series, window_coeff, poly_order):
    """
    centroid is a signal weighted average of time. We are essentially calculating the temporal middle of the signal

    t_series (iterable) = time points of data_series
    data_series (iterable) = the signal intensity series
    window_coeff (int) = filter window size
    poly_order (int) = the order of savgol fit polynomial
    """
    #set buffer to 0
    buffer = 0.0
    #pack t_series and d_series together and calculate square weighted sum
    for t, d in zip(t_series, sg.savgol_filter(data_series, window_coeff, poly_order)):
        buffer += (d**2) * t
    #return square weighted average
    return buffer/(np.sum(data_series**2))

def align_centroid(base_time, base_sig, time1, sig1, time2, sig2): #TODO: arbitrary number of centroids
    """
    aligns several (3) time series based on their relative temporal centroids

    base_time = BOLD
    rest = self explanatory

    *time* (iterable) = time points / sampling points
    *sig* (iterable) = sampling intensity at each time/sampling points

    [*]sig[*] must have equal length to [*]time[*]

    returns a dictionary containing  {base_time, base_sig, sig1, sig2} properly aligned and interpolated
    """
    #get all centroids
    base_centroid = get_centroid(base_time, base_sig)
    sig1_centroid = get_centroid(time1, sig1)
    sig2_centroid = get_centroid(time2, sig2)

    #shift times and construct interpolators
    sig1_resampler = interp.interp1d(time1 - sig1_centroid + base_centroid, sig1)
    sig2_resampler = interp.interp1d(time2 - sig2_centroid + base_centroid, sig2)

    #interpolate signals
    sig1_exact_aligned = sig1_resampler(base_time)
    sig2_exact_aligned = sig2_resampler(base_time)

    return {'base_time' : base_time, 'base_sig' : base_sig, 'sig1' : sig1_exact_aligned, 'sig2' : sig2_exact_aligned}

def grad_constant(c1,c2,c3,s1n, s2n, bn):
    """
    part of mathematical gradient which is present in all components

    c1 (float) = constant 1
    c2 (float) = constant 2
    c3 (float) = constant 3

    s1n (float) = signal1 at time n
    s2n (float) = signal2 at time n
    bn (float) = base signal at time n
    """
    return(2*(c1*s1n + c2*s2n + c3 - bn))
def grad_C1(C1, C2, C3, S1, S2, B):
    """
    component 1 of gradient

    C1 (float) = constant 1
    C2 (float) = constant 2
    C3 (float) = constant 3

    S1 (iterable) = signal 1
    S2 (iterable) = signal 2

    B (iterable) = base signal
    """
    buffer = 0.0
    for s1n, s2n, bn in zip(S1, S2, B):
        buffer += grad_constant(C1, C2, C3, s1n, s2n, bn) * s1n/len(S1)
    return buffer

def grad_C2(C1, C2, C3, S1, S2, B):
    """
    component 2 of gradient

    C1 (float) = constant 1
    C2 (float) = constant 2
    C3 (float) = constant 3

    S1 (iterable) = signal 1
    S2 (iterable) = signal 2

    B (iterable) = base signal
    """
    buffer = 0.0
    for s1n, s2n, bn in zip(S1, S2, B):
        buffer += grad_constant(C1, C2, C3, s1n, s2n, bn) * s2n/len(S1)
    return buffer

def grad_C3(C1, C2, C3, S1, S2, B):
    """
    component 3 of gradient

    C1 (float) = constant 1
    C2 (float) = constant 2
    C3 (float) = constant 3

    S1 (iterable) = signal 1
    S2 (iterable) = signal 2

    B (iterable) = base signal
    """
    buffer = 0.0
    for s1n, s2n, bn in zip(S1, S2, B):
        buffer += grad_constant(C1, C2, C3, s1n, s2n, bn)/len(S1)
    return buffer

def linear_optimize(init_tuple, S1, S2, B, descent_speed=.1, lifespan = 100, epochs = 3):
    """
    linear coefficient optimizer by gradient descent
    returns (C1, C2, C3)
    where: C1 * S1 + C2 * S2 + C3 = B

    init_tuple (c1, c2, c3) = initial coefficeint guess
    S1 (iterable) = signal 1
    S2 (iterable) = signal 2
    B (iterable) = base signal
    descent_speed (float) = descending step size
    lifespan = number of steps per epoch
    epochs = number of step sizes
    """
    curr_C1 = init_tuple[0]
    curr_C2 = init_tuple[1]
    curr_C3 = init_tuple[2]


    for precision in range(epochs):
        print('EPOCH:     ', precision)
        time.sleep(.25)
        for i in tqdm(range(lifespan)):
            curr_C1 = curr_C1 - (descent_speed**(precision+1)* grad_C1(curr_C1, curr_C2, curr_C3, S1, S2, B))
            curr_C2 = curr_C2 - (descent_speed**precision)* grad_C1(curr_C1, curr_C2, curr_C3, S1, S2, B)
            curr_C3 = curr_C3 - (descent_speed**precision)* grad_C1(curr_C1, curr_C2, curr_C3, S1, S2, B)
    return_tuple = (curr_C1, curr_C2, curr_C3)
    return return_tuple
