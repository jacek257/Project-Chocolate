from math import e, log
from tqdm import tqdm
import numpy as np
import scipy.signal as sg
import time

class llc:
    def get_cost(mag, series):
        buffer = 0.0
        normalizer = len(series)
        for point in series:
            buffer += (mag - point)**2/normalizer
        return buffer
    def __get_grad(mag, series):
        buffer = 0.0
        normalizer = len(series)
        for point in series:
            buffer += (mag - point)*2/normalizer
            # buffer += 2/(mag-point)
        return buffer

    def descend_onto_line(guess_mag, series, descent_speed=.1, lifespan=100, epochs=3):
        if descent_speed >= 1:
            print("Bad descent speed")
            return np.nan
        curr_mag = float(guess_mag)
        for precision in range(epochs):
            print('EPOCH :   ', precision)
            time.sleep(.25)
            for i in tqdm(range(lifespan)):
                curr_mag = curr_mag - (descent_speed**precision)*llc.__get_grad(curr_mag, series)

        return curr_mag
