from math import e
import numpy as np
import scipy.signal as sg

class llc:
    def __get_cost(mag, series):
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
        return buffer

    def descend_onto_line(guess_mag, series, descent_speed=.1, lifespan=10000, epochs=3):
        if descent_speed >= 1:
            print("Bad descent speed")
            return np.nan
        curr_mag = guess_mag
        for precision in range(epochs):
            for i in range(lifespan):
                curr_mag = curr_mag - (descent_speed**precision)*__get_grad(curr_mag, series)

        return curr_mag
