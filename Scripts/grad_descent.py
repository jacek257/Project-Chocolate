from math import e
import numpy as np

class gd:
    def __grad_constant(M, alpha, phi, psi, t, y):
            return (2.0*(M/(1.0+e**(-1.0*alpha*(t-phi)))+psi-y))

    def __grad_M(M, alpha, phi, psi, time, y_series):
        buffer = 0.0
        for i, t in enumerate(time):
            y = y_series[i]
            buffer += (grad_constant(M, alpha, phi, psi, t, y)/(1.0+e**(alpha*(phi-t))))/len(time)
        return buffer

    def __grad_alpha(M, alpha, phi, psi, time, y_series):
        buffer = 0.0
        for i, t in enumerate(time):
            y = y_series[i]
            buffer += (grad_constant(M, alpha, phi, psi, t, y)*M*((t-phi)*e**(alpha*(phi-t)))
                        /((1+e**(alpha*(phi-t)))**2.0))/len(time)
        return buffer

    def __grad_phi(M, alpha, phi, psi, time, y_series):
        buffer = 0.0
        for i, t in enumerate(time):
            y = y_series[i]
            buffer += (grad_constant(M, alpha, phi, psi, t, y)*-1.0*alpha*M*e**(alpha*(phi-t))
                 /(1+e**(alpha*(phi-t)))**2.0)/len(time)
        return buffer

    def __grad_psi(M, alpha, phi, psi, time, y_series):
        buffer = 0.0
        for i, t in enumerate(time):
            y = y_series[i]
            buffer += grad_constant(M, alpha, phi, psi, t, y)/len(time)
        return buffer

    def __get_cost(M, alpha, phi, psi, time, y_series):
        buffer = 0.0
        for i, t in enumerate(time):
             buffer += (M/(1.0+e**(alpha*(phi-t)))+psi - y_series[i])
        return buffer
    def gradient_descent(init_4D_point, time, y_series, descent_speed = .1, epochs = 10000):
        """
        Simple gradient descent
        init_4D_point (tuple) = (M, alpha, phi, psi)
        y (numpy array) = observed data
        t (numpy array) = sampling points for data
        descent speed (float) = step size
        epochs (int) = # of steps
        """
        curr_M = init_4D_point[0]
        curr_alpha = init_4D_point[1]
        curr_phi = init_4D_point[2]
        curr_psi = init_4D_point[3]

        prev_M = 0.0
        prev_alpha = 0.0
        prev_phi = 0.0
        prev_psi = 0.0

    #     minima_reached = False
        minima = ()
        iteration = 0

        for i in range(epochs):
            prev_M = float(curr_M)
            prev_alpha = float(curr_alpha)
            prev_phi = float(curr_phi)
            prev_psi = float(curr_psi)

            curr_M -= (__grad_M(curr_M, curr_alpha, curr_phi, curr_psi, time, y_series) * descent_speed)
            curr_alpha -= (__grad_alpha(curr_M, curr_alpha, curr_phi, curr_psi, time, y_series) * descent_speed)
            curr_phi -= (__grad_phi(curr_M, curr_alpha, curr_phi, curr_psi, time, y_series) * descent_speed)
            curr_psi -= (__grad_psi(curr_M, curr_alpha, curr_phi, curr_psi, time, y_series) * descent_speed)

            iteration += 1

            if(__get_cost(prev_M, prev_alpha, prev_phi, prev_psi, time, y_series)<
               __get_cost(curr_M, curr_alpha, curr_phi, curr_psi, time, y_series)):
                minima_reached = True
                minima = (prev_M, prev_alpha, prev_phi, prev_psi)


        return minima

def generate_logistic_curve(constant_tuple, time):
    out_y = []
    for i, data in enumerate(time):
        out_y.append(constant_tuple[0]/(1+e**(-1*constant_tuple[1]*(data - constant_tuple[2])))+constant_tuple[3])
    return out_y
