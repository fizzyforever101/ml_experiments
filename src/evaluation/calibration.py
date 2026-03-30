from sklearn.calibration import calibration_curve

def get_calibration(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    return prob_true, prob_pred