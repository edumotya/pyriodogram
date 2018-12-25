"""
Extract ndft features using Lightweight non-uniform Fast Fourier Transform in Python [1].
It computes both the spectrum magnitude and unwrapped phase and also the Lomb-Scargle
normalised periodogram. It is a python reimplementation of the original c code [2], avoiding
issues related to package dependencies though.

[1] https://github.com/jakevdp/nfft
[2] http://corot-be.obspm.fr/code.php
"""

import numpy as np
import nfft # pip install nfft
import matplotlib.pyplot as plt

def _reduce_times(t, oversampling, tolerance):
    """
    Reduces the times to the interval [-1/2, 1/2).
    Args:
        t: the times
        oversampling: the oversampling factor.
    Returns:
        the reduced times.
    """
    tolerance = 0
    tmax = t[-1]; tmin = t[0]
    trange = oversampling * (tmax - tmin)
    k = 0.5 - tolerance
    return 2 * k * (t - tmin) / trange - k

def _dirty_spectrum(t, y, N, tolerance):
    """
    Computates the positive frequency part of the (unnormalised)
    Fourier transform of a times -series (t, y).
    Args:
        t: the times reduced to [1/2 , 1/2)
        y: the measurements (None for computing the fft of the window)
        N: the number of positive frequencies
        oversampling: the oversampling factor.
    Returns:
        coeff: the Fourier coefficients
    """
    # Catch window spectrum
    if y is None:
        y = np.ones_like(t, dtype=np.float)
    # Compute nfft. Note: the oversampling parameter has nothing to do with the internal
    # oversampling parameter of the ndft named sigma.
    dft = nfft.nfft_adjoint(t, y, N=2*N, use_fft=False, truncated=True, tol=tolerance)
    # Keep only positive part
    dft_pos = np.zeros(N+1, dtype=np.complex)
    dft_pos[:N] = dft[N:]
    # NFFT plan produces Fourier components for frequencies in an open interval [-m, m)
    # but here we're working with the closed interval.
    dft_pos[-1] = np.conj(dft_pos[0])
    return dft_pos

def _false_alarm_prob(Pn, n, N, oversampling):
    """
    Computes the false alarm probability of the periodogram Pn. Formally,
    the probability that a peak of a given power appears in the periodogram
    when the signal is white Gaussian noise.
    A. Schwarzenberg-Czerny (MNRAS 1998, 301, 831), but without explicitely
    using modified Bessel functions.
    Args:
        Pn: periodogram.
        n: number of points of the original time domain signal.
        N: number of frequencies.
        oversampling: the oversampling factor >= 1.
    """
    effm = 2. * N / oversampling
    Ix = 1. - np.power(1 - 2 * Pn / n, 0.5 * (n - 3))
    # Handle negative number raise to floating power (giving nan) and
    # also floating type overflow
    return np.clip(np.nan_to_num(proba, 1.), 0., 1.)

def _true_alarm_prob(Pn, n, N, oversampling):
    """
    Computes the TRUE alarm probability of the periodogram Pn. Actually, it is
    1 - false alarm prob, a more useful feature.
    Args:
        Pn: periodogram.
        n: number of points of the original time domain signal.
        N: number of frequencies.
        oversampling: the oversampling factor >= 1.
    """
    effm = 2. * N / oversampling
    Ix = 1. - np.power(1 - 2 * Pn / n, 0.5 * (n - 3))
    proba = np.power(Ix, effm)
    # Handle negative number raise to floating power (giving nan) and
    # also floating type overflow
    return np.clip(np.nan_to_num(proba, 0.), 0., 1.)
    
def extract(t, y, hifac=1, oversampling=5, tolerance=1e-8):
    """
    Computes ndft features. Either the magnitude and phase of the positive
    side of the spectrum or the the Lomb-Scargle normalised periodogram of
    the input time series.
    Args:
        t: the observation times.
        y: non uniformly sampled observations.
        oversampling: the oversampling factor >= 1.
        hifac: the highest frequency in units of the Nyquist > 0.
        frequency that would correspond to a uniform sampling.
    Returns:
        freqs: frequencies where the spectrum/periodogram is computed
        mag:  magnitude of the Fourier transform
        phase: unwrapped phase of the Fourier transform
        Pn: LombScargle periodogram
    """
    assert(hifac>0)
    assert(oversampling>=1)
    if len(t) != len(y):
        raise AssertionError('Times (t) and observations (y) must have the same number of points')
    # Center observations
    y -= np.mean(y)
    # Determines the Nyquist frequency for a uniform sampling and the frequency resolution.
    # fc = len(y) / (2 * T)
    # fmax = hifac * fc
    # N = fmax / df
    df = 1.0 / (oversampling * (t[-1] - t[0]))
    # Defermines the highest frequency, fmax, and the corresponding index, N, in
    # the positive part of the spectrum.
    N = int(np.floor(0.5 * len(y) * oversampling * hifac))
    # Move times scale
    # Note: after computing the Nyquist freq and freq resolution, but it actually does not matter.
    t = _reduce_times(t, oversampling, tolerance)

    #  Unnormalised Fourier transform of the observations
    Y = _dirty_spectrum(t, y, N, tolerance)

    # Output frequencies
    freqs = df*np.arange(1, N+1)

    # Spectrums -magnitude and phase-
    # Magnitude
    mag = np.abs(Y[1:])
    # Unwrapped phase
    phase = np.unwrap(2*np.angle(Y[1:], deg=False))/2

    # Periodogram
    # Unbiased variance of the observations
    var = np.var(y, ddof=1)
    # Unnormalised Fourier transform of the window.
    W = _dirty_spectrum(t, None, 2*N, tolerance)
    # Compute the LombScargle periodogram
    z1 = Y[1:]
    z2 = W[np.arange(2, 2*N+1, 2)] # 2*N+1 interval including last one
    hypo = np.abs(z2)
    hc2wt = 0.5 * np.imag(z2) / hypo
    hs2wt = 0.5 * np.real(z2) / hypo
    cwt = np.sqrt(0.5 + hc2wt)
    swt = np.sign(np.sqrt(0.5 - hc2wt), hs2wt)
    den = 0.5 * len(y) + hc2wt * np.real(z2) + hs2wt * np.imag(z2)
    cterm = np.square(cwt * np.real(z1) + swt * np.imag(z1)) / den
    sterm = np.square(cwt * np.imag(z1) - swt * np.real(z1)) / (len(y) - den)
    # Actual periodogram
    Pn = (cterm + sterm) / (2 * var)
    # True alarm probability
    proba = _true_alarm_prob(Pn, len(y), N, oversampling)

    return freqs, mag, phase, Pn, proba

def plot(t, y, oversampling=5):
    """
    Plot ndft features extracted with ndft_features.
    """
    freqs, mag, phase, Pn, proba = extract(t, y, hifac=1, oversampling=oversampling)
    fig, ax = plt.subplots(6, 1, figsize=(14, 12))
    ax[0].scatter(t, y)
    ax[0].set_title('Flux')
    ax[1].plot(freqs, mag)
    ax[1].set_title('Spectrum magnitude')
    ax[2].plot(freqs, phase)
    ax[2].set_title('Spectrum phase')
    ax[3].plot(1/freqs, Pn)
    ax[3].set_title('LombScargle periodogram')
    ax[4].plot(1/freqs, proba)
    ax[4].set_title('Periodogram true alarm probability')
    # Estimated phase
    h = 1/freqs[np.argmax(Pn)]
    phase_hat = (t/h)%1
    ax[5].scatter(phase_hat, y)
    ax[5].set_title('Phase from estimated period, where h=%.4f'%h)
    fig.tight_layout()
    return freqs, mag, phase, Pn, proba
