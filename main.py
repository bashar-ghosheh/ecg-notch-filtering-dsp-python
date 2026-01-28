import wfdb
import numpy as np
from scipy import fftpack
import scipy.signal as scipy_signal
import matplotlib.pyplot as plt


def load_mit_bih_data(record_identifier):
    record_data = wfdb.rdrecord(record_identifier, pn_dir='mitdb')
    ecg_signal = record_data.p_signal[:, 0]
    sampling_rate = record_data.fs

    annotation_data = wfdb.rdann(record_identifier, 'atr', pn_dir='mitdb')
    ground_truth_peaks = annotation_data.sample

    return ecg_signal, sampling_rate, ground_truth_peaks


def apply_fixed_notch(input_signal, sampling_rate, target_frequency=60.0):
    filter_numerator, filter_denominator = scipy_signal.iirnotch(target_frequency, 30.0, sampling_rate)
    return scipy_signal.filtfilt(filter_numerator, filter_denominator, input_signal)


def apply_adaptive_fft_filter(input_signal, sampling_rate):
    # 1. Compute FFT of the entire signal and Look for peak near 60Hz
    freqs = fftpack.fftfreq(len(input_signal), d=1.0 / sampling_rate)
    spectrum = np.abs(fftpack.fft(input_signal))

    mask = (freqs > 55) & (freqs < 65)
    freqs_in_range = freqs[mask]
    spectrum_in_range = spectrum[mask]
    peak_index = np.argmax(spectrum_in_range)
    dominant_freq = freqs_in_range[peak_index]

    # 2. Design Notch Filter at that EXACT frequency
    b, a = scipy_signal.iirnotch(dominant_freq, 30.0, sampling_rate)

    # 3. Apply Filter
    return scipy_signal.filtfilt(b, a, input_signal)


def pan_tompkins_detector(filtered_input_signal, sampling_rate):
    nyquist_frequency = 0.5 * sampling_rate
    bandpass_cutoff_frequencies = [5 / nyquist_frequency, 12 / nyquist_frequency]

    # Phase 1: Bandpass Filter
    filter_numerator, filter_denominator = scipy_signal.butter(1, bandpass_cutoff_frequencies, btype='band')
    bandpass_filtered_ecg = scipy_signal.filtfilt(filter_numerator, filter_denominator, filtered_input_signal)

    # Phase 2: Derivative
    differentiated_ecg = np.diff(bandpass_filtered_ecg, prepend=0)

    # Phase 3: Squaring
    squared_ecg = differentiated_ecg ** 2

    # Phase 4: Integration
    integration_window_length = int(0.150 * sampling_rate)
    integrated_ecg = np.convolve(squared_ecg, np.ones(integration_window_length) / integration_window_length, mode='same')

    return integrated_ecg, bandpass_filtered_ecg, differentiated_ecg, squared_ecg

def plot_time_and_frequency_domains(input_signal, sampling_rate, plot_title, plot_color):
    samples_to_plot = int(3 * sampling_rate)
    half_samples = int(samples_to_plot / 2)

    # Time axis
    time_axis = np.arange(samples_to_plot) / sampling_rate

    # Frequency axis and Magnitude using fftpack
    frequency_values = fftpack.fftfreq(samples_to_plot, d=1.0 / sampling_rate)[:half_samples]
    magnitude_spectrum = np.abs(fftpack.fft(input_signal[:samples_to_plot])[:half_samples]) * (2.0 / samples_to_plot)

    # --- Time Domain Plot ---
    plt.figure()
    plt.plot(time_axis, input_signal[:samples_to_plot], color=plot_color)
    plt.title(plot_title + " - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.show()

    # --- Frequency Domain Plot ---
    plt.figure()
    plt.plot(frequency_values, magnitude_spectrum, color=plot_color)
    plt.title(plot_title + " - Frequency Domain (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.axis([0, 100, 0, max(magnitude_spectrum) * 1.1])
    plt.show()


def plot_pan_tompkins_stages(bandpass_data, derivative_data, squared_data, integrated_data, sampling_rate):
    samples_to_plot = int(3 * sampling_rate)
    time_axis = np.arange(samples_to_plot) / sampling_rate

    # Phase 1
    plt.figure()
    plt.plot(time_axis, bandpass_data[:samples_to_plot], color='c')
    plt.title("Phase 1: Bandpass (5-12 Hz)")
    plt.xlabel("Time (s)")
    plt.show()

    # Phase 2
    plt.figure()
    plt.plot(time_axis, derivative_data[:samples_to_plot], color='y')
    plt.title("Phase 2: Derivative (Slope Detection)")
    plt.xlabel("Time (s)")
    plt.show()

    # Phase 3
    plt.figure()
    plt.plot(time_axis, squared_data[:samples_to_plot], color='m')
    plt.title("Phase 3: Squaring (Nonlinear Amplification)")
    plt.xlabel("Time (s)")
    plt.show()

    # Phase 4
    plt.figure()
    plt.plot(time_axis, integrated_data[:samples_to_plot], color='k')
    plt.title("Phase 4: Moving Window Integration")
    plt.xlabel("Time (s)")
    plt.show()


def calculate_metrics(raw_signal, filtered_signal, detected_peaks, ground_truth_peaks, fs, noise_freq=60.0):
    # 1. Residual Noise at Target Frequency (60Hz)
    # We use FFT to measure the magnitude of the signal remaining at 60Hz
    n_samples = len(filtered_signal)
    freqs = fftpack.fftfreq(n_samples, d=1.0 / fs)
    fft_spectrum = np.abs(fftpack.fft(filtered_signal))

    # Find the index in the frequency array closest to 60Hz
    target_idx = np.argmin(np.abs(freqs - noise_freq))
    residual_noise_magnitude = fft_spectrum[target_idx]

    # 2. Signal-to-Noise Ratio (SNR)
    noise_component = raw_signal - filtered_signal

    signal_power = np.sum(filtered_signal ** 2)
    noise_power = np.sum(noise_component ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    # 3. QRS Detection Metrics (TP, FP, FN)
    tolerance_samples = int(0.15 * fs)

    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives (Missed Beats)

    # Make copies to track which peaks have been matched
    unmatched_detected = list(detected_peaks)

    for gt_peak in ground_truth_peaks:
        # Find all detected peaks within tolerance of this ground truth peak
        candidates = [d for d in unmatched_detected if abs(d - gt_peak) <= tolerance_samples]

        if candidates:
            # If we found a match, it's a True Positive
            tp += 1
            # Remove the closest match from the available list (one-to-one matching)
            closest_match = min(candidates, key=lambda x: abs(x - gt_peak))
            unmatched_detected.remove(closest_match)
        else:
            # If no match found within tolerance, it's a Missed Beat (FN)
            fn += 1

    # Any detected peaks remaining that weren't matched to a GT peak are False Positives
    fp = len(unmatched_detected)

    accuracy = tp / (tp + fp + fn) * 100

    return residual_noise_magnitude, snr, accuracy, fn, fp

TARGET_RECORD_ID = '105'
raw_ecg_signal, sampling_rate_hz, expert_ground_truth = load_mit_bih_data(TARGET_RECORD_ID)

# 1. Apply Filters
fixed_notch_filtered_signal = apply_fixed_notch(raw_ecg_signal, sampling_rate_hz)
adaptive_filtered_signal = apply_adaptive_fft_filter(raw_ecg_signal, sampling_rate_hz)

# 2. Run Pan-Tompkins on Filtered Signals
integrated_adaptive, bp_adaptive, deriv_adaptive, sq_adaptive = pan_tompkins_detector(adaptive_filtered_signal, sampling_rate_hz)
integrated_fixed, bp_fixed, deriv_fixed, sq_fixed = pan_tompkins_detector(fixed_notch_filtered_signal, sampling_rate_hz)

# 3. Visualization
plot_time_and_frequency_domains(raw_ecg_signal, sampling_rate_hz, "Original Raw ECG", 'r')
plot_time_and_frequency_domains(fixed_notch_filtered_signal, sampling_rate_hz, "Fixed 60Hz Notch Filtered", 'b')
plot_time_and_frequency_domains(adaptive_filtered_signal, sampling_rate_hz, "Adaptive FFT Filtered", 'g')
plot_pan_tompkins_stages(bp_adaptive, deriv_adaptive, sq_adaptive, integrated_adaptive, sampling_rate_hz)
plot_pan_tompkins_stages(bp_fixed, deriv_fixed, sq_fixed, integrated_fixed, sampling_rate_hz)

# 4. Calculate and print results
peak_distance_threshold = int(0.2 * sampling_rate_hz)
height_threshold_fixed = np.mean(integrated_fixed) * 1.5
height_threshold_adaptive = np.mean(integrated_adaptive) * 1.5

peaks_indices_fixed, _ = scipy_signal.find_peaks(integrated_fixed, height=height_threshold_fixed, distance=peak_distance_threshold)
peaks_indices_adaptive, _ = scipy_signal.find_peaks(integrated_adaptive, height=height_threshold_adaptive, distance=peak_distance_threshold)

res_fixed, snr_fixed, acc_fixed, fn_fixed, fp_fixed = calculate_metrics(
    raw_ecg_signal,
    fixed_notch_filtered_signal,
    peaks_indices_fixed,
    expert_ground_truth,
    sampling_rate_hz,
    noise_freq=60.0
)

res_adaptive, snr_adaptive, acc_adaptive, fn_adaptive, fp_adaptive = calculate_metrics(
    raw_ecg_signal,
    adaptive_filtered_signal,
    peaks_indices_adaptive,
    expert_ground_truth,
    sampling_rate_hz,
    noise_freq=60.0
)


print("Ground Truth (Actual Beats):    " + str(len(expert_ground_truth)))

print("==============================")
print("Detected (Fixed Notch):         " + str(len(peaks_indices_fixed)))
print("Residual 60Hz Mag:              " + "{:.2f}".format(res_fixed))
print("SNR (dB):                       " + "{:.2f}".format(snr_fixed))
print("QRS Accuracy (%):               " + "{:.2f}".format(acc_fixed))
print("Missed Beats (FN):              " + str(fn_fixed))
print("False Detections (FP):          " + str(fp_fixed))
print("==============================")

print("Detected (Adaptive FFT):        " + str(len(peaks_indices_adaptive)))
print("Residual 60Hz Mag:              " + "{:.2f}".format(res_adaptive))
print("SNR (dB):                       " + "{:.2f}".format(snr_adaptive))
print("QRS Accuracy (%):               " + "{:.2f}".format(acc_adaptive))
print("Missed Beats (FN):              " + str(fn_adaptive))
print("False Detections (FP):          " + str(fp_adaptive))
print("==============================")