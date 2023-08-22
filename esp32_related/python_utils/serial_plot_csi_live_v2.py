import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import collections
from wait_timer import WaitTimer
from read_stdin import readline, print_until_first_csi_line

# Set subcarrier to plot
subcarrier = 44

# Set numbers of packets to plot
nb_packets = 2000

# Wait Timers. Change these values to increase or decrease the rate of `print_stats` and `render_plot`.
print_stats_wait_timer = WaitTimer(0.2)
render_plot_wait_timer = WaitTimer(0.2)

# Deque definition
perm_amp = collections.deque(maxlen=nb_packets)
perm_phase = collections.deque(maxlen=nb_packets)

# Filtered vector definition
amp_f = []
ph_f = []

# Mean vectors definition
mean_amplitudes = []
mean_phases = []

# Variables to store CSI statistics
packet_count = 0
total_packet_counts = 0

# Create figure for plotting
# Adapt it according to your needs
plt.ion()
fig = plt.figure()
# fig.add_subplot(231)
# fig.add_subplot(232)
# fig.add_subplot(233)
# fig.add_subplot(234)
# fig.add_subplot(235)
# fig.add_subplot(236)
fig.add_subplot(211)
fig.add_subplot(212)
fig.canvas.draw()
plt.show(block=False)


def carrier_plot_amp(amp):
    plt.clf()
    # df = np.asarray(amp, dtype=np.int32)
    # Trouver la longueur maximale des éléments de amp
    max_length = max(len(a) for a in amp)

    # Créer un tableau numpy de forme (len(amp), max_length)
    df = np.vstack([np.pad(a, (0, max_length - len(a)), mode='constant') for a in amp])

    # Can be changed to df[x] to plot sub-carrier x only (set color='r' also)
    plt.plot(range(10000 - len(amp), 10000), df[:, subcarrier], color='r')
    plt.xlabel("Counts")
    plt.ylabel("Amplitude")
    plt.xlim(0, 10000)
    plt.title(f"Amplitude plot of Subcarriers 0 to {subcarrier}")
    # TODO use blit instead of flush_events for more fastness
    # to flush the GUI events
    fig.canvas.flush_events()
    plt.show()


def carrier_plot_phase(ph):
    plt.clf()
    # df = np.asarray(ph, dtype=np.int32)
    # Trouver la longueur maximale des éléments de ph
    max_length = max(len(a) for a in ph)

    # Créer un tableau numpy de forme (len(ph), max_length)
    df = np.vstack([np.pad(a, (0, max_length - len(a)), mode='constant') for a in ph])

    # Can be changed to df[x] to plot sub-carrier x only (set color='r' also)
    plt.plot(range(10000 - len(ph), 10000), df[:, subcarrier], color='b')
    plt.xlabel("Counts")
    plt.ylabel("Phase")
    plt.xlim(0, 10000)
    plt.title(f"Phase plot of Subcarriers 0 to {subcarrier}")
    # TODO use blit instead of flush_events for more fastness
    # to flush the GUI events
    fig.canvas.flush_events()
    plt.show()


# Adapt it according to your needs
def plot_amp_ph_v(amp, ph):
    plt.clf()

    # Maximum lenght of amp and ph
    max_length_amp = max(len(a) for a in amp)
    max_length_ph = max(len(a) for a in ph)

    # Numpy array with shape (len(amp), max_length)
    df_amp = np.vstack([np.pad(a, (0, max_length_amp - len(a)), mode='constant') for a in amp])
    df_ph = np.vstack([np.pad(a, (0, max_length_ph - len(a)), mode='constant') for a in ph])

    # Subplot 1 (Amplitude raw)
    plt.subplot(2, 1, 1)
    plt.plot(range(nb_packets - len(amp), nb_packets), df_amp[:, subcarrier], color='r')
    #plt.plot(range(10000 - len(amp), 10000), df_amp[subcarrier-1, subcarrier], color='r')
    plt.xlabel("Counts")
    plt.ylabel("Amplitude")
    plt.xlim(0, nb_packets)
    plt.title(f"Raw Amplitude")

    # Subplot 2 (Amplitude filtered)
    # plt.subplot(2, 3, 2)
    # plt.plot(range(nb_packets - len(amp_f), nb_packets), amp_f, color='r')
    # #plt.plot(range(10000 - len(amp), 10000), df_amp[subcarrier-1, subcarrier], color='r')
    # plt.xlabel("Counts")
    # plt.ylabel("Amplitude")
    # plt.xlim(0, nb_packets)
    # plt.title(f"Filtered Amplitude")

    # Subplot 3 (Amplitude filtered & averaged)
    plt.subplot(2, 1, 2)
    plt.plot(range(nb_packets - len(mean_amplitudes), nb_packets), mean_amplitudes, color='r')
    #plt.plot(range(10000 - len(amp), 10000), df_amp[subcarrier-1, subcarrier], color='r')
    plt.xlabel("Counts")
    plt.ylabel("Amplitude")
    plt.xlim(0, nb_packets)
    plt.title(f"Filtered & Averaged Amplitude")

    # Subplot 4 (Phase raw)
    # plt.subplot(2, 2, 3)
    # plt.plot(range(nb_packets - len(ph), nb_packets), df_ph[:, subcarrier], color='b')
    # #plt.plot(range(10000 - len(ph), 10000), df_ph[subcarrier-1, subcarrier], color='b')
    # plt.xlabel("Counts")
    # plt.ylabel("Phase")
    # plt.xlim(0, nb_packets)
    # plt.title(f"Raw Phase")

    # Subplot 5 (Phase filtered)
    # plt.subplot(2, 3, 5)
    # plt.plot(range(nb_packets - len(ph_f), nb_packets), ph_f, color='b')
    # #plt.plot(range(10000 - len(ph), 10000), df_ph[subcarrier-1, subcarrier], color='b')
    # plt.xlabel("Counts")
    # plt.ylabel("Phase")
    # plt.xlim(0, nb_packets)
    # plt.title(f"Filtered Phase")

    # Subplot 6 (Phase filtered & averaged)
    # plt.subplot(2, 2, 4)
    # plt.plot(range(nb_packets - len(mean_phases), nb_packets), mean_phases, color='b')
    # #plt.plot(range(10000 - len(ph), 10000), df_ph[subcarrier-1, subcarrier], color='b')
    # plt.xlabel("Counts")
    # plt.ylabel("Phase")
    # plt.xlim(0, nb_packets)
    # plt.title(f"Filtered & Averaged Phase")

    # Show
    fig.canvas.flush_events()
    plt.show()


def outliers_rm(amp, ph):
    res_amp = amp[0:5]
    for i in range(5, len(amp)):
        mean = (amp[i-4]+amp[i-3]+amp[i-2]+amp[i-1])/4
        if amp[i] > mean*2 :
            res_amp.append(mean)
        else :
            res_amp.append(amp[i])

    res_ph = ph[0:5]
    for j in range(5, len(ph)):
        mean = (ph[i-4]+ph[i-3]+ph[i-2]+ph[i-1])/4
        if ph[i] > mean*2 :
            res_ph.append(mean)
        else :
            res_ph.append(ph[i])

    return res_amp, res_ph
    

def process(res):
    # Parser
    all_data = res.split(',')
    csi_data = all_data[25].split(" ")
    csi_data[0] = csi_data[0].replace("[", "")
    csi_data[-1] = csi_data[-1].replace("]", "")

    csi_data.pop()
    csi_data = [int(c) for c in csi_data if c]
    imaginary = []
    real = []
    for i, val in enumerate(csi_data):
        if i >= 12 and i <= 117:
            if i % 2 == 0:
                imaginary.append(val)
            else:
                real.append(val)

    csi_size = len(csi_data)-22
    amplitudes = []
    phases = []
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            phase_calc = math.atan2(imaginary[j], real[j])
            amplitudes.append(amplitude_calc)
            phases.append(phase_calc)

        perm_amp.append(amplitudes)
        perm_phase.append(phases)
        
        amp_f0, ph_f0 = outliers_rm(amplitudes, phases)
        amp_f.append(amp_f0)
        ph_f.append(ph_f0)

        if len(amp_f) > 5 and len(ph_f) > 5:
            temp_amp = (mean_amplitudes[-4]+mean_amplitudes[-3]+mean_amplitudes[-2]+mean_amplitudes[-1]+amp_f0[-1])/5
            temp_ph  = (ph_f0[-5]+ph_f0[-4]+ph_f0[-3]+ph_f0[-2]+ph_f0[-1])/5
            mean_amplitudes.append(temp_amp)
            mean_phases.append(temp_ph)
        else:
            mean_amplitudes.append(np.mean(amp_f0))
            mean_phases.append(np.mean(ph_f0))



print_until_first_csi_line()

while True:
    line = readline()
    if "CSI_DATA" in line:
        process(line)
        packet_count += 1
        total_packet_counts += 1

        if print_stats_wait_timer.check():
            print_stats_wait_timer.update()
            print("Packet Count:", packet_count, "per second.", "Total Count:", total_packet_counts)
            packet_count = 0

        if render_plot_wait_timer.check() and len(perm_amp) > 2 and len(mean_amplitudes) > 2:
            render_plot_wait_timer.update()
            plot_amp_ph_v(perm_amp, perm_phase)
