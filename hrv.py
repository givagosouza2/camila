pip install git+https://github.com/Aura-healthcare/hrvanalysis.git

from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features, get_frequency_domain_features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hrvanalysis import plot_psd
import seaborn as sns
from hrvanalysis import plot_poincare
from hrvanalysis import plot_timeseries
import streamlit as st
from scipy.signal import butter, filtfilt
from scipy import signal
import scipy.interpolate
from scipy.signal import welch

# Função para calcular e plotar a PSD


def calcular_psd(nn_intervals_list, fs=4):
    # Passo 1: Converter para segundos
    nn_intervals_sec = np.array(nn_intervals_list) / 1000

    # Passo 2: Criar uma série temporal uniforme usando interpolação
    # Tamanho da lista
    tamanho_lista = len(nn_intervals_sec)

    # Calcular o intervalo entre os valores
    intervalo = (600 - 0) / (tamanho_lista - 1)

    # Criar a sequência usando range e ajustando para valores float
    time_original = [round(0 + i * intervalo, 2) for i in range(tamanho_lista)]
    time_uniform = np.arange(0, time_original[-1], 1/fs)  # Tempo uniforme
    nn_uniform = np.interp(time_uniform, time_original, nn_intervals_sec)

    # Passo 3: Calcular a PSD usando o método de Welch
    frequencies, psd = welch(nn_uniform, fs=fs, nperseg=256)

    return frequencies, psd


def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype, analog=False)
    y = filtfilt(b, a, data)
    return y


st.set_page_config(layout="wide")
st.title('Rotina para análise de dados do projeto de Camila Braga')
# Abrindo o arquivo de texto
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Escolha o arquivo referentes a variação da frequência cardíaca", type="txt")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None, names=['Value'])
        rr_intervals_without_outliers = remove_outliers(rr_intervals=df['Value'],
                                                        low_rri=300, high_rri=2000)
        interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                           interpolation_method="linear")

        # This remove ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(
            rr_intervals=interpolated_rr_intervals, method="malik")
        # This replace ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(
            rr_intervals=nn_intervals_list)

        time_domain_features = get_time_domain_features(
            interpolated_nn_intervals)
        frequency_domain_features = get_frequency_domain_features(
            interpolated_nn_intervals)

        # Tamanho da lista
        tamanho_lista = len(interpolated_nn_intervals)

        # Calcular o intervalo entre os valores
        intervalo = (10 - 0) / (tamanho_lista - 1)

        # Criar a sequência usando range e ajustando para valores float
        tempo_min = [round(0 + i * intervalo, 2) for i in range(tamanho_lista)]

        nn_series = pd.Series(interpolated_nn_intervals)

        # Calcular média móvel com janela de tamanho 3
        window_size = 60
        moving_average = nn_series.rolling(window=window_size).mean()

        # Criando o gráfico
        fig, ax = plt.subplots()
        ax.set_ylabel("RR Interval (ms)", fontsize=15)
        ax.set_xlabel("Time (min)", fontsize=15)
        ax.plot(tempo_min, interpolated_nn_intervals, 'k')
        ax.plot(tempo_min, moving_average, 'r')
        ax.plot([4, 4], [600, 1000], '--b')
        ax.plot([5, 5], [600, 1000], '--b')

        st.pyplot(fig)

        try:
            # Normalizar para segundos
            interpolated_nn_intervals_sec = np.array(
                interpolated_nn_intervals) / 1000
            frequencies, psd = calcular_psd(interpolated_nn_intervals_sec, 4)

            # Exibir gráfico
            st.subheader("Gráfico da Densidade Espectral de Potência (PSD)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(frequencies, psd)  # Escala logarítmica no eixo Y
            ax.set_title('Densidade Espectral de Potência (PSD)')
            ax.set_xlabel('Frequência (Hz)')
            ax.set_ylabel('Densidade de Potência')
            ax.grid()
            st.pyplot(fig)

            # SDNN
            sdnn = np.std(interpolated_nn_intervals, ddof=1)
            # RMSSD
            rmssd = np.sqrt(np.mean(np.diff(interpolated_nn_intervals) ** 2))
            # pNN50
            diff_nn = np.abs(np.diff(interpolated_nn_intervals))
            pnn50 = np.sum(diff_nn > 50) / len(diff_nn) * 100

            # Faixas de frequência
            vlf_band = (0.003, 0.04)
            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.4)

            # Função para calcular potência em uma faixa
            def calcular_potencia(frequencies, psd, faixa):
                idx = np.logical_and(
                    frequencies >= faixa[0], frequencies < faixa[1])
                return np.trapz(psd[idx], frequencies[idx])

            # Potências absolutas
            vlf_power = calcular_potencia(frequencies, psd, vlf_band)
            lf_power = calcular_potencia(frequencies, psd, lf_band)
            hf_power = calcular_potencia(frequencies, psd, hf_band)

            # Potência total
            total_power = vlf_power + lf_power + hf_power

            # Potências relativas
            lf_relative = (lf_power / total_power) * 100
            hf_relative = (hf_power / total_power) * 100

            # Razão LF/HF
            lf_hf_ratio = lf_power / hf_power
        except Exception as e:
            st.error(f"Erro ao processar os dados: {e}")
    # plot_psd(nn_intervals_list, method="welch")
        # plot_psd(nn_intervals_list, method="lomb")
        # plot_poincare(nn_intervals_list)
        # plot_poincare(nn_intervals_list, plot_sd_features=True)
with col2:
    uploaded_file = st.file_uploader(
        "Importe o arquivo referente ao tremor de mão", type=["csv"])
    if uploaded_file is not None:
        custom_separator = ';'
        # Allocation of the data to the variables
        df = pd.read_csv(uploaded_file, sep=custom_separator)
        t = df.iloc[:, 0]
        x = df.iloc[:, 1]
        y = df.iloc[:, 2]
        z = df.iloc[:, 3]

        x = signal.detrend(x)
        y = signal.detrend(y)
        z = signal.detrend(z)

        t_ = np.arange(start=t[0], stop=t[len(t)-1], step=10)
        f1 = scipy.interpolate.interp1d(t, x)
        x_interp = f1(t_)
        f2 = scipy.interpolate.interp1d(t, y)
        y_interp = f2(t_)
        f3 = scipy.interpolate.interp1d(t, z)
        z_interp = f3(t_)
        t_ = t_/1000
        t_point = (t_/60)

        x_filt = butterworth_filter(x_interp, 15, 100, order=4, btype='low')
        y_filt = butterworth_filter(y_interp, 15, 100, order=4, btype='low')
        z_filt = butterworth_filter(z_interp, 15, 100, order=4, btype='low')

        # norm calculation

        rms_norm = []
        rms_x = []
        rms_y = []
        rms_z = []
        tempo_rms = []
        for i in range(len(z_filt)-5):
            rms_x.append(np.mean(x_filt[i+5])**2)
            rms_y.append(np.mean(y_filt[i+5])**2)
            rms_z.append(np.mean(z_filt[i+5])**2)
            rms_norm.append(np.sqrt(rms_x[i]+rms_y[i]+rms_z[i]))
            tempo_rms.append(t_point[i])

        fig, ax = plt.subplots()
        signal = rms_norm
        ax.plot(tempo_rms, signal, 'black')
        ax.plot([5, 5], [np.min(signal), np.max(signal)], '--r')
        ax.plot([4, 4], [np.min(signal), np.max(signal)], '--b')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMS amplitude (g)')
        st.pyplot(plt)
