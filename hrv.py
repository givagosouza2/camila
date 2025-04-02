from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features, get_frequency_domain_features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hrvanalysis import plot_psd
from hrvanalysis import plot_poincare
from hrvanalysis import plot_timeseries
import streamlit as st
from scipy.signal import butter, filtfilt, welch, detrend
import scipy.interpolate

# Função para calcular e plotar a PSD


def calcular_psd(nn_intervals_list, fs=4):
    nn_intervals_sec = np.array(nn_intervals_list) / 1000
    tamanho_lista = len(nn_intervals_sec)
    intervalo = (600 - 0) / (tamanho_lista - 1)
    time_original = [round(0 + i * intervalo, 2) for i in range(tamanho_lista)]
    time_uniform = np.arange(0, time_original[-1], 1/fs)
    nn_uniform = np.interp(time_uniform, time_original, nn_intervals_sec)
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

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Escolha o arquivo referentes a variação da frequência cardíaca", type="txt")
    if uploaded_file is not None:
        df_vfc = pd.read_csv(uploaded_file, header=None, names=['Value'])
        rr_intervals = remove_outliers(
            rr_intervals=df_vfc['Value'], low_rri=300, high_rri=2000)
        rr_intervals = interpolate_nan_values(
            rr_intervals=rr_intervals, interpolation_method="linear")
        nn_intervals = remove_ectopic_beats(
            rr_intervals=rr_intervals, method="malik")
        nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals)

        time_domain = get_time_domain_features(nn_intervals)
        frequency_domain = get_frequency_domain_features(nn_intervals)

        tamanho_lista = len(nn_intervals)
        intervalo = (10 - 0) / (tamanho_lista - 1)
        tempo_min = [round(0 + i * intervalo, 2) for i in range(tamanho_lista)]

        nn_series = pd.Series(nn_intervals)
        moving_average = nn_series.rolling(window=10).mean()

        fig, ax = plt.subplots()
        ax.set_ylabel("RR Interval (ms)", fontsize=15)
        ax.set_xlabel("Time (min)", fontsize=15)
        ax.plot(tempo_min, nn_intervals, 'k')
        ax.plot(tempo_min, moving_average, 'r')
        ax.plot([4, 4], [600, 1000], '--b')
        ax.plot([5, 5], [600, 1000], '--r')
        st.pyplot(fig)

        try:
            nn_sec = np.array(nn_intervals) / 1000
            frequencies, psd = calcular_psd(nn_sec, 4)

            st.subheader("Gráfico da Densidade Espectral de Potência (PSD)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(frequencies, psd)
            ax.set_title('Densidade Espectral de Potência (PSD)')
            ax.set_xlabel('Frequência (Hz)')
            ax.set_ylabel('Densidade de Potência')
            ax.grid()
            st.pyplot(fig)

            sdnn = np.std(nn_intervals, ddof=1)
            rmssd = np.sqrt(np.mean(np.diff(nn_intervals) ** 2))
            diff_nn = np.abs(np.diff(nn_intervals))
            pnn50 = np.sum(diff_nn > 50) / len(diff_nn) * 100

            def calcular_potencia(f, p, faixa):
                idx = np.logical_and(f >= faixa[0], f < faixa[1])
                return np.trapz(p[idx], f[idx])

            vlf = calcular_potencia(frequencies, psd, (0.003, 0.04))
            lf = calcular_potencia(frequencies, psd, (0.04, 0.15))
            hf = calcular_potencia(frequencies, psd, (0.15, 0.4))
            total = vlf + lf + hf
            lf_rel = (lf / total) * 100
            hf_rel = (hf / total) * 100
            lf_hf = lf / hf

            st.text(f'VLF power: {vlf}')
            st.text(f'LF power: {lf}')
            st.text(f'HF power: {hf}')
            st.text(f'Total power: {total}')
            st.text(f'Relative LF: {lf_rel}')
            st.text(f'Relative HF: {hf_rel}')
            st.text(f'LF/HF: {lf_hf}')

            output_file = "output.txt"
            with open(output_file, "w") as file:
                file.write(
                    f"{vlf}\n{lf}\n{hf}\n{total}\n{lf_rel}\n{hf_rel}\n{lf_hf}\n")
            with open(output_file, "r") as file:
                contents = file.read()
            st.download_button("Baixar resultados - VFC",
                               data=contents, key='download_results_vfc')

        except Exception as e:
            st.error(f"Erro ao processar os dados: {e}")

with col2:
    uploaded_file = st.file_uploader(
        "Importe o arquivo referente ao tremor de mão", type=["csv"])
    if uploaded_file is not None:
        df_tremor = pd.read_csv(uploaded_file, sep=';')
        t = df_tremor.iloc[:, 0]
        x = detrend(df_tremor.iloc[:, 1])
        y = detrend(df_tremor.iloc[:, 2])
        z = detrend(df_tremor.iloc[:, 3])

        t_ = np.arange(start=t[0], stop=t[len(t)-1], step=10)
        x_interp = scipy.interpolate.interp1d(t, x)(t_)
        y_interp = scipy.interpolate.interp1d(t, y)(t_)
        z_interp = scipy.interpolate.interp1d(t, z)(t_)

        t_ = t_ / 1000
        t_point = t_ / 60

        x_filt = butterworth_filter(x_interp, 15, 100)
        y_filt = butterworth_filter(y_interp, 15, 100)
        z_filt = butterworth_filter(z_interp, 15, 100)

        rms_norm = []
        tempo_rms = []
        for i in range(len(z_filt)-5):
            rms_x = np.mean(np.square(x_filt[i:i+5]))
            rms_y = np.mean(np.square(y_filt[i:i+5]))
            rms_z = np.mean(np.square(z_filt[i:i+5]))
            rms_norm.append(np.sqrt(rms_x + rms_y + rms_z))
            tempo_rms.append(t_point[i])

        norma_series = pd.Series(rms_norm)
        moving_average_tremor = norma_series.rolling(window=10).mean()

        fig, ax = plt.subplots()
        ax.plot(tempo_rms, rms_norm, 'black')
        ax.plot(tempo_rms, moving_average_tremor, 'r')
        ax.plot([5, 5], [np.min(rms_norm), np.max(rms_norm)], '--r')
        ax.plot([4, 4], [np.min(rms_norm), np.max(rms_norm)], '--b')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMS amplitude (g)')
        st.pyplot(fig)

        for index, t_val in enumerate(tempo_rms):
            if t_val >= 5:
                t_start = index
                break

        def media_segura(lista, i1, i2):
            return np.mean(lista[i1:i2]) if i2 <= len(lista) else np.nan

        resultados = [
            media_segura(rms_norm, t_start-600, t_start-300),
            media_segura(rms_norm, t_start-300, t_start),
            media_segura(rms_norm, t_start, t_start+300),
            media_segura(rms_norm, t_start+300, t_start+600),
            media_segura(rms_norm, t_start+600, t_start+900),
            media_segura(rms_norm, t_start+900, t_start+1200),
            media_segura(rms_norm, t_start+1200, t_start+1500),
            media_segura(rms_norm, t_start+1500, t_start+1800),
            media_segura(rms_norm, t_start+1800, t_start+2100),
            media_segura(rms_norm, t_start+2100, t_start+2400),
            media_segura(rms_norm, t_start+2400, t_start+2700),
            media_segura(rms_norm, t_start+2700, t_start+3000),
        ]

        output_file = "output.txt"
        with open(output_file, "w") as file:
            file.write("\n".join(str(val)
                       for val in resultados if not np.isnan(val)))

        with open(output_file, "r") as file:
            contents = file.read()
        st.download_button("Baixar resultados - Tremor",
                           data=contents, key='download_results_tremor')
