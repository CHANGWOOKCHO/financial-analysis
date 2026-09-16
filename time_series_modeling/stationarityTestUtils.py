import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

warnings.filterwarnings("ignore")

def adf_test(x):
    indices = ['Test Statistic', 'p-value',
               '# of Lags Used', '# of Observations Used']

    adf_test = adfuller(x, autolag='AIC')
    results = pd.Series(adf_test[0:4], index=indices)

    for key, value in adf_test[4].items():
        results[f'Critical Value ({key})'] = value

    return results


def kpss_test(x, h0_type='c'):
    indices = ['Test Statistic', 'p-value', '# of Lags']

    kpss_test = kpss(x, regression=h0_type)
    results = pd.Series(kpss_test[0:3], index=indices)

    for key, value in kpss_test[3].items():
        results[f'Critical Value ({key})'] = value

    return results

def test_autocorrelation(x, n_lags=40, alpha=0.05, h0_type='c'):
    SIG_LEVEL = 0.05 # 5% 유의수준설정
    adf_results = adf_test(x)
    kpss_results = kpss_test(x, h0_type=h0_type)

    adf_txt_result = "정상성 데이터" if adf_results['p-value'] < SIG_LEVEL else "비정상성 데이터"
    kpss_txt_result = "비정상성 데이터" if kpss_results['p-value'] < SIG_LEVEL else "정상성 데이터"
    
    print(f"ADF 검정통계량: {adf_results['Test Statistic']:.2f} (p-value: {adf_results['p-value']:.2f}) -> 검정결과:{adf_txt_result}")
    print(f"KPSS 검정통계량: {kpss_results['Test Statistic']:.2f} (p-value: {kpss_results['p-value']:.2f}) -> 검정결과:{kpss_txt_result}")

    fig, ax = plt.subplots(2, figsize=(16, 8))
    plot_acf(x, ax=ax[0], lags=n_lags, alpha=alpha)
    plot_pacf(x, ax=ax[1], lags=n_lags, alpha=alpha)

    return fig