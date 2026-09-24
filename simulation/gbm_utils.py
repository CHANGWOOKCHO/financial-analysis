import numpy as np
from scipy.stats import norm

'''
#시뮬레이션 함수 정의

s_0 : 시작주가
mu : 기대수익률
sigma : 변동성
n_sims : 시뮬레이션할 주가 경로 개수
T : 시뮬레이션 기간
N : 시간 구간 설정 개수
random_seed : 난수 재현을 위한 시드
antithetic_var : 대조변수법 사용여부
'''
def simulate_gbm(s_0, mu, sigma, n_sims, T, N, random_seed=42, antithetic_var=False):
    np.random.seed(random_seed)
    
    dt = T/N  #시간 간격 설정
    
    if antithetic_var:
        dW_ant = np.random.normal(scale = np.sqrt(dt), size=(int(n_sims/2), N)) #브라운 운동의 랜덤 변화량 생성
        dW = np.concatenate((dW_ant, -dW_ant), axis=0)                          #같은 난수의 반대 방향 생성
    else: 
        dW = np.random.normal(scale = np.sqrt(dt), size=(n_sims, N))
  
    # simulate the evolution of the proces
    S_t = s_0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * dW, axis=1)) #GBM 공식
 
    return S_t

def black_scholes_analytical(S_0, K, T, r, sigma, type='call'):

    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if type == 'call':
        option_premium = (S_0 * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1))
    elif type == 'put':
        option_premium = (K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S_0 * norm.cdf(-d1, 0, 1))
    else:
        raise ValueError('Wrong input for type!')

    return option_premium


def lsmc_american_option(S_0, K, T, N, r, sigma, n_sims, option_type, poly_degree, random_seed=42):

    dt = T / N
    discount_factor = np.exp(-r * dt)

    gbm_simulations = simulate_gbm(s_0=S_0, mu=r, sigma=sigma, 
                                   n_sims=n_sims, T=T, N=N,
                                   random_seed=random_seed)

    if option_type == 'call':
        payoff_matrix = np.maximum(
            gbm_simulations - K, np.zeros_like(gbm_simulations))
    elif option_type == 'put':
        payoff_matrix = np.maximum(
            K - gbm_simulations, np.zeros_like(gbm_simulations))

    value_matrix = np.zeros_like(payoff_matrix)
    value_matrix[:, -1] = payoff_matrix[:, -1]

    for t in range(N - 1, 0, -1):
        regression = np.polyfit(
            gbm_simulations[:, t], value_matrix[:, t + 1] * discount_factor, poly_degree)
        continuation_value = np.polyval(regression, gbm_simulations[:, t])
        value_matrix[:, t] = np.where(payoff_matrix[:, t] > continuation_value,
                                      payoff_matrix[:, t],
                                      value_matrix[:, t + 1] * discount_factor)

    option_premium = np.mean(value_matrix[:, 1] * discount_factor)
    return option_premium
