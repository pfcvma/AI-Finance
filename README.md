# AI-Finance
Sogang University - AI Finance (Capstone Design)
> Final Grade: A0
> 
> 발표자료: [AI금융_최종발표.pdf](https://drive.google.com/file/d/1mjXmIR7LGPpoW9KtJJkbiDYgfw9uworg/view?usp=sharing)

<br>

## 주제

**ESG 우수기업 포트폴리오 구성 및 시장 수익률과의 비교를 통한 초과수익여부 검증**
<br>
([SWIC - ESG Active Fund Project](https://github.com/pfcvma/PythonStockTool) 및 [Financial Big Data Analysis - ESG Correlation Project](https://github.com/pfcvma/esg_return_correlation)의 심화연구)

<br>

## 목차

1. ESG 투자란?
2. ESG 우수 기업의 수익성
    1. ESG 관련 ETF와 시장 수익률과의 비교
    2. ESG 미흡 기업과 우수 기업 수익률 비교: ESG의 우수성에 관한 논문 분석
3. 투자 전략
    1. ESG 투자의 효과 및 안정성
        1. 2021년 데이터 전수조사, 다중회귀분석, 로지스틱, 클러스터링(PCA, TSNE) 통한 직접 분석
        2. Sector 변수 : ESG와 시가총액이 수익률에 미치는 영향과는 큰 관련성이 없음
        ESG와 수익률 : 2021년 기준, 유의미한 관련성이 확인되지 않음
        ESG와 시가총액 : ESG 등급이 높은 그룹 = 시가총액이 높은 기업들이 포함된 그룹
    2. 섹터별 ESG 등급 우수기업 선정: ESG 등급 + 상대가치 분석 (PER/ROE) + ESG 등급 상승 여부
    3. 종목별 포트폴리오 비중 구성
    4. 백테스팅
4. 전략 실행(포트폴리오 비중 구성 및 백테스팅)
    1. Harry Markowitz의 포트폴리오 선택 이론
        1. 최적의 포트폴리오는 각 위험 수준 하에서 가장 효율적인 포트폴리오의 집합인 효율적 프론티어(Efficient Frontier)와 효율적 투자선(Sharpe Ratio)의 접점 위에 존재한다. 해당 이론을 기반으로, Python을 활용하여 펀드 내 자산군 간 표준편차가 가장 낮으면서도 수익률이 가장 높은 포트폴리오 값을 산출한다. 이를 통해 투자 성과가 가장 뛰어난 포트폴리오를 구성한다.
        2. 포트폴리오의 기대효용 극대화를 목표로 하며, 위험도는 분산으로 측정되어 기대수익과 위험도의 최적값이 되는 비율을 채택한다. Sharpe Ratio가 최대가 되는 지점을 찾아서 투자 성과가 가장 좋은 펀드 비율을 찾아낸다.
    2. 포트폴리오 비중 구하기 - ML 기반, 최적의 초과수익률을 달성하는 포트폴리오 산출
        1. 최적화된 평균 분산 포트폴리오(Mean-Variance Optimization, MVO)를 구하기
        2. Python의 라이브러리인 `PyPortfolioOpt`를 사용
        - 필요한 라이브러리 설치:
        
        ```python
        pip install PyPortfolioOpt
        ```
        
        - 주식 데이터 가져오기 및 전처리:
        
        ```python
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from pypfopt import EfficientFrontier
        from pypfopt import risk_models
        from pypfopt import expected_returns
        
        # 주식 종목 선택
        tickers = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT']
        
        # 주식 데이터 다운로드
        prices = yf.download(tickers, start='2018-01-01', end='2023-01-01')['Adj Close']
        
        # 연간 수익률 및 공분산 행렬 계산
        returns = expected_returns.mean_historical_return(prices)
        cov_matrix = risk_models.sample_cov(prices)
        ```
        
        - 최적화된 평균 분산 포트폴리오 계산:
        
        ```python
        # 최적화된 포트폴리오 계산
        ef = EfficientFrontier(returns, cov_matrix)
        weights = ef.max_sharpe()  # 최대 샤프 비율을 가진 포트폴리오를 찾습니다.
        cleaned_weights = ef.clean_weights()
        
        # 결과 출력
        print(cleaned_weights)
        ```
        
        - 위 코드는 주어진 주식 종목들에 대해 최적화된 평균 분산 포트폴리오를 계산하고, 각 주식의 비중을 출력
        - 이를 통해 효율적인 주식 포트폴리오 비중을 도출
    3. 결과 분석 및 시장 수익률과의 비교, 초과수익률 도출
        1. 2022년 시장 수익률 vs ESG 포트폴리오(모의투자), 그래프 시각화
5. 모의투자
    1. 실제 2022년(1년간) 모의투자 가정 → 모의투자 결과 분석
        - Python의 `backtrader` 라이브러리를 사용하여 백테스팅 수행
        1. 필요한 라이브러리 설치:
        
        ```python
        pip install backtrader
        ```
        
        1. 백테스팅 전략 정의:
        
        ```python
        import backtrader as bt
        
        class MLStrategy(bt.Strategy):
            def __init__(self):
                self.data_close = self.datas[0].close
        
            def next(self):
                # 머신러닝 모델을 사용하여 포트폴리오 비중을 결정하는 로직 구현
                # 예를 들어, cleaned_weights를 사용하여 포트폴리오를 구성
                # 이 예제에서는 단순히 매수 신호만 발생시키는 전략을 사용
                if not self.position:
                    self.buy()
        
        ```
        
        1. 백테스팅 환경 설정 및 실행:
        
        ```python
        import yfinance as yf
        
        # 주식 데이터 다운로드
        data = yf.download('AAPL', start='2018-01-01', end='2023-01-01')
        
        # 백테스팅 환경 설정
        cerebro = bt.Cerebro()
        cerebro.addstrategy(MLStrategy)
        cerebro.broker.set_cash(100000)  # 초기 자본 설정
        
        # 주식 데이터 추가
        feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(feed)
        
        # 백테스팅 실행
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        
        # 백테스팅 결과 시각화
        cerebro.plot()
        
        ```
        
        - 위 코드는 `backtrader` 라이브러리를 사용하여 주식 포트폴리오의 백테스팅을 수행.
        - `MLStrategy` 클래스에서 머신러닝 모델을 사용하여 포트폴리오 비중을 결정하는 로직을 구현
    
6. 추가연구
    1. ESG Index 자체가 유의미했는가?
        - 데이터 2022년 1년간 분석 한계: ESG 누적 데이터 수집 어려움
        - ESG Index 자체가 유의미했는가?
        - 국내 데이터 한정, 해외는? : 시장 환경에 따른 차이
    2. 데이터 전처리 수행
        - 독립변수 : 시가총액(AV_log, 로그 변환), ESG, 섹터 더미변수(Sector)
        - 종속변수 : ret
        - Crawled ESG Index data of 500+ Korean companies, and relevant corporate return data for 2 years (except for COVID-19) using `yfinance` module
    3. 기초통계량 분석, Scatter Plotting
        - 시가총액 대비 Return에서 E, S, G의 분포
    4. 다중회귀분석
        - 전체 표본 및 섹터별 다중회귀분석
        - 전체 및 섹터별 ESG, E, S, G score과 Return의 관계 파악
        - R2 = 0.125
    5. Logistic Regression
        - G(지배구조 점수)가 높을수록 양의 평균 return이 될 확률이 낮아짐: G Score – Return plotting에서 고득점일수록 음수 return이 많음
        - log_AV(시가총액 로그)가 높을수록 양의 평균 return이 될 확률이 높아짐
    6. PCA, TSNE, K-Means Clustering
        - Model 1: 기업 Sector 포함. Sector 더미변수 + AV_log + E + S + G
        - Model 2: 기업 Sector 제외. AV_log + E + S + G
        - Elbow Score, Biplot of Eigenvector 확인
        - Sector보다 E, S, G에 의해 Cluster가 나누어지는 경향이 더 강함, Model 2에서는 Sector 더미변수를 제외한 Clustering 분석 수행

7. 결론
    1. Sector 변수
        - ESG와 시가총액이 수익률에 미치는 영향과는 큰 관련성이 없음
    2. ESG와 수익률
        - 2022년 기준, 유의미한 관련성이 확인되지 않음
    3. ESG와 시가총액
        - ESG 등급이 높은 그룹 = 시가총액이 높은 기업들이 포함된 그룹

<br>

    ## 피드백
    
    - 한국 vs 해외 (유럽 등), 한국 esg 등급과 외국 esg 등급 우수 이들의 수익률은 어떤지도 비교
    - 즉 시장환경에 따른 차이도 같이 분석 (ex. 해외는 에너지 별로 안 쓰는 클라우드 기업 등이 esg 높음)
    - esg 인덱싱이 천차만별. 해당 기준과 관련하여서도 분석 필요
    - 시가총액과 따라감. 생긴지 얼마 안 된 기준이기도 해서, 실제 교수님 인사이트와 유사한 결론 도출됨.
    - Library 생성, 나만의 human capital 도출
    - 대학원생 컨택하기(?)
