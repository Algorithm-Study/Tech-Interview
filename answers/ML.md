## Problem & Answer

### 알고 있는 metric에 대해 설명해주세요. (ex. RMSE, MAE, recall, precision ...)


>💡 metric은 크게 Classification metric과 Regression metric으로 나눌 수 있고 Classsification metric에는 Accuracy, F1 등이 있고 Regression metric에는 MSE, MASE 등이 있다


- Classification
    - Accuracy (정확도)
        - 전체중에서 True의 개수
        - 분류기의 성능을 측정할 때 가장 간단히 사용할 수 있음
        - optimize 하기 어려움
        - $TP +TN \over {TP +FP + TN + FN}$
    - Error Rate (오류율)
        - Accuracy의 반대로, 전체 데이터 중에서 잘못 분류한 FP, FN의 비율
        - $FP+FN \over {TP +FP + TN + FN}$
    - Confusion Matrix
        
        <center><img src="../img/ML/img0.png" width="30%" height="30%"></center>
        
        - Positive, Negative는 모델의 예측을 나타는 것
        - True/False는 예측의 정답 여부를 나타내는 것
        - True Positive(TP)는 Positive로 분류하고 맞은(True) 것, True Negative(TN)는 Negative로 분류하고 맞은 것을 나타냄
        - 위에서 표시된 TP, FP, FN, TN을 바탕으로 다양한 Metric을 계산 가능
        
        <center><img src="../img/ML/img1.png" width="80%" height="80%"></center>
        
        - 다중 클래스에서의 분류 결과에 대한 시각화(대각 성분 = 맞은 예측)
    - Preicision (정밀도)
        - Positive라고 예측한 것중에 맞은(True) 예측
        - $TP \over {TP + FP}$
    - Recall (재현율, 민감도)
        - True labeling Positive였던 것들 중에서 맞은 예측
        - FN는 Negative로 분류했으나  실제로는 Positive 였기 때문에 틀려서 FN이 된 것
        - $TP \over {TP + FN}$
    - Fall-Out
        - 실제로는 Negative인데 모델이 Positive로 오탐한 비율
        - $FP \over {FP + TN}$
    - F1 Score
        - 정밀도와 재현율의 관계는 trade-off이므로 정밀도와 재현율의 조화 평균을 metric으로 활용한 것이 F1 score이다.
        - Precision은 모두 Positive로 분류하면 100%가 되기 때문에 Recall과의 조화평균을 통해 성능을 평가하는 지표
        - $2\over{{1\over Recall}+{1\over Precision}}$
    - Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        - ROC : 분류 모델의 성능을 보여주는 그래프
        - AUC : 곡선 아래 영역
        - Fall out과 Recall을 통해 FPR, TPR을 X,Y축으로 두고 Threshold를 변경시키면서 그린 곡선을 ROC라고 한다
        - TPR : Sensitivity($TP\over{TP+FN}$) : 민감도, 재현율
        - FPR : specificity($FP\over{FP+TN}$) : 특이도
        - 이 때, ROC를 수치화 할 수 있는 방법이 딱히 없으므로, Area Under Curve라는 곡선 밑 부분의 넓이 값을 통해 성능을 측정한다.
        - Recall이 높고, Fall Out은 낮을 수록 넓이가 1에 가까워져 좋은 모델이 된다.
        - 이중 분류에만 사용
        - 특정 threshold를 설정
        - 예측의 순서에 의존적이며 절대값엔 의존적이지 않음
    
    <center><img src="../img/ML/img2.png" width="40%" height="40%"> <img src="../img/ML/img3.png" width="55%" height="55%"></center>
    
    - Precision Recall Curve
        - confidence 레벨에 대한 threshold 값의 변화에 따라 계산된 Precision 및 Recall을 그래프로 시각화한 것
        
        <center><img src="../img/ML/img4.png" width="80%" height="80%"></center>
        
        - 데이터 라벨의 분포가 심하게 불균등 할때 ROC 그래프보다 분석에 유리함
        - X축은 Recall 값을, Y축은 Precision 값을 사용
        - Base line= P / (P+N)을 기준으로 위에 위치할수록 좋은 모델
            - 따라서 아래의 경우에 A 모델이 더 좋은 모델임
            
            <center><img src="../img/ML/img5.png" width="80%" height="80%"></center>
            
        - F-Beta Score
            - F1 score에서 Recall에 가중치를 주어서 평가하는 Metric ($\beta$가 1인 경우 F1 score와 동일)
        - Average Precision
            - 정량적 평가를 위해 PR Curve의 아래 영역을 계산한 것
        - Recall At Fixed Precision
            - Precision threshold가 주어졌을 때  가장 큰 recall 값을 구하는 것
    - KL Divergence
        - 두 확률 분포의 차이를 수치로 표현한 것
    - Logloss
        - 잘못된 답변에 대해 더 강하게 패널티 부여
    - Hamming Distance
        - 두 길이가 같은 문자열 사이의 거리를 측정(몇개의 문자를 바꿔야 같아지는가?)
    - Jaccard index
        - 두 집합이 공통적으로 가진 것을 기반으로 하는 비유사성 측도
    - Label Ranking Average Precision
        - 멀티 레이블인 경우에 사용하는 AP
    - Label Ranking Loss
        - 멀티 레이블인 경우 사용하는 Loss
- Regression
    - Cosine Similarity
        - 유사도 측정
        $similarity = cos(\theta) = {{A \cdot B}\over \parallel A\parallel \parallel B\parallel}$
    - Explained Variance
        - $1- {{Sum\,of\,Squared\,Residuals - Mean\,Error}\over Total\,Variance}$
    - MAE (Mean Absolute Error)
        - 모델의 예측값과 실제값의 차이를 모두 더한 오차 절대값의 평균
        - MSE와 다르게 오차가 커도 큰 불이익을 주지 않는다.
        - Outlier의 영향을 받지 않음
    - MAPE (Mean Absolute Percentage Error)
        - MAE를 percent로 변환한 weight 버전
    - MSE (Mean Squared Error)
        - 실제 값과 예측 값 차이의 면적의 합
        - 오차를 제곱한 뒤 평균하여 산출
    - MSPE (Mean Squared Percentage Error)
        - MSE를 percent로 변환한 weight 버전
    - RMSE (Root Mean Squared Error)
        - 평균 오차 제곱합(MSE)에 루트를 씌워 오차율을 보정해줌
        - 회귀 metric으로 많이 사용됨
    - Mean Squared Log Error
        - 예측 값과 GT에 로그를 취한 뒤 차를 더한 것의 평균
    - R2 Score (R-squared, 결정계수)
        - $1- {{Sum\,of\,Squared\,Residuals}\over Total\,Variance}$
        - 총제곱합(SST)에 대한 회귀제곱합(SSR)을 뜻하며 결정계수라고도 불림
        - 결정계수는 반응변수의 변동량(분산)에서 현재 적용모델이 설명할 수 있는 부분의 비율을 뜻함
        - 예측의 적합도를 0과 1 사이의 값으로 계산하고, 1에 가까울 수록 설명력이 높다고 말함
    - RMSLE (Root Mean Squared Logarithmic Error)
        - RMSE에 비해 아웃라이어에 강건해짐
        - 상대적 Error를 측정할 수 있음
        - Under Estimation(예측값 < 실제값)에 큰 페널티를 부여
    - Pearson Correlation Coefficient
        - 두 변수 간의 선형 상관 관계를 계량화한 것
    - Spearman Correlation Coefficient
        - 두 변수 간의 단조적 상관 관계를 계량화한 것
    - SMAPE (Symmetric Mean Absolute Percentage Error)
        - $SMAPE = {100 \over n}  \times \displaystyle \sum^n_{i=1} {\lvert Y_i-\hat Y_i\rvert \over(\lvert Y_i\rvert + \lvert \hat Y_i\rvert) /2}$
    

#### Reference

- [[ML] Metric 종류](https://wooono.tistory.com/99)
- [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/all-metrics.html)
- [mAP 정리](https://ctkim.tistory.com/79)
- [분류모델의 성능 평가](https://bcho.tistory.com/1206#recentEntrie)
- [[ML 이론] EVALUATION METRIC 정리](https://cryptosalamander.tistory.com/177)
- [Introduction to the precision-recall plot](https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/)

---
