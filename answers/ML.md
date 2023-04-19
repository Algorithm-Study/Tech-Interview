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

### Local Minima와 Global Minimum에 대해 설명해주세요.

>💡 Gradient Descent 방법을 활용하여 Cost Function의 최솟값을 찾게 되는데 기울기가 0이 되는 점이 여러 개 존재할 수 있다. <br>
Local Minima(극소값)은 주위의 모든 점의 함숫값 이하의 함숫값을 갖는 점의 함숫값이다. <br>
Global Minimum(최솟값)은 정의역(x가 될 수 있는 범위)의 모든 점에서의 최소점의 함숫값을 의미한다.

- **Example**
   <center><img src="../img/ML/img6.png" width="70%" height="40%"></center>

    - Global Minimum(최솟값)은 항상 Local Minima(극소값)이다.<br>
    - 하지만 Local Minima(극소값)이 항상 Global Minimum(최솟값)이 되는 것은 아니다.<br>
    - 아래 그림에서 A는 Global Minimum(최솟값)이고 E, G는 Local Minima(극소값)이다.<br>

#### Reference
- [위키피디아 - 극값](https://ko.wikipedia.org/wiki/%EA%B7%B9%EA%B0%92)
- [Local Minima와 Global Minima에 대하여 설명해주세요](https://velog.io/@lswkim/Local-Minima%EC%99%80-Global-Minima%EC%97%90-%EB%8C%80%ED%95%B4-%EC%84%A4%EB%AA%85%ED%95%B4%EC%A3%BC%EC%84%B8%EC%9A%94)
- [Maxima vs Minima and Global vs Local in Machine learning - Basic Concept](https://medium.com/@dilip.voleti/maxima-vs-minima-and-global-vs-local-in-machine-learning-basic-concept-741e760e9f80)

---

### 차원의 저주에 대해 설명해주세요.
>💡 데이터를 잘 표현하는 예측 모델을 만들기 위해서는 다양한 차원이 필요합니다. 하지만 이런 차원이 증가할 수록 모델의 성능이 떨어지는 현상을 차원의 저주라고 일컫습니다.<br> 이런 현상이 발생하는 이유는 차원이 증가함에 따라 더 많은 차원을 표현할 수 있는 데이터가 필요해지고 기존 데이터로는 개별 차원마다 원활한 학습이 이루어지지 않기 때문입니다.

- 차원은 높은데 적은 데이터 수를 가지고 모델을 학습시키게 되면 이 모델은 과대적합된 모델이 된다. 그 이유는 차원이 높아 그만큼 데이터를 설정하는 변수의 수가 많지만 데이터의 수는 적기 때문에 실제 데이터 차원에 해당 되는 공간의 많은 경우들을 확인할 수 없기 때문에 모델이 학습 데이터에 과적합된 학습을 하여 성능이 낮아진다.
<center><img src="../img/ML/img7.png" width="70%" height="40%"></center>

- 차원의 저주(Curse of dimensionality) 현상은 수치 분석, 샘플링, 조합, 기계 학습, 데이터 마이닝 및 데이터베이스와 같은 영역에서 발생한다. 이러한 문제의 공통 주제는 차원이 증가하면 공간의 부피가 너무 빨리 증가하여 사용 가능한 데이터가 희소해진다는 것이다. 신뢰할 수 있는 결과를 얻기 위해 필요한 데이터의 양이 차원에 따라 기하급수적으로 증가하는 경우가 많다. 
    - 차원 = 변수의 수 = 축의 수
        - 차원이 늘어난다 = 변수의 수가 많아진다 = 축의 개수가 많아진다 = 데이터의 공간이 커진다
    - 1차원 공간에서의 1,000개의 데이터가 존재할 때, 1,000개 정도의 데이터만 있어도 빈 곳이 없다.

    <center><img src="../img/ML/img8.png" width="70%" height="40%"></center>

    - 2차원 영역을 다 채우기 위해서는 20,000개의 데이터가 필요
    <center><img src="../img/ML/img9.png" width="45%" height="40%"> <img src="../img/ML/img10.png" width="45%" height="55%"></center>

    - 3차원 영역을 다 채우기 위해서는 100,000개의 데이터가 필요
    <center><img src="../img/ML/img11.png" width="45%" height="40%"> <img src="../img/ML/img12.png" width="45%" height="55%"></center>

- 차원의 저주 해결방법

    - 데이터 추가 수집
    - 공간 벡터의 거리를 측정할 때 Euclidean distance 대신 Cosine Similarity 활용
    - 차원 줄이기
        - Forward-feature selection
        - PCA/t-SNE

#### Reference
- [차원의 저주(Curse of dimensionality)란? - 자윰이의 성장일기](https://think-tech.tistory.com/9)
- [[빅데이터] 차원의 저주(The curse of dimensionality)](https://think-tech.tistory.com/9)
- [Curse of Dimensionality - A “Curse” to Machine Learning](https://towardsdatascience.com/curse-of-dimensionality-a-curse-to-machine-learning-c122ee33bfeb)
- [차원의 저주](https://oi.readthedocs.io/en/latest/ml/curse_of_dimensionality.html)

---

### dimension reduction기법으로 보통 어떤 것들이 있나요?

>💡 **Dimension reduction**은 **Feature extraction**, **Feature selection** 두 가지로 나눌 수 있습니다.
>
>**Feature selection**의 **장점**은 선택한 피처의 해석이 용이하다는 점이고 **단점**은 피처간 상관관계를 고려하기 어렵다는 점입니다. filter, wrapper, embedded methods와 같은 방법들이 해당됩니다.
>
>**Feature extraction**의 **장점**은 피처 간 상관관계를 고려하기 용이하고 피처의 개수를 많이 줄일 수 있다는 점이고 **단점**은 추출된 변수의 해석이 어렵다는 점입니다. 이러한 **Feature extraction**은 Linear, Non-Linear로 다시 나뉩니다.

Feature selection

- Filter
    - 통계를 기반으로 greedy하게 차원을 축소하는 방법입니다. 예를 들어 대상과의 상관관계가 기준이 될 수 있습니다. 이 방법은 가장 빠르고 간단한 방법입니다.
- Embedded
    - 예측 알고리즘의 일부입니다. 예를 들어, 트리 기반 알고리즘은 본질적으로 데이터 세트 기능에 점수를 매기고 중요도에 따라 순위를 매깁니다. Lasso L1 정규화는 본질적으로 가중치를 0으로 떨어뜨림으로써 중복 기능을 제거합니다.
- Wrapper
    - 가장 유용한 기능을 식별하기 위해 기능의 하위 집합이 있는 Validation set에서 예측 알고리즘을 사용합니다. 최적의 하위 집합을 찾는 데에 많은 계산 비용이 요구됩니다. 래퍼 방법은 역방향/전방향 선택과 같은 탐욕적인 결정을 내립니다. 이 선택은 피쳐를 차례로 탐욕스럽게 제거/선택합니다.

Feature Projection

- Linear
    - Original Feature를 선형으로 결합하여 Original data set을 더 적은 차원으로 압축합니다. 일반적인 방법에는 주성분 분석(PCA), 선형 판별 분석(LDA) 및 특이값 분해(SVD)가 포함됩니다.
- Non-Linear
    - 좀더 복잡하지만 Linear method로는 해결하기 힘들 때 유용한 차원 감소를 찾을 수 있습니다. 비선형 차원 감소 방법에는 커널 PCA, t-SNE, Autoencoders, Self-Organizing Maps, IsoMap 및 UMap이 포함됩니다.

<center><img src="../img/ML/img13.png" width="70%" height="40%"></center>

#### Reference

- [11 Dimensionality reduction techniques you should know in 2021](https://towardsdatascience.com/11-dimensionality-reduction-techniques-you-should-know-in-2021-dcb9500d388b)
- [What Is Dimensionality Reduction? Meaning, Techniques, and Examples](https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-dimensionality-reduction/)
- [Applied Dimensionality Reduction — 3 Techniques using Python](https://www.learndatasci.com/tutorials/applied-dimensionality-reduction-techniques-using-python/)

---

### PCA는 차원 축소 기법이면서, 데이터 압축 기법이기도 하고, 노이즈 제거기법이기도 합니다. 왜 그런지 설명해주실 수 있나요?

>💡 PCA(Principal Component Analysis, 주성분 분석)의 기본 개념은 주어진 벡터에서 선형 독립인 고유 벡터만을 남겨두고 차원 축소를 하고. 이때 상관성이 높은 독립 변수들을 N개의 선형 조합으로 만들어 변수의 개수를 요약, 압축하는 방법입니다.
>
>사영 후 원데이터의 분산을 최대한 보전할 수 있는 기저를 찾아 차원을 줄이므로 차원 축소 기법이며 그 결과 feature들의 수가 기존보다 작아지기 때문에 데이터 압축 기법입니다. PCA 이후 정보 설명력이 높은 주성분들만 선택하고 정보 설명력이 낮은, 노이즈로 구성된 변수들은 배제하기 때문에 노이즈 제거 기법이기도 합니다. (**노이즈를 완전 제거하지는 못함!**)

<center><img src="../img/ML/img14.png" width="70%" height="40%"></center>

#### Reference

- [[선형대수학 #6] 주성분분석(PCA)의 이해와 활용](https://darkpgmr.tistory.com/110)
- [[Machine learning] 차원축소, PCA, SVD, LSA, LDA, MF 간단정리 (day1 / 201009)](https://huidea.tistory.com/126)