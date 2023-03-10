## Problem & Answer

### 고유값(eigen value)와 고유벡터(eigen vector)이 무엇이고 왜 중요한지 설명해주세요.

$n \times n$ 행렬 $A$를 선형변환으로 봤을 때, <b>선형변환 $A$에 의한 변환 결과가 자기 자신의 상수배가 되는 0이 아닌 벡터를 고유벡터라고 하고 이 상수배 값을 고유값</b>이라 한다.   
- 선형변환(Linear Transformation): 선형 결합을 보존하는 두 벡터 공간 사이의 함수   
$T(a+b) = T(a) + T(b), T(ca) = cT(a)$를 만족하는 변환.   
- 아래와 같은 식을 만족하는 열벡터 $v$를 eigen vector, 상수 $\lambda$를 eigen value라 한다.   

$$
Av = \lambda v
$$

- eigen vector, eigen value는 임의의 벡터를 어느 방향으로 변화시켰는지, 변환 과정에서 변화 없이 유지 되는 부분은 어느 부분인지에 대한 정보를 담고 있다.
- 어떤 물체나 영상 등은 수많은 벡터의 뭉치로 볼 수 있는데 eigen vector와 value를 활용해 물체나 영상이 어떤 식으로 변화하는지에 대한 정보를 파악할 수 있게 도와준다.
- 데이터의 특징을 파악할 수 있게 도와주는 SVD(특이값분해), Pseudo-Inverse, 선형연립방정식의 풀이, PCA(주성분분석)에 사용한다.

![data image](../img/Math/img0.png)

각 데이터의 Feature 간의 유사도는 모델 학습 결과에 영향을 끼치기 때문에 고유값과 고유벡터를 구해서 완전 독립 관계인 feature를 생성 혹은 추출하는 것이 중요하다.

#### Reference

- [[선형대수학 #3] 고유값과 고유벡터 (eigenvalue & eigenvector)](https://darkpgmr.tistory.com/105)
- [고유값(eigen value)과 고유벡터(eigen vector), 왜 중요한가?](https://kejdev.github.io/machinelearning/2021/01/04/eigen-value-eigen-vecotor.html)
- [eigen vector & eigen value](https://variety82p.tistory.com/entry/eigen-vector-eigenvalue?category=996031)

---

### 샘플링(Sampling)과 리샘플링(Resampling)이 무엇이고 리샘플링의 장점을 말해주세요.

샘플링은 모집단에서 일부만을 뽑아내서 모집단 전체의 경향성을 살펴보고 싶어 사용하는 방법으로 표본추출이라고 한다.
- Sampling 방식
    - Probability Sampling
        - Simple Random Sampling
        - Cluster Sampling(군집 표집)
        - Stratified Sampling(층화 표집)
        - Systematic Sampling(계통 표집)
    - Non-Probability Sampling
        - Convenience Sampling(편의 표본 추출)
        - Judgmental/Purposive Sampling(판단 표본)
        - Snowball/Referral Sampling(눈덩이 표집)
        - Quota Sampling( 할당량 샘플링)
- Sampling 장점
    - 시간과 비용을 줄임
    - 데이터가 정확성
    - 더 많은 모집단에 대한 예측 가능
    - 더 적은 자원으로 진행 가능

하지만 매우 정교한 추출이 이루어져도 모집단과 정확하게 일치할 수는 없으므로 이를 보완하기 위해 샘플링된 데이터에서 부분집합을 뽑아 통계량의 변동성을 확인하는 방법을 사용하는데 이를 리샘플링이라고 한다.
- 대표적인 리샘플링 기법으로는 k-fold 교차검증, bootstrapping 기법이 존재
- k-fold: k-1개의 부분집합들을 훈련 세트로 사용하고 나머지 하나의 부분집합을 테스트 세트로 사용하는 것을 말함
    - k번의 훈련과 테스트를 거쳐 결과의 평균을 구할 수 있음
- bootstrapping
    1. 표본 중 m개를 뽑아 기록하고 다시 제자리에 둔다.
    2. 이를 n번 반복한다.
    3. n번 재표본추출한 값의 평균을 구한다.
    4. 1~3단계를 R번 반복한다.(R: 부트스트랩 반복 횟수)
    5. 평균에 대한 결과 R개를 사용하여 신뢰구간을 구한다.
- 표본을 추출하면서 원래의 데이터셋을 복원하기에 모집단의 분포에 어떤 가정도 필요 없이 표본만으로 추론이 가능

#### Reference

- [샘플링과 리샘플링](https://variety82p.tistory.com/entry/%EC%83%98%ED%94%8C%EB%A7%81%EA%B3%BC-%EB%A6%AC%EC%83%98%ED%94%8C%EB%A7%81?category=996031)
- [DATA - 12. 부트스트랩(Bootstrap)](https://bkshin.tistory.com/entry/DATA-12)
- [샘플링과 리샘플링의 차이는 무엇일까?](https://kejdev.github.io/machinelearning/2021/01/25/sampling-resampling.html)

---

### 확률 모형과 확률 변수는 무엇인가요?

<aside>
💡 <b>확률 변수란 확률로 표현하기 위한 event를 정의하는 것으로 Sample space에서 확률 변수가 취할 수 있는 값에 따라 세 가지로 나뉜다.</b>

</aside>

- 이산확률 변수: 취할 수 있는 값이 정수인 경우
- 연속확률 변수: 취할 수 있는 값이 실수인 경우
- 혼합랜덤 변수: 위의 두 가지 경우가 섞인 경우

확률 분포란 **확률변수의 모든 값과 그에 대응하는 확률들이 어떻게 분포**하고 있는지를 말한다.

이런 변수 성격에 따라 각각이 가지는 확률 함수의 이름도 달라지는데, 이산형은 **확률 질량 함수**라고 하고, 연속형은 **확률 밀도 함수**라고 부른다. 확률함수는 해당 확률 변수가 가진 확률 분포를 나타낸다.

- 확률 함수: 확률 변수를 0과 1 사이의 확률로 mapping하는 것
    - 일반적으로 수학에서 다루는 미지수라 생각하면 됨
    - $P(X=event) = probability$

![표본 공간과 확률의 관계](../img/Math/img1.png)

<aside>
💡 <b>확률 모형이란 확률 변수를 이용하여 데이터 분포를 수학적으로 정의하는 것을 말한다.</b>

</aside>

- 함수에 쓰인 계수들을 모수(parameter)라고 부른다.
- 확률 분포를 안다. → 확률 분포를 나타내는 확률 분포 함수를 안다. → 함수식을 구성하는 모수를 안다.
    - 확률 분포를 알면 확률 변수의 모든 값과 그에 대응하는 확률들을 안다.
    - 모수 추정이 중요한 이유

#### Reference
- [http://www.stat.yale.edu/Courses/1997-98/101/probint.htm](http://www.stat.yale.edu/Courses/1997-98/101/probint.htm)
- [https://ko.wikipedia.org/wiki/확률_변수](https://ko.wikipedia.org/wiki/%ED%99%95%EB%A5%A0_%EB%B3%80%EC%88%98)
- [숨니의 무작정 따라하기 - 확률변수와 확률모형](https://sumniya.tistory.com/24)
- [hanse’s.log - 확률 모형과 확률 변수](https://velog.io/@ohs2251/%ED%99%95%EB%A5%A0-%EB%AA%A8%ED%98%95%EA%B3%BC-%ED%99%95%EB%A5%A0-%EB%B3%80%EC%88%98)
- [Jupyter Community](https://notebook.community/zzsza/Datascience_School/09.%20%EA%B8%B0%EC%B4%88%20%ED%99%95%EB%A5%A0%EB%A1%A02%20-%20%ED%99%95%EB%A5%A0%20%EB%B3%80%EC%88%98/03.%20%ED%99%95%EB%A5%A0%20%EB%AA%A8%ED%98%95%EC%9D%B4%EB%9E%80)

---

### 누적 분포 함수와 확률 밀도 함수는 무엇인가요? 수식과 함께 표현해주세요.

<aside>
💡 <b>누적분포함수란 확률론에서 주어진 확률 분포가 특정 값보다 작거나 같은 확률을 나타내는 함수</b>

</aside>

확률은 사건(event)이라는 표본의 집합에 대해 할당된 숫자이고 어떤 사건에 어느 정도의 확률이 할당되었는지 묘사한 것을 확률 분포라한다. 확률 분포를 묘사할 때, 모든 사건들을 하나씩 제시하는 것은 어려우므로 확률 변수를 이용한다면 묘사 작업이 간편해진다. 사건(event)이 구간(interval)이 되고 이 구간을 지정하는데 시작점과 끝점이라는 두개의 숫자만 있으면 되기 때문이다. 하지만 이를 정의할 때, 숫자 하나만으로 정의가 가능하도록 시작점을 음수 무한대로 통일하여 확률 분포를 서술할 수 있고, 이를 누적 확률 분포(CDF)라고 한다. 

$$
F(a) = P (X \le a) = \begin{cases}
   \sum_{x \le a} p(x) &\text{if } 이산형 \\
   \int_{-\infty}^a p(x)dx &\text{if } 연속형
\end{cases}
$$

누적 분포 함수는 확률 분포를 함수라는 편리한 상태로 바꾸어 주었고, 확률이 어느 사건에 어느 정도 분포되어 있는지 수학적으로 명확하게 표현해 준다. 그러나 누적 분포 함수는 분포의 형상을 직관적으로 이해하기 어렵다는 단점이 있고, 이를 알기 위해서 확률 변수가 나올 수 있는 전체 구간을 아주 작은 폭을 가지는 구간들로 나눈 다음 각 구간의 확률을 살펴보는 것이 편리하다. 그러나 이 과정에서 구간의 폭을 어느 정도로 정의해야 하는지에 대한 추가적인 약속이 필요하고 이러한 단점을 보완하기 위해 상대적인 확률 분포 형태만을 보도록 만들어진 것이 확률 밀도 함수이다. 누적 확률 분포 그래프 x축의 오른쪽으로 이동하면서 크기의 변화를 살펴보면 특정한 구간에서 확률이 배정되지 않는다면 기울기가 0이고 배정된다면 기울기가 0이 아닐 것이다. 따라서 기울기의 크기를 보면 각 위치에 배정된 확률의 상대적인 크기를 알 수 있고, 기울기의 크기를 구하는 수학적 연산이 미분이므로 확률 밀도 함수는 누적 분포 함수의 미분으로 정의한다. 이러한 이유로 확률밀도함수와 누적분포함수는 미적분의 관계를 가지고 있다.(데이터가 연속형일때만 해당)

$$
\int_{-\infty}^{\infty} f(x)dx = 1
$$

$$
P(a\le X \le b) = \int_a^b f(x)dx
$$

확률 밀도 함수는 확률 변수 X가 어떤 값 x를 가질 확률을 나타내는 것이 아니라, x 근처에서의 확률 밀도를 나타내는 것이다. 이때, 확률 밀도 함수는 다음과 같은 특성을 가진다.

1. f(x)는 항상 0보다 크거나 같다.
2. X의 값이 특정한 구간 a와 b 사이에 있을 확률은 확률 밀도 함수 f(x)와 a, b 사이의 면적을 구하여 나타낼 수 있다.
3. 확률 밀도 함수의 전체 면적은 1이다.

확률 밀도 함수는 각 확률 변수의 분포마다 다르게 정의된다. 예를 들어, 정규 분포의 확률 밀도 함수는 다음과 같다.

$$ 
f(x) = {1 \over \sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2 \over 2\sigma^2}
$$

여기서 $\mu$는 평균값이며, $\sigma$는 표준 편차이다. 이 함수는 종 모양으로, 평균값을 중심으로 좌우로 대칭이며, 표준 편차가 작을수록 곡선이 좁아진다.

**Reference**
- [https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/distribution-functions/cumulative-distribution-function.html](https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/distribution-functions/cumulative-distribution-function.html)
- [https://ko.wikipedia.org/wiki/확률_밀도_함수](https://ko.wikipedia.org/wiki/%ED%99%95%EB%A5%A0_%EB%B0%80%EB%8F%84_%ED%95%A8%EC%88%98)
- [https://ko.wikipedia.org/wiki/누적_분포_함수](https://ko.wikipedia.org/wiki/%EB%88%84%EC%A0%81_%EB%B6%84%ED%8F%AC_%ED%95%A8%EC%88%98)
- [notebook community](https://notebook.community/junhwanjang/DataSchool/Lecture/08.%20%EA%B8%B0%EC%B4%88%20%ED%99%95%EB%A5%A0%EB%A1%A0%202%20-%20%ED%99%95%EB%A5%A0%EB%B6%84%ED%8F%AC/4\)%20%EB%88%84%EC%A0%81%20%EB%B6%84%ED%8F%AC%20%ED%95%A8%EC%88%98%EC%99%80%20%ED%99%95%EB%A5%A0%20%EB%B0%80%EB%8F%84%20%ED%95%A8%EC%88%98)
- [[확률/통계] 누적분포함수 (CDF, Cumulative Distribution Function)](https://roytravel.tistory.com/349)
- [확률 분포 함수와 확률 밀도 함수의 의미](https://velog.io/@groovallstar/%ED%99%95%EB%A5%A0-%EB%B6%84%ED%8F%AC-%ED%95%A8%EC%88%98%EC%99%80-%ED%99%95%EB%A5%A0-%EB%B0%80%EB%8F%84-%ED%95%A8%EC%88%98%EC%9D%98-%EC%9D%98%EB%AF%B8)

---

### 조건부 확률은 무엇인가요?

<aside>
💡 <b>Conditional Probability(조건부 확률이란?)</b>   

주어진 사건이 발생했을 때, 다른 한 사건이 일어날 확률$P(B|A)$와 같이 표현하며 사건$A$에 대한 사건$B$의 조건부 확률이라 한다.

</aside>

$$
P(B|A)={P(A \cap B) \over P(A)}
$$

$$
P(A \cap B) = P(B|A)P(A)
$$

이를 활용해 아래와 같이 베이즈 정리를 유도할 수 있다.

<center><img src="../img/Math/img2.png" width="80%" height="80%"></center>

- $D$: 새로 관찰되는 데이터
- $\theta$: 모델에서 계산하고 싶어하는 모수 (가설)
- 사후확률(Posterior): 데이터를 관찰했을 때, 이 가설이 성립할 확률 (데이터 관찰 이후 측정하기 때문에 사후확률)
- 사전확률(Prior): 가설에 대해 사전에 세운 확률 (데이터 관측 이후 사후확률이 사전확률이 된다.)
- 가능도(Likelihood): 현재 주어진 모수 (가정) 에서 이 데이터가 관찰될 가능성
- 증거(Evidence): 데이터 전체의 분포

#### Reference
- [http://www.stat.yale.edu/Courses/1997-98/101/condprob.htm](http://www.stat.yale.edu/Courses/1997-98/101/condprob.htm)
- [BoostDevs](https://github.com/boostcamp-ai-tech-4/ai-tech-interview/blob/main/answers/1-statistics-math.md#5)

---

### 공분산과 상관계수는 무엇일까요? 수식과 함께 표현해주세요.

<aside>
💡 <b>공분산(Convariance)</b>

확률변수의 기댓값과 분산이 확률분포에 대한 정보를 주듯이 두 확률변수 X, Y 사이에서 정의된 공분산은 두 확률변수의 선형관계에 대한 정보를 알려 준다.

**상관계수(Correlation Coefficient)**
공분산의 크기는 두 변수의 측정 단위에 따라 달라지므로 의미를 부여하기에 적절하지 않다. 공분산을 각 변수의 표준편차로 나누면 어떤 단위를 사용하든 같은 값을 얻게 되는데, 이 값을 상관계수라고 한다.

</aside>

**공분산**

두 확률변수 X, Y의 기대값을 각각 $\mu x = E(X), \mu y = E(Y)$라고 하면 공분산은 다음과 같이 $X$의 편차와 $Y$의 편차의 곱으로 정의된다.

- 기본 정의
    
$$
Cov(X,Y) = \sigma_{xy} = E[(X-\mu x)(Y-\mu y)]
$$
    
- 이산 확률인 경우
    
$$
Cov(X,Y) = \sum_{(x,y)\in S} \sum (X - \mu x)(Y - \mu y)f(x,y)
$$
    
- 연속 확률인 경우
    
$$
Cov(X,Y) = \int_{S_2}\int_{S_1} (X - \mu x)(Y - \mu y)f(x,y)dxdy
$$
    

기대값의 성질을 이용하여 정리하면 다음과 같이 표현된다.

$$
Cov(X,Y) = E(XY) - E(X)E(Y)
$$

**공분산의 기본 성질**

- 상수 a에 대하여 $Cov(X, a) = 0$
- $Cov(X, Y) = Cov(Y,X)$
- $Cov(X,X) = V(X) \ge 0$
- $Cov(X_1+X_2,\ Y)=Cov(X_1,Y)+Cov(X_2,Y)$
- $Cov(aX,Y)=aCov(X,Y)$

**그래프 상에서의 공분산**

<center><img src="../img/Math/img3.png" width="90%" height="90%"></center>

- $Cov(X,Y) > 0$ : X가 증가할 때, Y도 증가한다.
- $Cov(X,Y) < 0$ : 가 증가할 때, Y는 감소한다.
- $Cov(X,Y) = 0$ : 아무런 선형관계가 없다.

**상관없음과 독립성**

두 확률변수 X, Y가 독립이면 $E(XY) = E(X)E(Y)$이므로 공분산은 0이 된다. 즉 X와 Y는 상관없다. 그러나 역은 일반적으로 참이 아니다.

**상관계수**

두 확률 변수 X, Y의 상관계수는 $\rho x,y$ 또는 $Corr(X,Y)$로 나타내며 다음과 같이 정의한다

$$
\rho_{x,y} = Corr(X,Y) = {Cov(X,Y)\over \sqrt {V(X)V(Y)}} ,-1\le \rho \le 1
$$

- 상관계수의 성질
    - 상관계수의 절대값은 1을 넘을 수 없다.
    - 확률변수 X, Y가 독립이라면 상관계수는 0이다.
    - X, Y가 선형관계인 Y = aX + b의 관계일 때 a가 양수면 상관계수는 1 a가 음수면 -1이다.

#### Reference
- [https://online.stat.psu.edu/stat414/lesson/18/18.1](https://online.stat.psu.edu/stat414/lesson/18/18.1)
- [https://datascienceschool.net/02 mathematics/07.05 공분산과 상관계수.html](https://datascienceschool.net/02%20mathematics/07.05%20%EA%B3%B5%EB%B6%84%EC%82%B0%EA%B3%BC%20%EC%83%81%EA%B4%80%EA%B3%84%EC%88%98.html)
- [공분산(Covariance)과 상관계수(Correlation)](https://destrudo.tistory.com/15)
- [https://terms.naver.com/entry.naver?docId=3404964&cid=47324&categoryId=47324](https://terms.naver.com/entry.naver?docId=3404964&cid=47324&categoryId=47324)

---

### 신뢰 구간의 정의는 무엇인가요?

<aside>
💡 <b>모집단에서 표본을 추출하여 표본평균을 계산했을 때 정해진 확신의 정도를 가지고 미지의 모수가 속할 것으로 기대되는 구간을 신뢰구간(confidence Interval)이라 한다.

이때 추정한 구간 안에 실제로 모수가 들어있을 확신의 정도를 신뢰수준(신뢰도, confidence level) 또는 신뢰계수 (confidence coefficient)라 한다. 보통 신뢰수준은 백분율로 나타내며, 신뢰계수는 0부터 1 사이의 값으로 나타낸다.</b>

</aside>

1m 물체의 길이를 추정할 때 길이는 유한하다고 말한다면 신뢰수준 100%가 될 수 있지만 이것은 무의미한 추정이므로 적절한 신뢰수준의 신뢰구간을 추정하는 것이 중요하다.

관측 횟수가 n이고 관측 결과의 평균이 M이라면, n번씩 관측할 때마다 얻어지는 평균값 M들의 값은 당연히 참값에 매우 가까워진다. 더욱 가까워진다는 것은 M들의 표준편차가 작다는 뜻이 되는데, 관측 결과 하나하나가 표준편차 σ인 정규분포를 따른다면, n번 관측하여 얻어지는 M의 표준편차는 다음과 같이 된다.

$$
N번\ 관측하여 \ 얻어지는\ M의\ 표준편차\ =\ \sigma / \sqrt{n}
$$

표본 평균의 평균은 모평균과 같으므로 참값이 m미터라면 **중심극한정리**에의해 관측 결과의 평균 M은 평균이 m이고 표준편차가 $\sigma/\sqrt{n}$인 정규분포를 따르게 되고 정규분포 그래프를 통해 다음의 사실을 알 수 있다.

$$
m-\sigma/\sqrt {n}< M < m+\sigma/\sqrt {n}\ 일\ 확률이\ 약\ \ 68 \\% 이다.
$$

위 식을 m을 기준으로 이항 정리하면 아래와 같다

$$
M-\sigma/\sqrt{n}< m < M+\sigma/\sqrt{n}\ 일\ 확률이\ 약\ \ 68 \\% 이다.
$$

이를 통해 참값 m이 관측으로부터 구한 평균값 M 근처에 있을 확률을 표현할 수 있게 된다.

확률 68%를 **신뢰수준**, **신뢰도**라 하며 100%에서 신뢰수준을 뺀 32%를 **유의수준**이라고 한다.

이는 표본을 100번 추출했을 때 100개의 신뢰구간 중에 95개 정도가 모수의 참값을 포함하고 5개 정도는 모수의 참값을 포함하지 않는다는 것을 의미한다.

<center><img src="../img/Math/img4.png" width="70%" height="70%"></center>

**신뢰 구간의 특성**

- 신뢰 구간이 좁을수록 모집단 평균 추정치가 정확해진다.
- 일반적으로 표본 크기가 클수록, 신뢰 구간이 좁아진다.

**중심극한정리**

- 모집단에서 표본 평균을 끝없이 구할 수록 표본 평균의 분포는 정규 분포를 따르게 됨

#### Reference

- [https://angeloyeo.github.io/2021/01/05/confidence_interval.html](https://angeloyeo.github.io/2021/01/05/confidence_interval.html)
- [https://terms.naver.com/entry.naver?docId=6512706&cid=60207&categoryId=60207](https://terms.naver.com/entry.naver?docId=6512706&cid=60207&categoryId=60207)
- [https://terms.naver.com/entry.naver?docId=3569431&cid=58944&categoryId=58970](https://terms.naver.com/entry.naver?docId=3569431&cid=58944&categoryId=58970)

---

### P-value를 모르는 사람에게 설명한다면 어떻게 설명하실 건가요?

<aside>
 
💡 <b>p-value란 얻은 검정 통계량(ex, t-value)보다 크거나 같은 값을 얻을 수 있는 확률 이때, 검정 통계량의 대부분이 귀무가설을 가정하고 얻게 되는 값</b>

</aside>

“모평균이 100이다”라는 귀무가설이 참이라는 가정 하에서, 100개의 데이터를 sampling 할 때 이론적으로 나올 수 있는 평균의 분포에서, 지금 내가 갖고 있는 표본 평균의 값인 95보다 큰 값이 나올 수 있는 확률을 말한다.

위의 예시에서 그럴 확률이 매우 낮다면 귀무가설을 기각할 수 있게 된다.

우연히 발생할 가능성이 매우 희박한 사건이 실제로 발생했을 경우, 그것을 우연이 아니라고 생각하는 경향이 있고, p-value 역시 그와 같은 경향을 따른 것이기 때문이다.

일반적으로 P-value가 5% 보다 작으면 귀무가설을 기각한다.
하지만 이 5% 기준은 전통적으로 사용하는 수치이기 때문에 실험자가 조정 가능하다.

일반적으로 p-value가 너무 낮으면, 그렇게 낮은 확률의 사건이 실제로 일어났다고 생각하기 보다는 귀무가설이 틀렸다고 생각하게 된다. 그래서 귀무가설을 기각하고 대립가설을 채택하게 되는데 p-value는 검정 통계량이 압축적으로 담고 있던 정보를 더 압축했기 때문에 효과의 크기(effect size)가 커지거나 표본의 크기(n)가 커지면 p-value는 작아진다는 것을 명심해야 한다.

p-value가 의미하는 것: 효과의 크기 (effect size, 표본의 크기)
- p-value가 작은 이유는 effect size가 크기 때문인지 표본의 크기가 크기 때문인지 알 수 없음

<img src="../img/Math/img5.png" width="45%" height="45%"> <img src="../img/Math/img6.png" width="45%" height="45%">


#### Reference

- [https://angeloyeo.github.io/2020/03/29/p_value.html](https://angeloyeo.github.io/2020/03/29/p_value.html)
- [https://www.scribbr.com/statistics/p-value/](https://www.scribbr.com/statistics/p-value/)
- [https://adnoctum.tistory.com/332](https://adnoctum.tistory.com/332)
- [https://terms.naver.com/entry.naver?docId=3580638&cid=58944&categoryId=58970](https://terms.naver.com/entry.naver?docId=3580638&cid=58944&categoryId=58970)
- [https://terms.naver.com/entry.naver?docId=6458445&cid=68053&categoryId=68053](https://terms.naver.com/entry.naver?docId=6458445&cid=68053&categoryId=68053)

---

### R square의 의미는 무엇인가요?

<aside>
 
💡 <b>Coefficient of Determination은 회귀 모델에서 독립 변수가 종속변수를 얼마만큼 설명해 주는지 가리키는 지표(높을 수록 잘 설명)</b>

</aside>

#### 결정계수 시각화 예시
<center><img src="../img/Math/img7.png" width="45%" height="45%"></center>

#### 결정계수 계산 방법
$$
R^2 = 1-{Residual\ Sum \ of \ Squares \  (SSR) \over total\ Sum \ of \ squares\ (SST)} = {Explained\ Sum \ of\ Squares \ (SSE) \over total\ Sum \ of \ squares \ (SST)}\\
$$

1. SST : 
    관측값에서 관측값의 평균을 뺀 결과의 총합
    
$$
SST = \sum^n_{i=1}(y_i-\overline y)^2
$$

2. SSE :
    추정값에서 관측값의 평균을 뺀 결과의 총합
    
$$
SSE = \sum^n_{i=1}(\hat y - \overline y_i)^2
$$

3. SSR :
    관측값에서 추정값을 뺀 값의 총합
    
$$
SSR = \sum^n_{i=1}(y_i-\hat y)^2
$$

#### Reference

- [https://m.blog.naver.com/tlrror9496/222055889079](https://m.blog.naver.com/tlrror9496/222055889079)

- [https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html](https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html)
---

### 평균(mean)과 중앙값(median) 중에 어떤 케이스에서 뭐를 써야할까요?

<aside>
 
💡 <b>
1. 자료값의 분포가 대칭적이고 단일봉 형태인 경우 : 평균
2. 자료의 값에 한 개 이상의 극단적인 값이 있는 경우 : 중앙값
3. 자료의 값의 분포가 한 쪽으로 치우친 경우 : 중앙값

</b>

</aside>

#### 평균 종류
1. 산술평균 : 
    이상치에 민감
    
 $$
{ a+b }\over n
$$
    
2. 기하평균 :
    기간 별 상승 하락 계산 시 유용
    
 $$
\sqrt[n]{(ab)}
$$
    
3. 조화평균 :
    F1 score를 구할 때 사용
    
$$
2ab \over {a+b}
$$

#### Reference

- [https://terms.naver.com/entry.naver?docId=3338104&cid=47324&categoryId=47324](https://terms.naver.com/entry.naver?docId=3338104&cid=47324&categoryId=47324)

- [https://lsh-story.tistory.com/76](https://lsh-story.tistory.com/76)
---
