# tech-interview

## AI Math

1. list와 array는 자료구조적으로 어떻게 다른가?

    - 속도는 왜 다를까?

    - hint) 객체를 어떻게 받아들이는지?

2. 컴퓨터는 행을 중시하는가? 열을 중시하는가?

    - numpy와 pandas는 axis를 선택할 수 있는가?

    - 강의에서는 numpy는 행을 우선시 하고, 행렬은 행을 우선시 하고 있는 것으로 생각하자.

3. transformation이라는 것은 무엇인가?

    - hint) vector의 방향성을 고민해보면 좋겠습니다.

    - vector도 방향성이 있을까?

4. 우리는 몇차원의 세계를 살아가고 있는가?

    - hint) 차원의 개념을 다시 한번 복습.

5. L2 -norm은 어떤 원을 말하는가?

    - Machine learnig/ Deep learning에서 많이 쓰임.

    - 어떤 기하학적 성질이 있을까?

6. 차원에 따라 Metric은 다를까?

    - Curse of Dimensionality는 무엇일까?

7. 1강(20:36 ~)에서 Robust 학습이 나오는데 Robust란 무엇일까요?

8. 두 벡터 사이의 거리를 이용하여 각도를 계산하는데, 이 각도를 무엇을 의미를 할까요?

    - 두 벡터의 독립, 종속은 무엇일까요?

9. numpy에서의 transpose와 .T 는 무엇이 다를까?

10. 모든 행렬에서 역행렬이 존재할까?

    - 존재하지 않는다면 왜 존재하지 않을까?

11. 선형회귀분석의 가정조건은 무엇일까?

    - i.i.d 조건이란 무엇일까?

12. 다중회귀분석은 코드를 어떻게 작성을 해야할까?

13. 다중공산성(Multicollinearity)란 무엇일까?

    - 항상 해결책이 있을까?

14. 위로 볼록, 아래로 볼록, 극대 ,극소는 무엇인가?

15. 경사하강법을 쉽게 설명한다면? (대상 : 중학생) / 학습률과 gradient의 방향성에 대해서 서술하시오.

16. 함수라는 것은 유치원생분들에게 설명해보시오.

17. Stochastic gradient descent의 단점은 무엇인가 ? batch_size가 작으면 좋을까? 크면 좋을까?

    - batch_size에 따라서 성능이 달라질까?

    - batch에 따라 목적함수가 바뀐다는 것은 무엇일까?

18. Local Minimum이나 Global Minimum은 무엇인가? 어떻게 구할수 있을까?

19. (5강. P.7 Softmax함수)

    - denumerator = np.exp(vec - np.max(vec, axis=-1, keepdims=True)) 에서  axis=-1은 무엇인가?

20. One-hot vector의 문제점은 무엇인가?

    - 해결 방법에는 무엇이 있을까?

21. 딥러닝에서 왜 ReLU 함수를 많이 사용할까?

22. latent Vector(잠재 벡터)는 무엇일까?

23. Tensor는 무엇일까?

24. Entropy라는 것은 무엇인가? / 불확실성이란 무엇일까?

25. 이산형/연속형 확률변수의 기댓값은 어떻게 되는가?

26. 주사위의 확률값은? 동전의 앞/뒷면의 확률값은? 확신하는가?

27. 확률분포는 무엇일까? 왜 중요할까?

28. 독립사건과 배반사건은 무엇인가? 무엇이 다른가?

29. 사후확률과 사전확률, 가능도(likelihood)란 무엇인가?

30. 첨도(Kurtsis), 왜도(Skewness)로 무엇을 판단할수 있는가?

31. (6강 P.19 몬테카를로 샘플링) Monte Carlo Sampling 이란? 몬테카를로는 이산형이든 연속형이든 상관없이 성립한다.

    - $X^{i} ~ (i.i.d) P(x)$

    - 참고 : [https://studyingrabbit.tistory.com/34](https://studyingrabbit.tistory.com/34)

    - 참고 : [https://codingdojang.com/scode/507?langby=python](https://codingdojang.com/scode/507?langby=python)

32. (7강.통계학 맛보기) 불편(Unbiased) 추정량은 무엇일까?

33. 중심극한정리(Central Limit Theorem,CLT)는 무엇일까?

34. 최대가능도 추정법이란?

35. Likelihood VS Probability의 차이는 무엇일까요?

36. 쿨백-라이블러 발산(KL Divergence)는 무엇인가요?

    - 최대가능도 추정법은 쿨백-라이블러 발산을 왜 최소화할까요?

37. 베이즈정리란 무엇일까?

38. 혼동행렬(Confusion Matrix)란 무엇인가?

39. 인과관계추론(Causality Inference)은 무엇인가? /  인과관계 VS 상관관계

    - 인과관계는 데이터 분포의 변화에 강건한 예측모형을 만들 때 필요합니다.

    - 인과관계만으로는 높은 예측 정확도를 담보하기는 어렵습니다.

    - 인과관계를 알아내기 위해서는 중첩요인에 대한 파악이 필요합니다.

40. Convolution의 Kernel은 현실에서 예시가 무엇일까?

41. 주가도 시계열인데 왜 맞추지 못할까? 그 이유는 무엇일까?

42. 가변적인 데이터는 무엇인가?

43. truncated BPTT의 문제점은 무엇일까?

44. 강화학습이란 무엇인가?

## DL Basic

1. Sigmoid함수와 Softmax함수의 차이는 무엇인가?

2. 머신러닝과 딥러닝의 차이는 무엇인가? 
    
    - 정의는 무엇인가?
    
    - 머신러닝의 분야 / 딥러닝의 분야는 무엇이 있을까?

3. Tensorflow 와 pytorch가 다른 것은? 
    
    - 입력받는 것은 어떻게 다른가? 
    
    - Tensorflow ->pytorch가 구현이 되려면 어떻게 할까? / pytorch->tensorflow는 어떻게 해야할까?

4. 지능이라는 것은 무엇일까?

    - DL 책 : Deep learning / by lan Goodfello,Yoshua Benigo, Aaron Courviile
    
    - DL 강의 : CS231n / NLP : CS224N / graph : CS224W

5. 이상치 처리 방법에는 무엇이 있는가?

6. Taylor expansion이란 무엇인가?

7. 왜 Optimization은 First Order Approximation을 사용을 한는가? Second Order Approximation을 하면 안될까? / 극대,극소는 무엇인가?

8. Support Vector Machine 이란?- Deep Learning ideas that have stood the test of time([https://dennybritz.com/posts/deep-learning-ideas-that-stood-the-test-of-time/](https://dennybritz.com/posts/deep-learning-ideas-that-stood-the-test-of-time/))

9. Autoencoder란? / variational autoencoder 란?

10. GPU는 왜 빠른가 ? TPU는 무엇인가?

11. Self-supervised learning는 무엇일까? - SimCLR

12. affine transformation이란?

13. Linear Regression의 가정조건은 무엇인가? - Linear Neural Networks의 조건은 무엇일까?

14. sklearn의 cross_val_score에서는 score = neg_mean_Square_error를 취하고, mean_square_error를 취하면 우리가 알고 있는 수식과 동일하지 않는다. 그 이유는? - $R^2$(결정계수)도 -1이 벗어나는 이유는 무엇인가? 

    - [https://stackoverflow.com/questions/48244219/is-sklearn-metrics-mean-squared-error-the-larger-the-better-negated](https://stackoverflow.com/questions/48244219/is-sklearn-metrics-mean-squared-error-the-larger-the-better-negated) 
    
    - [https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1](https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1)
    
    - [https://datascience.stackexchange.com/questions/93531/neg-mean-squared-error-in-cross-val-score](https://datascience.stackexchange.com/questions/93531/neg-mean-squared-error-in-cross-val-score)

```
import numpy as np
from sklearn.datasets import load_boston    
from sklearn.linear_model import RidgeCV    
from sklearn.cross_validation import cross_val_score    
boston = load_boston()    
np.mean(cross_val_score(RidgeCV(), boston.data, boston.target, scoring='mean_squared_error'))        -154.53681864311497
```

15. CUDA와 cudnn이란?

16. Pytorch에서 transforms.ToTenser()를 하는 이유는?

17. Num_woekers의 기능은 무엇인가? 정상적으로 잘 작동하는가?

18. Class의 상속개념은 무엇인가 ? Pulic? Private?

19. 가중치 초기화에는 무엇이 있는가? 역할을 무엇인가?

20. Gradient Descent는 무엇인가 ? Gradient Exploding 이란?

21. torch.no_grad() -> gradient를 계산하지 않겠다라는 의미는 무엇인가?

22. Batch_size 와 Epoch는 항상 큰게 좋은것일까?

23. 분류 평가지표중에 trade-off관계가 있는 지표는 무엇인가?

24. 자연어 처리 분야에는 무엇이 있는가? 또한 CV or 추천시스템 분야는 무엇이 있을까?

25. Data Leakage문제는 무엇인가? (ML의 catboost의 논문에서 나오게 됨.) - 참고 : Data Leakage에 대한 개인적인 정리입니다([https://dacon.io/forum/403895](https://dacon.io/forum/403895))

26. Ensemble learning이란 무엇인가?

27. sharp/falt minimizer란? - 논문 : On Large-batch Training for Deep learning : Generalization Gap and Shapr Minima, 2017

28. Window Size라는 것은 무엇을 의미하는가?

29. Random_seed는 무엇인가?

30. Norm Penalty는 무엇인가? / lagrange multiplier는 무엇인가?

31. Filter는 왜 홀수만 사용하는가?

32. CNN의 가정사항은 무엇인가?

33. Bottleneck architecture란?

34. Local response normalization이란?

35. Skip Connection의 장점은 무엇인가?

36. Convolution / Deconvolution의 장단점은 무엇인가?

37. RCNN에서 Negative sampling이란 무엇인가?

38. Bounding Box는 꼭 SXS으로만 grid를 해야될까?

39. Hidden Markov Model이란? / autoregressive model이란?

40. Image Processing에는 어떤 패키지가 유용할까?

41. CNN에서의 Output_size는 어떻게 측정되는가?    - 5X5 filter 2번과 3x3 filter size는 같은가?

42. CNN은 Image에서만 적용이 가능한가?    - 왜 MLP에서는 공간학적 정보가 없어지게 되는가?

43. Pooling layer의 기능은 무엇인가?

44. Dimension reduction이란? Principal Component analysis란? - 고유값(Eigenvalue)와 고유벡터(EigenVector) 란?

45. Network가 깊어지게 되면 단점이 무엇이 있을까?

46.  VGGnet에서 왜 3x3 convolution을 사용하였는가?

47. Skip-Connection의 장점은 무엇인가? - Bottleneck architecture의 단점이 존재할까? - 병목현상이라는 것은 무엇인가?

48. Batch Normalization과 Dropout은 무엇인가? - 어디서 사용하는게 좋을까?

49. Deformable CNN에 대해서 알아보자.

50. Global Average Pooling에 대해서 알아보자.

51. Detection 과 Semantic Segmentation의 차이는 무엇인가? 어떠한 것이 중심일까?

52. Convolutionalization을 하는 이유는 무엇인가?

53. Autoencoder / variational Autoencoder란?

54. Convolutional neural network는 항상 이미지에서만 사용한 것 인가? - 마찬가지로 Recurrent neural network도 항상 언어에서만 사용이 가능한가?

55. Markov Process는 무엇인가?

56. Vector의 유사도를 어떻게 측정은 하는가?

57. CNN와 RNN에서의 각각 activation 함수로 무엇을 많이 쓰는가? 그 이유는?

58. long Short Term Memory의 장점은 무엇인가?(+gate recurrent unit)

59. Attention과 Transformer에 대해 설명해주세요.

60. Transformer에서 K, Q, V 각각은 어떻게 만들어지나요?

61. Attension, Transformer 학습 방식을 응용한 모델에는 무엇이 있나요?

62. attention functions으로는 additive attention과 dot-product attention이 있는데 각각은 무엇인가? 논문에서는 어떠한 것을 제시를 하였고 왜 그랬는가?

63. postion encoding이란? - offset은 무엇인가?

64. Transformer의 한계는 무엇인가?

65. Transformer는 input order에 왜 independent하게 각 단어에 encoding이 어떻게 되는가?

66. 독립 항등 분포 (iid, independent and identically distributed)은 무엇인가?

67. 배반사건과 독립사건의 차이는 무엇인가?  예제는 무엇이 있는가?

68. 두 확률분포간의 거리를 측정하는 방법에는 무엇이 있는가?

69. Monte Carlo Method란?

70. Empirical Risk Minimization 란?

71. P-value, 왜도(skew), 첨도(kurtosis)의 정의는무엇인가?

72. VAE,GAN, Diffusion model의 장/단점은 무엇일까?
