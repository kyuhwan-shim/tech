---
title: "[기계학습] 6. CNN CoreTech"
excerpt: "합성곱 신경망 핵심 기술"
slug: "6-ml-cnn-core"
layout: post
category: "ml"
lang: en
tags: ["ml", "coursework"]
date: 2020-11-12T20:28:09-05:00
use_math: true
draft: false
---
PART VI / CNN Core Technology
ⓒ [라온피플](https://blog.naver.com/laonple), Stanford cs231n

**fly to the moon.**
<!--more-->

# 1. Batch Normalization [1]
- 딥러닝에서 가장 골치 아픈 문제 중 하나는 vanishing/exploding gradient 문제이다. Layer 수가 적은 경우는 그 문제가 심각하지 않지만, layer 수가 많아지면 많아질수록 누적되어 나타나기 때문에 심각하게 된다. 그 이유는 활성함수로 sigmoid나 hyper-tangent와 같은 비선형 포화함수(non-linear saturating function)를 사용하게 되면, 입력의 절대값이 작은 일부 구간을 제외하면 미분값이 0 근처로 가기 때문에 역전파(back-propagation)을 통한 학습이 어려워지거나 느려지게 된다. ([Part V. Best CNN Architecture] 8. ResNet [1] 참고)
- 이 문제에 대한 해결책으로 2011년 ReLU(Rectifier Linear Unit)을 활성함수로 쓰는 방법이 소개되어 문제가 완화되기는 했지만, 이것은 간접적인 회피이지 본질적인 해결책이 아니라서 망이 깊어지면 여전히 문제가 된다. dropout이나 기타 regularization 방법들 역시 본질적인 해결책이 아니기 때문에 여전히 일정 layer 수를 넘어가게 되면 “training”을 성공시킨다는 것을 보장할 수 없게 된다. 그러다가 2015년에 획기적인 방법 두개가 발표가 되는데, 그것은 BN(Batch Normalization)과 Residual Network이다. Residual Network에 대한 설명은 이미 앞에서([Part V. Best CNN Architecture] 8. ResNet [1] ~ 8. ResNet [8] 참고) 충분히 했으나, BN에 대한 설명은 충분하지 못했던 것 같다.
- BN은 2015년 발표된 이래 최근 딥러닝에는 거의 대부분 적용이 되는 추세이며, ResNet에도 BN이 적용이 되었고, 구글의 Inception-V2 이후의 후속 구조에는 모두 BN이 적용이 된다.
Batch Normalization은 구글의 Sergey Ioffe와 Christian Szegedy가 공동으로 발표했으며, 논문이 깔끔하게 잘 되어 있기는 하지만 몇군데는 혼동되는 부분이 있으며, 통계적인 지식을 필요로 하기 때문에 이해에 어려운 부분도 있다. 이번 Class는 “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”를 중심으로 살펴 볼 예정이다.
Internal Covariate Shift
최근 딥러닝에는 대부분 GPU가 사용이 되고 있으며, GPU를 효율적으로 사용할 수 있도록 보통은 32~256 크기를 갖는 mini-batch SGD(stochastic gradient descent) 방법을 많이 사용한다.
SGD 방식이 효율적이기는 하지만, 효과를 거두려면 hyper-parameter의 설정에 신경을 많이 써줘야 하며, 특히 초기값과 학습 진도율(learning rate)은 매우 중요한 요소가 된다. 학습 시 현재 layer의 입력은 모든 이전 layer의 파라미터의 변화에 영향을 받게 되며, 망이 깊어짐에 따라 이전 layer에서의 작은 파라미터 변화가 증폭되어 뒷단에 큰 영향을 끼치게 될 수도 있다.
이처럼, 학습하는 도중에 이전 layer의 파라미터 변화로 인해 현재 layer의 입력의 분포가 바뀌는 현상을 “Covariate Shift”라고 한다. 이것은 예전 TV 오락 프로에서 귀마개를 하고 상대방의 입 모양을 보고 무슨 말인지 알아내는 게임과 비슷하다.
다른 비슷한 비유로는 Quora에서 Debiparad Ghosh가 설명한 것처럼, Covariate shift는 건축에서 하중에 의해 기둥이 휘어지는(buckling)과 비슷하다고 볼 수 있다.
기둥이 휘어지는 것을 막으려면 위 그림에서 c나 d의 경우처럼 휘어짐을 방지하는 수단이 필요하게 되며, 그 수단으로는 batch normalization이나 whitening 기법을 들 수가 있다.
Covariate Shift를 줄이는 방법
Internal Covariate Shift를 줄이는 대표적인 방법 중 하나는 각 layer로 들어가는 입력을 whitening 시키는 것이다. 여기서 whitening을 시킨다는 의미는 입력을 평균 0, 분산 1로 바꿔준다는 것이다. (백색 잡음을 생각하면 좀 더 이해가 쉬울 것 같다)
하지만 단순하게 whitening만을 시킨다면, whitening 과정과 parameter를 계산하기 위한 최적화 과정(backpropagation)과 무관하게 진행이 되기 때문에 특정 파라미터가 계속 커지는 상태로 whitening이 진행이 될 수 있다. 논문에 나오는 것처럼 whitening을 통해 loss(cost function)이 변하지 않게 되면, 최적화 과정을 거치면서 특정 변수(논문의 b)가 계속 커지는 현상이 발생할 수 있다.
그러므로 단순하게 whitening을 통해 평균과 분산을 조정하는 것보다는 좀 더 확실한 방법이 필요하며, 그것이 바로 BN(batch normalization)이다.
BN은 평균과 분산을 조정하는 과정이 별도의 process로 있는 것이 아니라, 신경망 안에 포함이 되어 training 시에 평균과 분산을 조정하는 과정 역시 같이 조절이 된다는 점이 단순 whitening과 구별되는 차이점이다.
Batch Normalization(BN)
BN의 개념은 쉬운 듯 하면서도 어려운 점이 있기 때문에 정확하게 이해를 해줘야 하며, 많은 사람들이 각자 자신의 방법으로 설명하려고 애를 썼다.
Normalization은 원래 training 전체 집합에 대하여 실시를 하는 것이 최고의 효과를 거둘 수 있겠지만, mini-batch SGD 방식을 사용하게 되면, 파라미터의 update가 mini-batch 단위로 일어나기 때문에, mini-batch 단위로 BN을 실시한다. 단 mini-batch 집합의 선정은 가급적이면 correlation이 적어 mini-batch가 전체 집합을 대표하는 것이라고 생각해도 무방하도록 해줘야 한다.
학습 시 BN 방법은 아래와 같다.
여기서 평균과 분산을 구하는 과정은 통계학에서 평균과 분산을 구하는 기본 방식이기 때문에 특별한 설명이 없어도 이해할 수 있다. 평균과 분산을 구하게 되면 입력을 정규화시킨다. 정규화 과정에서 평균을 빼주고 그것을 분산으로 나눠주게 되면    의 분포는 -1 ~ 1의 범위로 된다.
BN이 whitening과 다른 부분은 평균과 분산을 구한 후 정규화 시키고 다시 scale과 shift 연산을 위해 γ와 β가 추가 되었으며, γ와 β의 추가됨으로써 정규화 시켰던 부분을 원래대로 돌리는 identity mapping도 가능하고, 학습을 통해 γ와 β를 정할 수 있기 때문에 단순하게 정규화만을 할 때 보다 훨씬 강력해진다.
BN은 보통 non-linear 활성 함수 앞쪽에 배치가 되며 아래 그림과 같은 형태가 된다.
BN은 신경망에 포함이 되기 때문에 back-propagation을 통해 학습이 가능하며, back-propagation 시에는 아래와 같은 chain rule이 적용이 된다.
Training과 Test 사 차이
BN은 테스트 시와 학습시에 적용하는 방법이 좀 다르다. 학습 시에 각 mini-batch 마다 γ와 β를 구하고 그 값을 저장해 놓는다. Test 시에는 학습 시 mini-batch마다 구했던 γ와 β의 평균을 사용한다는 점이 다르며, 아래 그림처럼 표현이 가능하다.
테스트 시의 유사 코드는 아래와 같다. 유사 코드를 보면 알 수 있듯이 평균은 각 mini-batch에서 구한 평균들의 평균을 사용하고, 분산은 분산의 평균에 m/(m-1)을 곱해주는 점이 다르다. 여기서 m/(m-1)를 곱해주는 이유는 통계학적으로 unbiased variance에는 “Bessel’s correction” 을 통해 보정을 해주는 것이다. 이는 학습 전체 데이터에 대한 분산이 아니라 mini-batch 들의 분산을 통해 전체 분산을 추정할 때 통계학적으로 보정을 위해 베셀의 보정값을 곱해주는 방식으로 추정하기 때문이다. (위키백과 참고)
종합하면, Batch Normalization은 단순하게 평균과 분산을 구하는 것이 아니라, scale(γ)과 shift(β)를 통한 변환을 통해 훨씬 유용하게 되었으며, BN이 신경망의 layer 중간에 위치하게 되어 학습을 통해 γ와 β를 구할 수 있게 되었다. Covariate shift 문제로 인해 망이 깊어질 경우 학습에 많은 어려움이 있었지만, BN 통해 Covariate shift 문제를 줄여줌으로써 학습의 결과도 좋아지고 빨라지게 되었다.
이번 Class에서는 BN의 기본 개념에 대하여 살펴보았다. 다음 Class에서는 BN을 Convolutional Neural Network에 적용할 때의 신경을 써줘야 할 부분과 BN의 성능 등에 대하여 알아볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅵ. CNN 핵심 요소 기술]
1. Batch Normalization [2]
CNN의 핵심 요소 기술 ? Batch Normalization ? Part2
[Part VI. CNN 핵심 요소 기술] 1. Batch Nomalization [1] 을 통해, 요즘 딥러닝의 대세로 자리잡은 BN(Batch Normalization)의 기본 개념에 대해 알아보았다. BN을 통해 딥러닝의 학습 단계에서 고질적인 문제인 vanishing/exploding gradient 문제를 해결하고, 학습 시간을 크게 줄일 수 있을 뿐만 아니라, 학습의 정확도도 개선할 수 있게 되었다.
BN은 DNN의 non-linearity 앞 단에 위치하며, Covariate Shift 문제 해결을 위해 입력들의 분포를 조절하는 역할을 한다. 또한 망 속에 layer로 자리를 잡았기 때문에 back-propagation을 통해 관련 변수(scale, shift)를 최적화 할 수 있다. 입력의 분포를 개선하는 방식은 각 layer로 들어가는 입력 데이터에 대하여 평균을 0, 분산을 1로 정규화를 하고, 거기에 scale(γ)을 곱해주고 shift(β) 변수만큼 더해주는 것이며, scale과 shift 변수는 학습을 통해서 결정이 된다.
[Part VI. CNN 핵심 요소 기술] 1. Batch Nomalization [1] 에서 살펴본 기본 개념은 BN을 fully-connected network에 적용할 때에 대한 것이었으며, convolutional network에 적용하고자 할 때는 convolutional network의 특성을 고려해줘야 한다.
이번 Class에서는 BN을 convolutional network에 적용하는 방법 및 BN의 성능에 대하여 살펴 볼 예정이다.
BN을 convolutional network에 적용하는 방법
BN은 활성 함수의 종류와 상관없이 적용이 가능하다. 활성 함수를 g라고 하고, 가중치를 W, 입력 데이터를 u, 바이어스를 b라고 하면, 다음은 기본 신경망을 표현하는 함수 형태가 된다.
z = g(Wu + b)
non-linearity 함수 g 앞에 BN을 적용하게 되면, 결과적으로 x = Wu + b를 정규화 하는 것이 된다. Wu + b에 BN을 적용하면, 정규화 후 scale과 shift 항을 학습을 통해 결정하는데, b는 shift 항으로 대체할 수 있기 때문에 b는 무시할 수 있으며, 앞 선 식을 다시 정리하면 아래와 같은 형태가 된다.
z = g(BN(Wu ))
여기서 BN transform은 각각의 activation에 x = Wu에 독립적으로 적용하며, 또한 학습을 통해 변수 γ와 β가 쌍으로 결정이 된다.
하지만 convolutional layer에 적용할 때는 조금 달라지는 부분이 있는데, 이는 convolution의 특성을 살리기 위함이다. Convolutional layer에서는 shared weight와 sliding window 방식을 적용하여 출력 feature-map의 모든 픽셀에 대하여 동일한 연산을 수행한다. 마찬가지로 BN을 적용할 때는 mini-batch에 있는 모든 activation 뿐만 아니라 모든 위치까지 함께 고려해줘야 한다.
BN을 convolutional layer에 적용을 할 때 mini-batch의 크기를 m이라고 하고, 출력 feature-map의 크기를 p x q라고 하면, 출력 feature-map의 크기에 해당하는 만큼 sliding window를 움직이면서 연산을 해주기 때문에 mini-batch의 크기가 사실상 m’ = m x p x q로 커진 것으로 간주하고 평균과 분산을 구한다. γ와 β는 fully connected의 경우처럼 activation마다 붙는 것이 아니라, feature-map에 대하여 γ와 β쌍이 학습을 통해 결정이 된다.
MNIST 데이터를 이용한 실험
BN이 과연 효과가 있는지를 확인하기 위해 MNIST dataset을 이용한 실험을 하였다. MNIST dataset은 필기체 숫자 인식을 위한 데이다 집합이며 6만장의 학습 영상과 1만장의 테스트 영상이 있다. Yann LeCun 등이 필기체 인식에 사용을 하였으며, 데이터 집합의 크기가 작아 CNN 알고리즘의 초기 검증에 많이 사용한다. ([Part V. Best CNN Architecture] 2. LeNet 참고) MNIST Data의 예는 다음 그림과 같다.
MNIST 데이터를 검증에 사용한 이유는 fully-connected layer로만 구성된 간단한 신경망을 구성하고 과연 BN을 적용했을 때 효과가 있는지를 확인하기 위함이다.
MNIST 데이터 실험에는 폰트 크기가 28x28이므로 총 784개의 입력을 받아 100개의 뉴런(activation)을 가진 fully-connected hidden layer 3개 및 10개의 출력으로 구성되는 간단한 망을 구현하였다.
이 간단한 망에 대하여 BN을 적용했을 때와 그렇지 않을 경우에 대한 실험을 하였는데 결과는 아래 그림과 같다.
그림(a)을 보면 알 수 있듯이, BN을 적용한 경우에 학습 속도가 훨씬 빠르고 학습 결과도 좋다. 그림(b)와 그림(c)는 마지막 hidden layer에 있는 특정 뉴런(activation) 에서 입력 분포 값을 보여주는 것이다. BN 적용된 경우는 거의 변화가 없이 안정적이지만 BN이 적용되지 않는 경우는 값이 흔들리는 것을 확인할 수 있다.
Google Inception 구조에 ImageNet 데이타를 이용한 실험
BN이 fully connected layer로 구성된 간단한 망뿐만 아니라, 실제로 복잡한 Deep CNN에서도 효과가 있는지 실험을 하였다.
이미 2014년 구글팀은 Inception 구조를 이용하여 ILSVRC 대회에서 우승을 하였다. 그 Inception 구조에 BN을 적용하였을 때 효과적인지를 입증할 수 있다면 BN의 타당성을 입증할 수 있게 된다. (Google Inception 구조는 Class32 ~ Class38 [Part V. Best CNN Architecture] 5. GoogLeNet [1] ~ 5. GoogLeNet [6] 참고)
기존 인셉션 구조 변형
기존 Inception 구조에 그대로 BN을 적용하면 기대했던 효과를 거둘 수 없기 때문에 BN의 특성에 맞게 기존 Inception 구조를 약간 변경시켰다.
학습 진도율(learning rate)의 값을 상향. => 속도 향상
Dropout 제거 => BN을 적용하면 regularization 효과를 얻을 수 있기 때문에 dropout layer를 제거 가능
학습 데이터로부터 mini-batch를 정할 때 좀 더 철저하게 섞기 => correlation이 적을수록 효과적이기 때문에 같은 데이터가 mini-batch에 포함이 되지 않도록 했으며, 이를 통해 약 1% 정도 validation 정밀도 향상됨.
L2 weight regularization 값을 줄이기 => L2 weight를 1/5로 줄였더니 효과적.
학습 진도율을 더 빨리 감소시키기 => BN을 적용하면 학습 속도가 빠르기 때문에 기존 인셉션때보다 학습 진도율을 줄여주는 것을 더 빠르게 적용 (6배).
Local Response Normalization 제거 => LRN은 거의 빼는 것이 대세.
Photometric distortion을 줄이기 => 학습 시간이 빠르기 때문에 인위적인 변경을 통해 학습 데이터를 늘리는 대신에 real data 테스트에 집중. (변경을 전혀 시키지 않는 것이 아니라 변경의 강도를 줄임)
실험에 사용한 다양한 구조
다양한 구조 실험을 통해 어떤 구조가 효과적인지를 확인할 수 있도록 다음과 같은 5개 구조를 실험에 사용하였다.
Inception: 원래 인셉션 구조. 초기 학습 진도율에 0.0015를 사용.
BN-Baseline: 기본 인셉션과 동일하지만 각 nonlinearity앞에 BN을 배치.
BN-x5: BN-Baseline에서 초기 학습 진도율을 5배 키워 0.0075 적용.
BN-x30: BN-x5와 비슷하나 초기 학습 진도율을 30배 키워 0.045 적용.
BN-x5-Sigmoid: BN-x5와 동일하지만, nonlinearity에 ReLU 대신 Sigmoid 사용.
위 5개 결과에 대한 실험 결과는 다음과 같다. 위 결과는 구조 비교를 위해 single model single-crop에 대한 실험 결과이며, 기존 Inception의 경우 31x106 만큼 학습을 해줘야 정확도가 72.2%가 되었다.
기존 Inception 구조에 BN만 추가한 BN-Baseline의 경우, 기존 Inception보다 절반 정도의 학습 과정만을 써도 비슷한 수준의 정확도를 얻을 수 있었다. 학습 진도율을 키운 BN-x5의 경우는 기존 Inception 보다 14배 빠르게 비슷한 정확도에 도달할 수 있었다. 학습 진도율을 더 키운 BN-x30은 비슷한 정확도에 도달하는데 기존 Inception보다 6배 빠른 수준에 그쳤지만 최종 학습의 정확도는 74.8%까지 올라가 정확도가 개선되는 것도 확인하다.
ReLU 대신에 sigmoid 함수를 사용하는 경우에도 BN을 사용하면 학습을 성공시킬 수 있었으며 정확도를 69.8%까지 얻을 수 있었다.
종합하면, 아래 표는 정확도 72.2%까지 도달하는데 걸리는 시간을 표시한 것이며, 각각 얻은 최종 정확도 역시 표시되었다. BN-x5-sigmoid는 최종 정확도 69.8% 도달 시간은 의미가 없다.
모델 결합을 통한 성능 비교 (Ensemble Classification)
Sergey Ioffe가 논문을 발표할 당시만 하더라도 마이크로소프트의 ResNet이 최종 결과를 올리지 못한 모양이다. 논문에서는 Inception에 앞서 살펴본 방식과 같이 BN을 적용하고 multi-crop multi-model에 대한 ensemble을 적용했을 때 결과가 가장 좋은 것처럼 발표했으나, 이는 곧 ResNet 최종 결과가 발표되면서 결과가 뒤집어진다. 이후 다시 Inception-V3를 발표하여 자신들도 성능이 나온다고 주장을 한다. ([Part V. Best CNN Architecture] 8. ResNet [8] 참고).
종합하면, Batch Normalization을 적용하게 되면, Internal Covariate Shift 문제를 해결할 수 있으며, 이를 통해 딥러닝의 고질적인 문제들이 해결이 가능하게 되었다. Vanishing/exploding gradient 문제의 해결책이 생기면서 낮은 학습 진도율이 아니라 높은 학습 진도율을 적용할 수 있게 되어 학습 속도의 향상은 물론 학습의 정확도까지 개선할 수 있게 되었다. 이것이 바로 BN이 딥러닝의 대세로 자리를 잡은 이유이다.
다음 Class에서는 Dropout에 대하여 좀 더 자세하게 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅵ. CNN 핵심 요소 기술]
2. Dropout [1]
CNN의 핵심 요소 기술 ? Dropout ? Part1
[Part VI. CNN 핵심 요소 기술] 1. Batch Nomalization [1] ~ 1. Batch Nomalization [2] 를 통해 Deep Learning의 골칫거리인 vanishing/exploding gradient 문제를 해결하고 학습 속도를 개선하기 위한 BN(Batch Normalization)에 대하여 알아보았다.
머신 러닝에서 학습을 시킬 때 또 다른 문제 중 하나가 바로 Overfitting이다. Overfitting의 기본 개념과 Overfitting 문제를 해소하는 방법은 [Part III. Neural Networks 최적화] 1. Overfitting ~ 4. Dropout 에서 이미 살펴보았기 때문에 여기서는 간단하게만 살펴보고 Dropout을 집중적으로 살펴보기로 한다.
Overfitting의 개념
Overfitting은 아래 그림처럼, 학습 데이터에 지나치게 집중을 함으로써 실제 Test 데이터에서는 결과가 더 나쁘게 나오는 현상을 말한다.
만약에 동일한 점이 주어지고 그 점들을 대표할 수 있는 함수(곡선)를 추정하는 경우를 생각해보자. 만약 가운데가 적절한 추정이라고 한다면, 왼쪽 그림은 지나친 단순화를 통해 에러가 많이 발생하는 경우로 underfitting이라고 한다. 오른쪽은 점들을 너무 정확하게 표현한 나머지 학습 데이터에 대한 정확도는 좋지만 실제 test에서는 에러가 발생할 수 있는 상황이며 overfitting에 해당한다.
Overfitting을 표현하는 적절한 그래프 중 하나는 아래 그림과 같다. 학습 데이터에 대한 학습 결과는 계속 좋아지는데, 테스트 데이터를 이용한 결과는 더 이상 개선이 없는 그런 상황을 Overfitting이라고 이해하면 된다.
Overfitting에 대한 해결책
[Part III. Neural Networks 최적화] 2. Regularization ~ 4. Dropout 에서 살펴본 것처럼, Overfitting에 대한 다양한 해결책이 있다. 가장 즐겨 사용하는 방법으로는 regularization(일반화), 학습 데이터를 늘리는 방법(data augmentation) 방법을 들 수가 있다.
Regularization은 학습 時 지나치게 학습 데이터에 집중하는 것을 피하기 위한 일종의 penalty 를 부과하는 방법을 말하며, 대표적인 방법으로는 L1/L2-regularizaiton 방법을 들 수가 있다([Part III. Neural Networks 최적화] 2. Regularization).
학습 데이터의 양이 적은 경우, 적은 학습 데이터로 인해 일반화의 특성이 떨어지기 때문에 지능적으로 학습 데이터를 늘리는 방식이 많이 적용이 되고 있으며 자세한 내용은 [Part III. Neural Networks 최적화] 3. 지능적 훈련 데이터 를 참고하면 된다.
Dropout 은 2012년에 발표된 일종의 regularization 방법이며, 늦게 발표되었지만 기존 방식들보다 훨씬 효과적인 방법이며, 이후 딥러닝에서는 꼭 써야 되는 것처럼 인식이 되었었다.
하지만 2015년에 발표된 Batch Normalization의 부수적인 효과 중 하나가 regularization 효과이기 때문에 BN 연구자들은 Dropout이 없더라도 성능이 충분히 잘 나온다고 주장을 하고 있어 Dropout을 꼭 적용할지 여부는 각 설계자들의 결정에 달려 있는 것 같다.
Dropout 관련 논문
Dropout은 딥러닝 연구에서 중요한 연구 족적을 많이 남긴 캐나다 토론토 대학교 팀에서 발표한 알고리즘이며, 이들은 유명한 AlexNet에 실제로 Dropout을 적용하여 성능을 개선을 입증하였다.
2012년 ILSVRC에서 AlexNet은 기존 다른 어떤 방식보다 Image Classification 분야에서 압도적인 성능을 보였으며, computer vision 분야에서 다시 CNN 연구를 불붙게 만들었다. AlexNet의 주요한 특징으로는 GPU 사용, ReLU를 활성함수로 사용, Dropout 사용, 효과적인 Data augmentation 기법 적용 등, 요즘 CNN/DNN 분야에서 반드시 들어가는 개념이 모두 적용 되었다.
Dropout이 처음 발표가 될 당시만 하더라도 fully-connected layer에만 적용한 것으로 발표가 되었지만, 이후 2012년, 2013년에 Dropout을 개선한 논문들이 다른 연구팀에 의해 많이 발표가 되었다. 그러자 2014년에 초기 Dropout 개념을 좀 더 이론적으로 정립하고, convolutional layer에도 적용을 한 논문 “Dropout: A Simple Way to Prevent Neural Networks from Overfitting” 이라는 논문을 다시 발표한다.
이번 Class에서는 이 논문을 중심으로 Dropout에 대하여 살펴볼 예정이다.
Dropout
Dropout이란 말 그래도 네트웍의 일부를 생략하는 것이다. 아래 그림처럼 네트웍의 일부를 생략하고 학습을 진행하게 되면, 생략한 네트웍은 학습에 영향을 끼치지 않게 된다.
모델 결합(model combination)을 하게 되면 학습의 성능을 개선할 수 있다 모델 결합이 효과를 얻으려면, 서로 다른 학습 데이터를 이용해서 학습을 하거나, 모델이 서로 다른 구조를 가져야 한다. 하지만, 망이 깊은 경우에 1개의 망을 학습시키는 것도 만만치 않은 일인데 복수개의 망을 학습시키는 것은 매우 힘든 작업이 된다. 또한 어떻게 해서 다양한 모델을 학습시켰을지라도 (학습은 offline으로 시키므로) 다양한 모델을 실행시킬 때 연산 시간을 잡아먹기 때문에 빠른 response time이 요구되는 경우에는 곤란함이 있다.
Dropout은 위 2가지 문제를 해결하기 위해 개발된 것이다. 여러 개의 모델을 만드는 대신에 모델 결합에 의한 투표효과(Voting)와 비슷한 효과를 내기 위해 학습 사이클이 진행되는 동안 무작위로 일부 뉴런을 생략한다. 그렇게 되면 생략되는 뉴런의 조합만큼 지수함수적으로 다양한 모델을 학습시키는 것이나 마찬가지 이기 때문에 모델 결합의 효과를 누릴 수 있다.
또한 실제로 실행을 시킬 때는 생략된 많은 모델을 따로 실행시키는 것이 아니라, 생략된 모델들이 모두 파라미터를 공유하고 있기 때문에 모두 각각의 뉴런들이 존속할 (dropout 하지 않을) 확률을 각각의 가중치에 곱해주는 형태가 된다.
이것을 그림으로 표현하면 아래와 같은 형태가 된다. 학습 시에는 뉴런은 존속할 확률 p로 학습을 진행하고, 실행할 때는 각각의 넷에 얻어진 가중치에 존속할 확률 p를 곱해준다.
Dropout 효과
그런데 Dropout을 하면 왜 regularization 효과를 갖는 것일까? 학습을 시키다 보면, 학습 데이터에 의해 각각의 넷트의 weight들이 서로 동조화 되는 현상(co-adaptation)이 발생할 수 있는데 (이것은 신경망 구조의 본질적 이슈), 무작위로 생략을 하면서 학습을 시킴으로써 이런 동조화 현상을 피할 수 있게 된다.
그리고 이렇게 co-adaptation을 피해서 학습을 하게 되면, 아래 그림에서 볼 수 있는 것처럼 좀 더 선명한 feature를 얻게 된다. 아래 그림은 논문의 저자들이 Dropout의 효과를 확인하기 위해 feature visualization(ZFNet 부분 [Part V. Best CNN Architecture] 4. ZFNet [1] ~ 4. ZFNet [3] 참고)을 통해 확인한 결과이다.
또한 dropout을 하게 되면, hidden 뉴런들의 활성도(activity)가 좀 더 드문드문(sparsity) 해지는 경향이 생긴다. 아래 그림에서 왼쪽 그림은 dropout이 없는 일반 신경망에서 hidden layer의 활성도를 보여주고 있는데 히스토 그램이 좀 더 넓게 퍼져 있는 것을 확인할 수 있으며, 오른쪽은 0.5의 확률로 dropout을 실행하였을 때 히스토그램이 좀 더 집중되는 것을 알 수 있다. 의도하지 않았지만 좋은 성질을 얻게 된 것이다.
이번 Class에서는 Dropout의 기본 개념에 대하여 살펴보았다. 다음 Class에서는 이번 Class에서 살펴보지 못했던 Dropout의 특징, 성능 및 기타 자세한 부분에 대하여 알아볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅵ. CNN 핵심 요소 기술]
2. Dropout [2]
CNN의 핵심 요소 기술 ? Dropout ? Part2
[Part VI. CNN 핵심 요소 기술] 2. Dropout [1]에서는 Dropout의 기본 개념에 대하여 살펴보았다. Dropout 논문의 저자들은 Dropout을 사용하면 학습과정에서 신경망의 weight들이 서로 동조하는(co-adaptation) 하는 것을 막고, 모델 결합(model ensemble)에 의한 투표효과에 의해 신경망의 성능이 더 좋아진다고 주장하였다. 물론, Dropout이 왜 성공적인가를 다른 방식으로 설명하는 논문들도 꽤 있으며, 이것은 다음 Class에서 살펴 볼 예정이다.
이번 Class에서는 지난 Class에서 이어 Dropout에 대하여 상세하게 살펴보고, 논문의 실험 내용을 통해 성능에 어떤 영향을 끼치는지도 살펴볼 예정이다.
Dropout 모델링
Dropout 모델링은 Dropout이 적용되지 않은 표준 네트웍과 적용된 네트웍을 구별해서 생각하면 쉽다.
위 그림에서 왼쪽에 있는 표준 네트웍에 Dropout이 적용되면 오른쪽 그림처럼 모델링 할 수 있다.
먼저 표준 네트웍을 수식으로 표현하면 아래와 같다. 여기서 f(x)는 뉴런의 활성함수 이다. 입력으로 를 받아 망의 가중치   를 곱하고 출력으로    이 나온다.
Dropout을 적용한다는 것은 베르눌리 랜덤 변수인   을 곱해주는 것으로 생각할 수 있으며, 그렇게 되면 관련 수식은 아래와 같이 된다. 여기서 *는 각 항목별로 곱을 해준 것을 의미한다. 입력 y(l)에 Dropout 랜덤 변수 r(l)을 곱해주면 결과적으로 r(l)의 값에 따라 네크웍이 줄어드는 thinned 네트웍   이 되며 여기에 가중치 w(l+1)을 곱해준다.
여기서 베르눌리 랜덤 변수   은 유닛의 존재 유/무 두가지 값을 갖는 랜덤 변수를 말하며, 유닛이 존재할 확률이 p라고 하면, 평균은 p이고 분산은 p(1-p)인 변수를 말한다.
각각의 유닛에 대해 독립적으로(independent) 랜덤 변수를 곱해주기 때문에 n개의 유닛에 대해서는 2n개의 조합이 가능하게 된다. 결과적으로 이것은 실제 테스트 시에는 각각의 가중치가    = pW(l)이 됨을 의미한다. Dropout의 수식 전개에 대한 좀 더 자세한 설명은 “The Dropout Learning Algorithm ? Pierre Baldi & Peter Sadowski” 논문을 참고하면 좋을 것 같다.
Max-norm Regularization과 같이 사용하면 효과 만점
Momentum, annealed learning rates 및 L1/L2 weight decay와 같은 기존 regularization을 Dropout과 같이 사용하면 대부분 좋은 결과가 나온다. 하지만 특히 더 좋은 regularization 방법이 있는데 그것은 max-norm 이다.
Max-norm은 hidden layer로 들어오는 weight들이 특정 상수 c보다 작게  해주는 방법이다. 이 때 상수 c는 조율이 가능한 hyper-parameter이며, validation 데이터 집합을 이용해 결정한다. 그리고 max-norm 방법은 Dropout을 적용하지 않더라도 좋은 regularizer임이 이미 증명이 되었다.
Max-norm과 Dropout을 함께 사용을 하면, learning rate을 큰 값을 사용하는 것이 가능하며, 이것은 학습을 빠른 속도로 진행할 수 있음을 의미한다.
Dropout 성능 실험을 위한 실험 데이터
Dropout이 과연 효과가 있는지 실험을 위하여 다양한 테스트 데이터 및 구조에 대한 실험을 진행하였으며, 모든 경우에 Dropout을 적용한 결과가 더 좋게 나왔다. 다음은 성능평가에 사용한 실험 데이터 들이다.
MNIST: 필기체 숫자 인식용 표준 데이터
TIMIT:  잡음이 없는 상태에서 음성 인식을 위한 표준 데이터
CIFAR-10, CIFAR-100: Krizhevsky 등이 만든 작은 크기의 자연 영상 집합
Street View House Numbers data set(SVHN): 구글의 Street View에 찍힌 집 주소 영상.
ImageNet: 대용량 자연 영상 데이터 베이스
Reuters-RCV1: 로이터 뉴스 기사
Alternative Splicing data set: RNA splicing을 예측하기 위한 RNA feature들
각 데이터 들에 대한 자세한 정보는 아래 테이블과 같다.
이미지 데이터에 대한 실험
MNIST는 28x28 픽셀 크기의 필기체 숫자를 인식 성능을 평가하기 위한 데이터 집합이며, 아래처럼 기존 표준 신경망으로 구현했을 때와 Dropout을 다양한 방법으로 적용했을 때를 비교한 것이다. 여기서 DBM은 Deep Boltzmann Machine을 의미한다.
Dropout이 학습을 진행함에 따라 어떤 결과가 나타나는지 확인을 위해 같은 hyper-parameter에 p까지 고정을 시키고 실험을 하였으며 그 결과는 Dropout을 적용을 시키면 훨씬 성능이 좋고, 학습을 계속 진행시켰을 경우에 꾸준히 성능이 개선되는 것을 확인할 수 있다.
구글의 Street View에 찍힌 집주소 아래 그림과 같이 32x32의 작은 크기이며, CIFAR-10 이나 CIFAR-100의 경우도 category의 숫자만 차이가 있을 뿐이지 모두 32x32의 컬러 영상을 사용한다.
SVHN 데이터에 average pooling을 적용한 Multi-stage Conv Net으로 실험을 하였을 때는 9.06%의 에러율을 보였으며, average pooling 대신에 max-pooling을 적용한 경우에는 3,95%로 에러율이 낮아졌다. 여기에 fully connected layer에 대해서만 Dropout을 적용하면 3.02%였고, 모든 layer에 Dropout을 적용하면 에러율이 2.55%로 아주 우수한 결과가 나왔다. 결과는 아래 표와 같다.
CIFAR-10과 CIFAR-100의 실험 결과 역시 fully connected layer에만 Dropout을 적용해도 성능 개선이 확인되었으며 모든 layer에 Dropout을 적용하면 훨씬 성능이 개선되는 점을 확인할 수 있었으며, 결과는 아래 표와 같다.
ImageNet에 대한 실험 결과는 AlexNet Class ([Part V. Best CNN Architecture] 3. AlexNet [1] ~ 3. AlexNet [3])에서 이미 확인을 하였다.
음성 데이터(TIMIT)에 대한 실험
TIMIT 데이터는 미국식 영어의 대표적인 8개 방언을 사용하는 680명을 대상으로 잡음이 없는 환경에서 녹음을 한 것이다.
실험 결과는 6-layer의 신경망을 이용한 것과 DBN(Deep Belief Network) 경우로 나누어서 진행을 했는데, 아래 표처럼 어느 경우에나 Dropout을 사용하면 결과가 좋아졌다.
기타 실험
로이터의 80만개 이상의 기사를 대상으로 50개의 topic으로 분류하는 문서 분류기를 신경망으로 개발을 하였는데 에러율이 31.05%가 나왔다. 여기에 Dropout을 적용하여 다시 학습을 하였더니 에러율이 29.62%로 떨어졌으며, 이를 통해 text data에 대해서도 Dropout이 효과적임을 알 수 있다.
또한 RNA feature들을 기반으로 alternative splicing이 일어나는 것을 예측하는 경우에도 Dropout을 사용하는 것이 효과적임이 밝혀졌다.
위와 같이 다양한 데이터 및 다양한 구조에 Dropout을 적용하였을 때 모두 성능이 개선 되는 것을 확인할 수 있었으며, Dropout은 확실하게 신경망의 성능을 개선시킨다는 것이 입증되었다.
이번 Class에서는 Dropout의 모델 및 Dropout이 실제로 어느 정도나 효과가 있는지를 논문의 실험 결과를 통하여 살펴보았다. 다음 Class에서는 Dropout의 다른 특징 및 Dropout 논문에 고무된 많은 연구 결과가 나오는데 그것들에 대하여 알아볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅵ. CNN 핵심 요소 기술]
2. Dropout [3]
CNN의 핵심 요소 기술 ? Dropout ? Part3
[Part VI. CNN 핵심 요소 기술] 2. Dropout [2] 를 통해 Dropout을 적용하면, 음성, 영상, 텍스트 등 응용분야에 관계없이 신경망의 성능을 개선시킬 수 있다는 것을 확인하였다. 또한 다양한 regularization 기법들과 연동해서 사용하면 효과적이지만, 특히 Max-norm과 같이 사용했을 때 성능이 크게 개선된다는 것도 알 수 있었다. 아래 표는 MNIST 데이터에 대한 실험 결과이다.
이번 Class에서는 Dropout의 부가 효과에 대하여 살펴보고, Dropout 논문에 자극 받은 많은 연구 결과들 중에서 DropConnect 에 대하여 살펴볼 예정이다.
Dropout 의 부가 효과
앞서 [Part VI. CNN 핵심 요소 기술] 2. Dropout [1] 에서 살펴본 것처럼, Dropout을 사용하면 사용하지 않을 때보다 좀 더 선명한 특징(salient feature)을 끌어낼 수가 있게 된다. 그 이유는 뉴런들을 무작위로 생략시키면서 학습을 시키면 parameter들이 서로 동화(co-adaptation) 되는 것을 막을 수 있어, 좀 더 의미 있는 특징들을 더 추출하는 것으로 해석된다. 즉, 다른 파라미터와 같이 cost(loss) function을 줄여나가다 보면 파라미터의 공조 현상이 일어날 수 있는데, Dropout을 하게 되면 서로 의지하던 것을 스스로 해줘야 하기 때문에 좀 더 의미 있는 feature를 끄집어낼 수 있게 된다.
Dropout을 통해 얻을 수는 또 다른 부가 효과는 hidden layer에 있는 큰 activation을 보이는 뉴런의 수가 줄어들게(sparse) 된다. Dropout을 시키고 hidden layer에 대한 뉴론들의 activation에 대한 히스트그램을 구하면, 0에서 큰 peak가 1개 있고, activation이 큰 값을 보이는 뉴런은 몇 개 없다.
Hyper-parameter p(Dropout 확률 변수)가 성능에 미치는 영향
Dropout에서 중요한 hyper-parameter 변수인 p(엄밀한 의미에서는 존속할 확률 변수)를 변화시키는 실험을 다음 두가지 경우에 대하여 수행하였으며, 이를 통해 p의 변화가 결과에 어떤 영향을 주는지 확인하였다.
Hidden layer에 있는 뉴런의 개수를 고정시키고 p만 변화 시켜가면서 테스트
p값을 변화시키면서 hidden layer에 있는 뉴런의 수를 같이 변화시키면서 테스트 (n을 hidden layer에 있는 뉴런의 개수라고 한다면 np가 일정하게)
위 두가지 상황에 대한 실험 결과는 아래 그림과 같다.
Hidden layer에 있는 뉴런의 수를 고정하고 p를 변화시키면서 실험을 하면, 0.4  p    0.8 범위에서는 p 값이 변하더라도 검사 오차가 거의 일정하게 나오는 것이 확인이 되었다. 0.4 이하에서는 Dropout이 많아 underfitting이 일어나는 것으로 추정이 되며, 0.8이상이면 Dropout이 적기 때문에 overfitting의 효과가 나타나는 것으로 해석된다.
위 그림의 오른쪽은 np를 일정하게 유지하는 경우에 대한 실험인데 p가 낮을 때는 n만 고정시켰을 때보다 오차율이 낮은 것을 확인할 수 있었다. 이는 Dropout을 많이 시키고자 한다면 뉴런의 개수를 키우는 것이 underfitting의 영향을 줄일 수 있다는 (어찌보면 당연한) 것을 확인하였으며, p가 0.6일 때 가장 결과가 좋을 것을 알 수 있었다. 0.6과 0.5는 거의 유사한 값이기 때문에 계산의 편의를 위해 hidden layer에서의 Dropout 확률 p를 0.5로 정하는 것은 효과적이라는 것도 입증한다고 볼 수 있다.
학습 데이터 양에 따른 Dropout 효과
학습 데이터 양에 따른 Dropout 결과를 확인하기 위해 MNIST 데이터 중에서 무작위로 100, 500, 1K, 5K, 10K 및 50K를 골라 network 구조는 동일한 조건에서 실험을 하였으며, 결과는 아래와 같다.
학습 데이터의 양이 극히 적은 (100, 500)의 경우는 Dropout을 적용한 경우에 오히려 결과가 나쁘게 되었지만, 그 외의 경우는 모두 Dropout이 효과적이라는 것을 확인할 수 있다. 데이터의 양을 계속 늘리면 어느 순간을 지나면 Dropout의 효과가 줄어드는 것을 확인할 수 있는데, 이는 데이터의 양이 많아지면서 overfitting 가능성이 줄어들었기 때문으로 해석된다.
DropConnect의 기본 개념 소개
Dropout이 발표되고, 효과적인 방식이라는 것이 받아들여지면서, Dropout 논문에 자극을 받아 Dropout을 변형, 개선 및 수학적으로 증명하는 많은 논문들이 발표되었다. 그 중 주목을 받은 방식 중 하나가 DropConnect 이다.
DropConnect는 2013년 뉴욕대 팀이 발표를 하였으며, 논문 “Regularization of Neural Networks using DropConnect ? Li Wan, Matthew Zeiler, Yann LeCun, Lob Fergus”를 참고 하면 된다. (이름만 봐도 CNN 역사에서 쟁쟁한 인물들)
Dropout과 DropConnect의 차이는 위 그림에서 볼 수 있는 것처럼, Dropout은 뉴런을 생략하는 것이고, DropConnect는 connection을 생략(결과적으로 weight를 생략하는 것이며 노드(뉴런)은 남아 있음)하는 것이다. Dropout과 DropConnect에 대하여 수식으로 차이점을 명쾌하게 설명한 논문 “The Dropout Learning Algorithm ? Pierre Baldi & Peter Sadowski” 도 참고를 하면 좋을 것 같다.
k개의 출력을 갖는 표준 신경망의 출력은 다음과 같은 식으로 표시가 가능하다. 아래 식에서 wij는 각 net에 해당 가중치(weight)에 해당된다.
만약 위 표준 신경망에 Dropout을 적용하여 입력 노드(뉴런)을 생략하게 되면, 출력 변수는 아래와 같이 표현이 가능하다. 아래 수식에서 δj는 노드의 생략에 관련된 베르누이 랜덤 변수이며, 이것은 pj로 표시가 가능하다.
한편 동일 신경망에 DropConnect가 적용이 된다면, 출력 변수는 아래와 같이 표현이 가능하다. 위와 달라진 점이 있다면, 노드를 생략하는 경우는 노드 생략에 관련된 베르눌리 랜덤 변수 δj 대신에connection(net)을 생략하기 위한 δij 가 붙었다는 점이다. 이 점을 제외하면 거의 동일하다는 것을 알 수 있다.
DropConnect의 성능
Dropout과 DropConnect는 생략을 통해 파라미터가 동조(co-adaptation)하는 것을 막는다는 관점에서는 비슷하지만, DropConnect가 Dropout을 더 일반화 시킨 것으로 이해를 하는 것이 좋을 것 같다. Dropout처럼 노드(뉴런)를 생략하면 관련된 connection(weight)가 모두 생략이 되는 것이지만, connection만 생략하게 되면 훨씬 가능한 모델이 많이 나올 수 있다. 앞서 살펴본 수식도 이런 연유로 그렇게 된 것이다.
DropConnect 논문 저자들은 DropConnect 방식의 성능 분석을 위해 사용한 방식은 Dropout 논문 저자들이 했던 것과 동일한 벤치마크 데이터를 이용하는 것이다. 직접적인 비교를 하게 되면 바로 성능 비교가 되기 때문이다. 하지만 기대했던 것처럼, 눈에 띌 수준은 아니지만 DropConnect 방식이 Dropout 방식보다 다소 효과가 있는 것으로 나타났다.
먼저 MNIST 데이터를 이용해 실험을 하였는데 결과는 다음 표와 같다. 사용한 함수에 따라 DropConnect 방식과 Dropout 방식의 성능이 엇갈린 결과가 나왔다.
CIFAR-10에 대한 실험 결과는 아래와 같다. 여기서 활성 함수로 ReLU를 사용하였으며, Dropout이나 DropConnect 모두 사용하지 않을 때보다는 결과가 좋지만, 두 방식의 성능차를 확신할 수준은 아닌 것 같다.
SVHN 데이터를 이용한 실험 결과는 아래와 같다. 이 실험 결과를 보면 Dropout이나 DropConnect를 적용한 결과가 적용하지 않을 때와 비교하여 그리 좋다고 볼 수 없지만, 이는 논문 저자들의 학습 방법이 SVHN 데이터를 효과적으로 학습 시키기 위해 적용한 방식이 탁월했기 때문이라 해석된다. 다른 논문에 나온 실험 결과가 거의 2.8% 수준이고 DropOut 논문에서 2.4%로 밝혔는데 여기서는 model voting을 통해 error율 모두 1%대로 개선을 시켰다.
이론적으로는 DropConnect의 자유도(표현력)가 높아 Dropout보다 좋아야 할 것 같은데, 약간 좋은 수준이라서 논란의 여지는 있는 것 같다. 자세한 내용은 논문을 참고하면 될 것 같다.
DropConnect 논문이나 Dropout 논문에서 아주 Deep Network에 대한 실험을 한 것은 아니지만, Dropout이나 DropConnect 가 효과적이라는 것은 이미 여러 곳에서 입증이 되었다. 그 만큼 신경망 학습에서 overfitting 문제는 크고, 이 문제 해결을 위한 적절한 regularization 방법의 선택이 중요하며, Dropout 혹은 DropConnect 는 아주 효과적인 방식임에 틀림 없다.
이번 Class에서는 Dropout의 hyper-parameter에 대하여 살펴보았고, Dropout과 거의 유사한 개념인 DropConnect 방식도 살펴보았다. 다음 Class에서는 Dropout의 생략 개념을 확대/발전 시킨 “Stochastic pooling”과 “Maxout” 방식에 대하여 살펴 볼 예정이다.

# 3. Stochastic Pooling
CNN의 핵심 요소 기술 ? Stochastic Pooling
[Part VI. CNN 핵심 요소 기술] 2. Dropout [1] ~ 2. Dropout [3] 를 통해, 신경망의 학습 과정에서 파라미터의 동조화 현상으로 인해 overfitting이 발생하고 결과적으로 학습 효율이 떨어지게 되며, 이를 피하기 위한 매우 효과적인 regularization 기법인 Dropout에 대하여 살펴보았고, Dropout을 좀 더 일반화 시킨 DropConnect 방법도 살펴보았다.
Dropout의 효과는 많은 연구 팀에 자극을 주었으며, ZFNet 개발자로 유명한 뉴욕대의 Matthew Zeiler와 Rob Fergus도 그들 중 하나이다. 이들은 “Stochastic Pooling for Regularization of Deep Convolutional Neural Networks”라는 논문을 2013년에 발표하였으며, Dropout이 주로 fully connected layer에 적용하여 효과를 얻었던 것과 달리, pooling layer에 stochastic pooling 방법을 적용하여 큰 효과를 얻었다.
이번 Class에서는 위 논문을 중심으로 “stochastic pooling” 에 대하여 살펴볼 예정이다.
Pooling
Pooling은 CNN에서 매우 중요한 역할을 한다. 보통 CNN에서는 pooling이라고 부르는 sub-sampling 을 이용해 feature-map의 크기를 줄이고, 위치나 이동에 좀 더 강인한 성질을 갖는 특징(feature)을 추출할 수 있게 된다. ([Part IV. CNN(Convolution Neural Networks)] 1. CNN 개요 ~ 2. Why CNN ? 참고)
그동안 pooling 윈도우에서 최고값을 선택하는 max-pooling 방식이나, 윈도우 영역에서 평균 값을 취하는 average pooling 방식이 주로 쓰였다.  하지만 위 pooling 방식을 사용하게 되면 약간의 문제가 있다.
윈도우 영역에서 평균값을 취하는 average pooling 방식을 사용하면, deep CNN에서는 성능이 그리 좋지 못하다. 활성함수로 ReLU를 많이 사용하는데, ReLU 특성에 의해 0이 많이 나오게 되면 평균 연산에 의해 강한 자극이 영향이 줄어드는 효과(down-scale weighting)가 발생한다. 활성 함수로 tanh를 사용하게 되면 결과가 더 나빠질 가능성이 있는데, 그 이유는 강한 양의 자극과 음의 자극에 대한 평균을 취하게 되면 서로 상쇄되는 상황이 발생할 수도 있기 때문이다.
반면에 Max-pooling 방식을 사용하면, 앞서 살펴본 average pooling 같은 문제는 없지만, 학습 데이터에 overfitting 되기 쉽다.
Stochastic pooling은 max-pooling의 장점은 유지하면서 overfitting 문제를 해결하기 위한 방법이며, 다음에서 자세히 살펴볼 예정이다.
Stochastic Pooling (Training 時)
Stochastic pooling은 max-pooling과 average pooling의 문제점을 해결하기 위해 고안이 된 방법이다. 단순하게 최대 activation을 선택하거나 모든 activation의 평균을 구하는 것이 아니라, Dropout과 마찬가지로 확률 p에 따라 적절한 activation을 선택한다.
먼저 확률을 구하기 위해 pooling window에 있는 모든 activation의 합을 구한 후, 그 합으로 각각의 activation을 나눠주는 방식(즉, normalization)으로 확률을 구하며, 그 식은 아래와 같다.
각각의 확률을 구한 후 그 영역을 대표하는 activation은 확률에 따라 선정하게 되며, 식은 아래와 같다.
이렇게 선택을 하게 되면, max-pooling의 효과를 그대로 유지를 하면서, stochastic component를 이용해 overfitting을 피할 수 있게 된다. Max-pooling은 가장 강한 자극만을 선택할 뿐이지만, 경우에 따라서는 최대값이 아니더라도 더 중요한 정보를 갖고 있을 수 있으며, stochastic pooling을 쓰게 되면 이것이 가능해진다.
아래 그림에서 pooling window에는 max-pooling 방식을 사용하면 2.4를 선택해야 하겠지만, stochastic pooling 방식을 사용하면 1.6을 선택할 수도 있다. 만약에 average pooling을 선택한다면, 4/9 = 0.444가 되어 의미 없는 작은 값이 나오게 된다.
Back-propagation을 할 때는 max-pooling 방식과 마찬가지로 선정된 activation만 남기고 나머지는 전부 0으로 한 후 처리한다.
좀더 이해를 돕기 위해, 3x3 pooling 윈도우의 activation이 아래와 같이 된 경우를 고려해보자.
이 데이터를 기반으로 확률을 구하면 아래 그림의 왼쪽과 같으며, 이것은 크기에 따라 sorting을 하면 오른쪽과 같은 결과가 나온다.
만약에 stochastic pooling을 위해 5번을 선택하면, activation은 1.2가 선택이 된다. 위 예에서 activation 값이 1.5인 값이 2개가 있는데, 이것은 1번이나 2번을 선택하면, 나오게 된다. 결과적으로 같은 값을 갖는 것들이 많으면, stochastic pooling을 통해 선택될 확률이 증가하게 된다.
Probabilistic Weighting (Test 時)
학습을 마치고 실제로 적용을 할 때도 stochastic pooling을 사용하면 성능이 떨어지는 경향이 있기 때문에, 실제 Test 시에는 다른 방법을 사용한다. 실제로 적용할 때는 아래 식처럼, 각각의 activation에 확률을 weighting factor로 곱해주고 전체를 더하는 방법을 취한다. 이것을 “probabilistic weighting”이라고 부른다.
이 방법을 사용하면, pooling window에 있는 각각의 activation이 다른 weighting factor를 곱하기 때문에 average pooling과 다르다.
논문의 저자들은 probabilistic weighting을 통해, 마지 dropout이 test 시에 modeling averaging을 통해 성능을 끌어 올리는 것과 비슷한 효과를 얻을 수 있다고 주장한다. 학습을 시킬 때는 확률에 기반한 sampling 방식을 통해 다양한 model을 학습하고, test 시에는 probabilistic weighting을 통한 model averaging을 통해 성능을 올린다는 점은 dropout의 개념과 거의 유사하다.
Pooling window의 크기를 n 이라고 하고, d를 pooling region의 개수라고 했을 때, 가능한 모델의 수는 nd가 되기 때문에 dropout을 수행할 때보다 훨씬 많은 모델 수가 가능하며, 통상적인 pooling window의 크기인 n = 4, 9, 16인 경우에 가능한 모델의 수는 104 ~ 106 정도로 엄청나게 큰 조합이 가능해진다.
실제로 test 시에 stochastic pooling과 probabilistic weighting을 비교하는 실험을 수행했는데 probabilistic weighting을 사용하는 것이 더 좋은 결과를 나타냈다. 실험 결과는 뒤에서 같이 살펴보기로 한다.
Stochastic Pooling의 성능 실험 방법
성능을 확인하기 위한 실험을 dropout과 마찬가지로 SVHN, MNIST, CIFAR-10, CIFAR-100 등에 대하여 실험을 하였다. 단, 데이터를 그대로 사용하는 것이 아니라, 성능 향상과 빠른 학습을 위해 SVHN, CIFAR-10, CIFAR-100은 전체 이미지의 평균을 빼주고 처리를 하였으며, MNIST 데이터는 BW 이미지이기 때문에 평균을 빼주면 color shift가 많이 일어나 local contrast normalization만을 수행하여 처리하였다. 아래 그림에서 윗줄에 있는 그림은 원래 학습 데이터고 아랫줄에 있는 그림은 전처리를 한 것이다.
Stochastic pooling의 성능
먼저 CIFAR-10 데이터에 대하여 average, max, stochastic pooling을 적용하여 각각 실험을 한 결과는 아래 그래프와 같다.
이 그래프를 자세히 보면, 학습 결과는 max-pooling이 가장 좋다, 하지만 test 결과를 보면 max와 average pooling은 overfitting이 일어나는 것이 보인다. 학습 시 training error는 계속 감소를 하지만, test error는 거의 변화가 없는 것이 보인다.
반면에 stochastic pooling은 training error는 max-pooling보다 높게 나오지만, test error는 학습을 계속 시키면 계속 error가 줄어드는 것을 확인할 수 있다. 즉, overfitting 문제에서 max나 average pooling보다 자유롭다는 것을 확인할 수 있다.
실제로 280 epoch 동안 학습을 시키고 결과를 확인하면 아래 표와 같이 stochastic pooling을 사용하면 다른 어떤 경우보다 결과가 좋게 나옴이 수치적으로 보인다.
MNIST 데이터를 이용한 실험에서도 비슷한 결과를 얻었으며, 결과는 아래 표와 같다. Elastic distortion을 사용하면 학습 데이터를 지능적으로 늘리는 data augmentation 기법을 적용한 것이라서 이것과 비교를 하는 것은 공정하지 못하다.
CIFAR-100이나 SVHN에서도 어느 경우나 stochastic pooling의 결과가 가장 좋았으며, 이것으로 stochastic pooling은 확실히 overfitting을 피할 수 있는 pooling 방식이 입증되었다. (자세한 수치는 논문 참고)
Model averaging의 효과
Model averaging의 효과를 입증하기 위해, MNIST 데이터에 대하여 training과 test 시에 각각의 pooling 방법을 섞어 실험을 하였는데 그 결과는 아래 표와 같다.
이 결과를 보면, 굵은 글씨의 결과가 좋다는 것을 확인이 가능하다. 학습 시에 stochastic pooling을 사용하고, 테스트 시에 probabilistic weight을 사용했을 때 15.2%로 아주 좋은 성능이 나왔다.
위 표에서 stochastic-10과 stochastic-100은 실제로 model averaging의 효과를 확인하기 위해 stochastic pooling을 이용한 10개 100개 모델을 테스트 시에 만들어 실험을 한 결과이며 뒤에 붙는 숫자가 클수록 결과가 좋아졌다. 하지만 모델 수가 커질수록 연산 시간이 늘어나는 것을 감수해줘야 한다.
100개 모델에 대하여 test에 stochastic pooling을 적용하고 voting을 한 결과가 15.12%가 나왔으며 이것은 test에 probabilistic weighting을 사용한 결과보다는 약간 좋지만, 다른 뜻으로 해석을 한다면 test 시에 probabilistic weighting을 사용하면 model averaging의 효과도 얻을 수 있고, 연산 시간도 매우 짧게 할 수 있다는 것을 반증하는 것이다.
이번 Class에서는 Dropout 방식이 model averaging을 통해 fully connected layer에서 overfitting을 피하는 효과를 얻었듯이, pooling 시에 model averaging의 효과를 얻을 수 있는 stochastic pooling 방식에 대하여 살펴보았다. Overfitting은 학습에 장애물이기 때문에 적절한 방법을 사용하여 피하는 것이 꼭 필요하다.
다음 Class에서는 Dropout의 효과를 극대화 시키기 위해 새로운 activation 함수를 적용하여 성능을 끌어 올린 “Maxout” 방식에 대하여 살펴 볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅵ. CNN 핵심 요소 기술]
4. Maxout
CNN의 핵심 요소 기술 ? Maxout
[Part VI. CNN 핵심 요소 기술] 3. Stochastic Pooling 에서는 Stochastic pooling에 대하여 살펴보았다.  Stochastic pooling은 단순하게 최대값을 취하는 max-pooling과 평균을 취하는 average pooling의 단점을 model averaging을 통해 극복을 하고, 결과적으로 overfitting 문제를 피할 수 있다는 것을 알았다.
"결과적으로 Stochastic pooling은 Dropout과 맥을 같이 한다"는 것을 지난 class를 통해 확인하였다. Dropout에 영향을 받은 또 다른 방법이 2013년에 발표가 되는데, 그것은 바로 Maxout이다. Maxout은 Dropout의 효과를 극대화시키기 위해 독특한 형태의 활성 함수(activation function)를 고안한 것이라고 볼 수 있는데, 그 성능이 상당히 좋다.
이번 Class에서는 Ian J. Goodfellow(유명한 Yoshua Bengio의 지도학생)가 발표한 “Maxout Networks” 라는 논문을 중심으로, 그리고 Maxout에 대한 독특한 설명으로 이해를 쉽게 한 “From Maxout to Channel-Out: Encoding Information on Sparse Pathways” 논문을 참고로 하여 “Maxout” 에 대하여 살펴볼 예정이다.
Maxout
Maxout을 설명하는 설명하는 수식은 아래와 같다.
위 수식에서 h는 hidden layer를 나타내는 함수이고, 입력으로 들어오는 값들에서 최대값을 취한다. 여기서 W와 b는 학습을 통해서 결정이 되는 파라미터이다.
그런데, 위 수식만으로는 도대체 뭐가 뭔지 감을 잡기가 어렵다. 하지만 그림을 통해 이것을 효과적으로 설명한 web-site가 있어 소개한다.
(http://www.simon-hohberg.de/2015/07/19/maxout.html)
일반적으로 hidden layer는 1개의 layer로 구성이 되는 것과 달리, Maxout hidden layer는 2개의 layer로 구성이 되어 있다. 그 구성은 affine function을 수행하는 녹색 영역과 최대값을 선택하는 파란색 영역으로 구성이 된다. 녹색 영역은 전통적인 hidden layer 처럼 활성함수가 있는 것이 아니라 단순하게 입력 x를 각각의 weight에 곱해서 더하는 형식이기 때문에 affine function이라고 부른다. (녹색 영역에는 non-linear 함수가 없다)
위 그림을 보면 k개의 column이 있는데 이 k개의 column에서 동일 위치에 있는 것들 중에서 최고값을 파란색 영역에서 취하며, 최종적으로 m개의 결과가 나오게 된다.
이 그림은 Maxout을 설명하기 위한 그림 중에서 가장 잘 표현한 것 같다.
Maxout의 의미
이제 Maxout의 수식은 이해를 했는데, 이것이 도대체 어떤 의미를 갖는지 살펴보자.
간단하게 살펴보기 위해 입력이 2개이고 출력은 1개뿐이며, k가 3인Maxout unit을 그려보면 아래 그림과 같다.
만약에 이 3개의 유닛을 사용하여 f(x) = x2을 근사 시킨다고 하면, 아래 그림과 같은 형태가 나올 수 있다.
f(x)를 3개의 직선으로 근사를 시킨다면 위와 같은 모양이 되며, 원래 곡선이 볼록(convex)한 경우는 각각의 직선 중 가장 큰 값을 선택하면, 비교적 비슷한 모양을 얻을 수 있다. 당연히 k값이 클수록 구간이 좀 더 세분화 되면서 원래 곡선과 비슷한 형태를 갖게 된다. 여기서는 직선으로 근사시키는 상황이기 때문에 오차가 많은 것처럼 보이지만, affine 함수가 갖는 다양한 표현력을 고려하면 k 값이 크지 않더라도 convex 함수를 거의 표현할 수 있게 된다. 그런 의미에서 Maxout은 universal approximator라고 생각할 수 있다.
결과적으로 생각하면 Maxout은 affine 함수 부분과 최대값을 선택하는 부분을 이용해 임의의 볼록 함수(convex function)를 piecewise linear approximation을 하는 것이라고 할 수 있다. 아래 그림은 다양한 함수를 표현할 수 있는 것을 보여주는 예이다.
Maxout의 성능
Maxout의 성능 평가를 위해 표준 benchmark 데이터를 이용한 실험을 실시하였다.
먼저 MNIST 데이터에 대한 실험 결과는 아래 표와 같다. 앞선 Class에서 살펴본 Stochastic pooling을 써도 성능 개선이 있지만, Maxout을 적용하면 더 좋은 결과를 얻을 수 있음이 밝혀졌다.
CIFAR-10 데이터에 대한 실험 결과는 아래 표와 같다. 역시 Maxout을 사용하는 경우에 성능이 크게 개선되는 것을 확인할 수 있으며, data augmentation까지 같이 사용 하는 경우에 가장 좋은 결과가 나왔다.
CIFAR-100 데이터에 대한 실험 역시 Maxout을 사용한 경우에 가장 좋은 결과를 얻을 수 있으며, 아래 표와 같다.
SVHN에 대한 실험 결과 역시 Maxout을 적용한 경우에 가장 좋은 결과가 나왔으며, 아래 표와 같다.
Model Averaging
앞에서 Maxout을 사용하면 결과가 좋아지는 것을 확인하였다. Maxout을 사용하면 왜 이렇게 좋은 결과가 나오는 것일까?
그 이유는 Maxout에 Dropout을 적용하면서 생기는 model averaging 효과 때문이다.
1개의 신경망에 Dropout을 하면서 학습을 시키면, 마치 엄청나게 많은 수의 sub-model을 학습시키는 것과 유사한 효과를 얻을 수 있고, 이것을 실제 적용할 때는 각 weight에 dropout 확률 p를 곱해주는 간단한 방식으로 model averaging(ensemble 효과)을 얻을 수 있다는 것이다.
수학적으로는 Softmax 함수를 사용하는 경우 p를 곱해주는 것으로, 그것도 Softmax layer가 1 layer인 경우에 한하여 수학적으로 p를 곱해주는 것이 정확하게 averaging이 된다는 것만 증명이 되었다. 하지만 Softmax는 최종 단에서 분류 확률을 정할 때 사용할 수 있는 활성함수이지 중간 단에 사용할 수 있는 활성 함수는 아니다.
Sigmoid, tanh 및 ReLU와 같은 함수를 활성함수로 사용하여도 Dropout을 적용하면 효과가 있는 것이 확인이 되었다. 이 함수들을 적용할 때도 model averaging을 위해 dropout 확률 p를 곱해주는데 이것은 경험적으로 효과가 있다는 것이 밝혀진 것이지 수학적으로 그런 것은 아니다. (이 부분은 Dropout 논문을 참고하거나, [Part VI. CNN 핵심 요소 기술] 2. Dropout [1] ~ 2. Dropout [3] 참고)
Maxout 논문의 저자는 Dropout 효과에 깊은 인상을 받을 것 같고, 그것을 바탕으로 어떻게 하면, Dropout과 결합을 했을 때 가장 좋은 결과를 얻는 활성함수를 만들 수 있을까 고민을 한 것 같다.
모든 연속된 함수는 적당한 오차 범위 내에서 PWL을 이용해 근사 시킬 수 있으며, 오차를 가장 최소화 할 수 있는 방법을 마련하기 위해 최대값을 선택할 수 있도록 했으며, PWL을 위해 affine function을 구현하다 보니 전통적인 의미에서의 활성 함수에서 벗어나는 이상한 모양의 Network이 나오게 되었다.
결과적으로 sigmoid나 tanh와 같은 함수 대신에 PWL(Piecewise Linear Approximation) 기능을 갖는 활성 함수를 고안해 냈고, 이를 통해 Dropout 효과를 극대화 시킬 수 있었던 것으로 보인다.
Model averaging의 효과를 확인하기 위해, 동일 신경망에 활성함수만 Maxout과 tanh를 적용하여 실험한 결과는 아래와 같다. Maxout을 사용하는 경우가 tanh를 사용하는 것보다 좋은 결과를 얻을 수 있다.
결론적으로 tanh를 사용하는 것보다 Maxout을 사용하는 것이 model averaging의 효과를 극대화 시킬 수 있음을 확인하였다. 또한 ReLU를 사용할 때와 비교하면 training 시에 최적화에 좀 더 유리하다. 아래 결과는 ReLU와 Maxout을 사용하여 layer 수의 증가에 따른 비교 실험 결과이다. Maxout을 사용하면, layer 수가 많아지더라도 완만하게 성능이 나빠지는 것을 확인할 수 있으며, 이것은 overfitting에 있어 ReLU를 사용하는 것보다 Maxout이 훨씬 효과적이라는 의미가 된다.
Qi Wang의 논문에 따르면, Maxout이 ReLU보다 성능이 좋은 점을 좀 더 다른 관점에서 해석을 하였는데, 이것을 “경로(pathway)”의 관점에서 해석을 하였다. Maxout을 사용하면, 여러 개의 경로 중 하나를 선택할 수 있고, gradient가 back-propagation 되는 방향도 그 경로로만 전파가 되며, 이것은 학습 샘플로부터 얻은 정보를 sparse한 방식으로 encoding이 가능하다는 뜻이 된다.
물론 ReLU를 사용할 때도 sparse한 성질이 있기는 하지만, 경로 선택에 있어서의 자유는 제한적이며, 이런 이유로 인해서 ReLU를 사용하는 것이 좀 더 Overfitting될 가능성이 높다고 본다. 물론 학습할 때는 sparse한 경로를 통해 encoding하지만, test할 때는 모든 파라미터를 사용하기 때문에 model averaging의 효과를 얻을 수 있다.
이번 Class에서는 Maxout 방식에 대하여 살펴보았다. Maxout은 Dropout의 효과를 극대화시킬 수 있는 특수한 형태의 활성함수를 사용하는 방식이며, 실제 벤치마크 데이터를 이용한 실험을 통해 그 효과를 입증할 수 있었다.
이번 Class까지 진행을 하면서, CNN에 관련된 기본 이론이나 architecture에 대하여 어느 정도 살펴본 것 같다. CNN을 image classification이나 object detection에 사용할 수도 있지만, 중요한 application 중 하나가 image segmentation이다. 다음 class부터는 image segmentation에 대하여 차근차근 살펴 볼 예정이다.​
