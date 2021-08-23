---
title: "[기계학습] 4. Convolutional Neural Network"
excerpt: "합성곱 신경망"
slug: "4-ml-cnn"
layout: post
category: "ml"
lang: en
tags: ["ml", "coursework"]
date: 2020-11-06T20:28:09-05:00
use_math: true
draft: false
---

PART IV / CNN
ⓒ [라온피플](https://blog.naver.com/laonple), Stanford cs231n

### 1. CNN 개요

#### 1.1. 신경망 ? “CNN(Convolutional Neural Network)”

- 신경망을 통해 학습을 하게 되면, 쉽게 설명하기 어려운 많은 문제들을 해결할 수가 있다. 
- 다만, 영상에 기반한 인식 알고리즘에서 좋은 결과를 얻으려면, 사전에 많은 처리 과정을 필요로 하기 때문에 기존 multi-layered neural network를 바로 적용하는 것은 어려움이 있다. 
- 이번 class에서는 본격적으로 CNN(Convolutional Neural Network)에 대한 study에 들어가기에 앞서 기존 multi-layered neural network가 갖고 있는 문제점에 대하여 살펴볼 예정이다.

#### 1.2. CNN(Convolutional Neural Network)의 역사


- CNN은 1989년 LeCun이 발표한 논문 [Backpropagation applied to handwritten zip code recognition](https://sci-hub.se/10.1162/neco.1989.1.4.541) 에서 처음 소개가 되었고, 필기체 zip code 인식을 위한 프로젝트를 통해 개발이 되었다. LeCun이 처음 소개한 CNN은 필기체 인식에 있어서는 매우 의미 있는 결과가 나왔지만, 범용화에는 아직 미흡하였다. 
- 이후 2003년 Behnke의 논문 Hierarchical Neural Networks for Image Interpretation 을 통해 일반화가 되었으며, Simard의 논문 Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis](http://cognitivemedium.com/assets/rmnist/Simard.pdf)을 통해 단순화되면서 개념 확대의 단초가 마련이 되었다.
- 이후 GP-GPU(General Purpose GPU)를 통해 CNN을 구현할 수 있는 방법이 소개 되었고, DBN(Deep Belief Network) 등 많은 분야에서 활발하게 연구 및 적용이 되고 있다.

#### 1.3. 기존 Multi-layered Neural Network의 문제점

- 이상적인 머신 러닝 시스템이라면, 학습 데이터(training data)만 적절하게 많이 넣어주면, 알아서 분류까지 척척 해주는 것(unsupervised learning)을 기대할 것이다.
- 하지만 현실은 아직 그렇지 못하다. 그 대안이라면, 기존 지식(prior knowledge)를 이용해 네트워크의 구조를 특수한 형태로 변형을 시켜주는 것이다. 그렇다면 2-D 이미지가 갖는 특성을 최대한도로 살릴 수 있는 방법은 무엇일까?
- 아래 그림처럼, multi-layered neural network를 이용해 16x16 크기의 폰트를 갖는 필기체를 인식하는 경우를 살펴보자.

- 필기체 인식을 위해, 위 그림처럼 256개의 입력단과 100개의 hidden-layer 및 26개의 출력단으로 구성이 되면 (이것은 hidden layer가 1개만 있는 비교적 단순한 구조), 이 망에 필요한 가중치(weight)와 바이어스는 총 28,326개가 필요하게 된다. 폰트의 크기가 커지거나 hidden layer가 2단 이상이거나, 대소문자 구별이나 숫자까지 구별을 해야 한다면 파라미터의 개수가 엄청나게 많아지게 된다. 도대체 이것들을 어떻게 처리해야 할까?
- 기존 신경망의 문제점을 살펴보기 위해, 다음의 예를 고려해 보자. 기존 신경망은 아래 그림처럼, 전체 글자에서 단지 2 픽셀값만 달라지거나 2픽셀씩 이동만 하더라도 새로운 학습 데이터로 처리를 해줘야 하는 문제점이 있다.

- 또한 글자의 크기가 달라지거나, 글자가 회전되거나, 글자에 변형(distortion)이 조금만 생기더라도 새로운 학습 데이터를 넣어주지 않으면 좋은 결과를 기대하기 어렵다.

- 결과적으로 기존 multi-layered neural network는 글자의 topology는 고려하지 않고, 말 그대로 raw data에 대해 직접적으로 처리를 하기 때문에 엄청나게 많은 학습 데이터를 필요로 하고, 또한 거기에 따른 학습 시간을 대가로 지불해야 하는 문제점이 있다. 만약 32x32 폰트 크기에 대해 Black/White 패턴에 대해 처리를 한다면 결과적으로 232*32 = 21024개의 패턴이 나올 수 있고, 이것을 Gray-scale에 적용을 한다면 25632*32 = 2561024개의 어마어마한 패턴이 나오기 때문에, 전처리 과정 없이는 불가능하다라는 것을 미루어 짐작할 수 있다. 결과적으로 기존의 fully-connected multi-layered neural network를 사용하면, 3가지 측면에 문제가 발생함을 알 수 있다.
  - 1. 학습 시간(Training time)
  - 2. 망의 크기(Network size)
  - 3. 변수의 개수(Number of free parameters)

- 이번 class에서는 기존 fully-connected multi-layered neural network의 문제점에 대해 살펴봤다. CNN은 앞서 살펴본 것과 같은 문제점을 해결하기 위한 방법으로 제시가 되었으며, 다음 class부터 CNN에 대해서 상세히 살펴 볼 예정이다.

### 2. Why CNN ?

- 지난 class를 통해 MLP(multi-layer perceptron) 또는 multi-layered neural network를 비전(vision)에 적용할 때의 문제점에 대해 알아보았다.
- MLP는 모든 입력이 위치와 상관없이 동일한 수준의 중요도를 갖는다고 보기 때문에 fully-connected neural network를 구성하게 되면 free parameter의 크기가 엄청나게 커지게 되는 문제가 생긴다. 이것에 대한 해결책으로 신경망 연구자들은 visual cortex 부분과 유사한 신경망을 만들고 싶어 했고, 그 결과가 바로 CNN (Convolutional Neural Network) 이다.
- 이번 class에서는 CNN의 본격적인 구조 설명에 앞서 Receptive filed와 convolution에 대하여 살펴볼 예정이다.

#### 2.1. Receptive Field

- 생명과학 대백과사전에서 수용영역(Receptive Field)는 다음과 같이 정의가 된다.

> 정보처리와 관계되는 세포에 대해 응답을 일으키는 자극의 영역. 감각기의 수용 표면에서 감각자극이 단일감각신경에 반응, 즉 충격발생을 일으키는 영역이다. 특히 눈의 망막면 광자극에 관계되는 것을 말하며, 개구리 망막에서 미세조사법을 이용한 연구결과에 의해 명명된 용어이다. 넓이는 자극광 강도에 의존하고 역자극은 가장 민감한 중앙의 작은 부분에 한정되지만, 그 102~103배 강도에서는 2배 면적(1mm)으로 확대된다.

- 즉, 수용영역이란 외부 자극이 전체 영향을 끼치는 것이 아니라 특정 영역에만 영향을 준다는 뜻이다. 
- 마찬가지로 영상에서의 특정 위치에 있는 픽셀들은 그 주변에 있는 일부 픽셀들과만 correlation이 높을 뿐이며,거리가 멀어지면 멀어질수록 그 영향은 감소하게 된다. 이것은 손가락으로 몸의 여러 부분을 찔러 보았을 때 그것을 느낄 수 있는 범위가 제한적이라는 것을 떠올려보면 쉽게 이해가 가능할 것 같다. 그리고 어느 위치를 찌르냐에 따라서 느끼는 영역의 크기가 다르다는 것도 알 수 있다.
- 이와 유사하게 영상을 해석하여 “인식 알고리즘”을 수행하고자 할 경우 영상 전체 영역에 대해 서로 동일한 연관성(중요도)로 처리하는 대신에 특정 범위에 한정에 처리를 한다면 훨씬 효과적일 것이라는 것은 미루어 짐작을 할 수가 있다. 그리고 이것은 영상에만 한정할 것이 아니라 locality를 갖는 모든 신호들에 유사하게 적용할 수 있다는 추론도 쉽게 할 수 있다. 이런 아이디어에 기반하여 출현을 한 것이 바로 CNN이다.

#### 2.2. Convolution 이란?

- CNN에 대한 설명에 앞서 convolution에 대해 간단하게 살펴보자.
- 통상적으로는 대학교 2학년 “신호 및 시스템” 과정에서 처음 convolution을 접하게 되며, 특정 시스템에 입력이 가해졌을 때 시스템의 반응이 어떻게 되는지 해석하기 위한 용도로 사용한다. 영상 처리 분야에서 convolution은 주로 filter 연산에 사용이 되며, 영상으로부터 특정 feature 들을 추출하기 위한 필터를 구현할 때 convolution을 사용한다. 즉 3x3 또는 그 이상의 window 혹은 mask를 영상 전체에 대해서 반복적으로 수행을 하게 되면, 그 mask의 계수(weight) 값들의 따라 적정한 결과를 얻을 수 있다.
- 아래의 예를 보면, 왼쪽의 전체 이미지에서 노란색 부분이 현재 convolution이 일어나고 있는 영역이며, 빨간색 글자는 convolution의 kernel에 해당이 되고, 노란색 영역의 mask에 대해 연산을 수행하면, 결과는 오른쪽처럼 4가 나오며, 노란색 윈도우 영역을 오른쪽으로 1을 이동시켜 다시 결과를 구하면 3이 되며, 계속 이동을 시키면서 연산을 하면 최종적인 결과를 얻을 수 있다.
- 다음 그림은 원영상에 대해, 다양한 3x3 filter 연산을 한 경우이며, 필터의 종류에 따라 각기 다른 특징을 끄집어 낼 수가 있다.
- 다음의 예는 영상을 부드러운 이미지로 만들기 위한 Gaussian filter의 window 크기를 다르게 적용했을 경우, 다른 특징을 얻을 수 있음을 보여주는 예이다.

#### 2.3. CNN의 특징

- CNN에 Neural Network의 앞에 “Convolutional”이 붙은 이유는 convolution의 특성을 살린 신경망 연산을 한다는 뜻이며, CNN은 기존 multi-layered neural network에 비해 아래와 같은 중요한 특징을 갖는다.

##### 2.3.1. Locality (Local Connectivity)

- CNN은 receptive field와 유사하게 local 정보를 활용한다.공간적으로 인접한 신호들에 대한 correlation 관계를 비선형 필터를 적용하여 추출해 낸다. 이런 필터를 여러 개를 적용하면 다양한 local 특징을 추출해 낼 수 있게 된다. Subsampling 과정을 거치면서 영상의 크기를 줄이고 local feature들에 대한 filter 연산을 반복적으로 적용하면 점차 global feature를 얻을 수 있게 된다.

##### 2.3.2 Shared Weights

- Convolution의 개념을 설명할 때 보았던 것처럼, 동일한 계수를 갖는 filter를 전체 영상에 반복적으로 적용함으로 변수의 수를 획기적으로 줄일 수 있다.
- topology 변화에 무관한 항상성(invariance)를 얻을 수 있게 된다. 
- 이번 class에서는 CNN에 대한 본격적인 구조 소개에 앞서, CNN의 기본 개념이 visual cortex에 있는 receptive field 개념에 기반을 하고 있으며, convolution의 개념에 대한 간단한 소개를 하였고, CNN의 특징에 대해 알아보았다. 다음 class부터 CNN에 구조에 대하여 상세히 살펴 볼 예정이다.

### 3. CNN의 구조

- [Part Ⅳ. CNN] 1. CNN 개요 와 2. Why CNN 을 통해 기존 multi-layered neural network의 문제점과 CNN의 기본 개념에 대해서 살펴보았다. Multi-Layered Neural Network에서 기존 필기체 인식과 같은 인식 알고리즘을 수행 할 때, 2차원 영상이 갖는 공간적인 특징을 무시한 채 fully connected network를 사용한다면 free parameter의 개수가 엄청나게 많아지고, 그것으로 인한 overfitting의 문제때문에 원하는 학습 결과를 얻을 가능성이 희박해지는 문제점이 있었는데, CNN을 사용하면, 이런 문제를 효과적으로 극복할 수 있다.
- 이번 class에서는 CNN의 기본 구조에 대해서 살펴볼 예정이다.

#### 3.1. CNN의 구조

- CNN의 과정은 아래 그림과 같이 나타낼 수 있다. CNN의 과정은 크게 보면 다음과 같은 3단계 과정으로 이루어 진다.
  1. 특징을 추출하기 위한 단계
  2. topology 변화에 영향을 받지 않도록 해주는 단계
  3. 분류기 단계
- CNN 처리 과정은 단순하게 분류기로 구성이 된 것이 아니라 특징을 추출하기 단계가 내부에 포함이 되어 있기 때문에, raw image에 대해 직접 operation이 가능하며, 기존 알고리즘과 달리 별도의 전처리(pre-processing) 단계를 필요로 하지 않는다. 특징 추출과 topology invariance를 얻기 위해 filter와 sub-sampling을 거치며, 보통 이 과정을 여러 번을 반복적으로 수행하여 local feature로부터 global feature를 얻어낸다.
- 분류기 단계는 학습을 통해 다양한 경우에 대응할 수 있도록 해주는 것이 목표이며, 기존 신경망과 동일한 구조를 갖는다.
- 지난 Class에서 살펴본 것처럼, 대부분의 영상 인식 알고리즘에서는 특징을 추출하기 위해 filter를 사용한다. 보통 필터는 5x5 혹은 3x3과 같은 작은 영역(receptive field)에 대해 적용을 하며, 필터에 사용에는 계수들의 값에 따라 각각 다른 특징을 얻을 수가 있다. 일반적으로 이 filter의 계수들은 특정 목적에 따라 고정이 되지만, CNN에서 사용하는 filter 혹은 convolutional layer는 학습을 통해 최적의 계수를 결정할 수 있게 하는 점이 다르다. 통상적인 sub-sampling은 보통 고정된 위치에 있는 픽셀을 고르거나, 혹은 sub-sampling 윈도우 안에 있는 픽셀들의 평균을 취한다.
- 가령 아래 그림과 같이 픽셀이 구성이 된다면, P1~P4 중 1개를 매번 지정된 위치에서 고르거나, (P1 + P2 + P3 + P4)/4를 통해 평균을 취한다.
- 하지만, CNN에서의 sub-sampling은 신경 세포와 유사한 방식의 sub-sampling 방식을 취한다. 신경세포학적으로 살펴보면 통상적으로 강한 신호만 전달하고 나머지는 무시하는데, 이와 비슷하게 CNN에서는 max-pooling 방식의 sub-sampling 과정을 거친다. 개념만 간단하게 설명하자면, P1~P4 중 가장 큰 자극만 선택하는 것이다.
- 아래의 예에서 max-pooling을 선택하게 되면, 전체 sub-sampling 윈도우에서 가장 큰 값만 선택하기 때문에 오른쪽-위쪽에 있는 값처럼 결과가 나오고, average-pooling을 선택하면 각 window의 평균을 취하기 때문에 오른쪽-아래쪽에 있는 결과가 나온다. 이동이나 변형 등에 무관한 학습 결과를 보이려면, 좀 더 강하고 global한 특징을 추출해야 하는데, 이를 위해 통상적으로 (convolution + sub-sampling) 과정을 여러 번을 거치게 되면, 좀 더 전체 이미지를 대표할 수 있는 global한 특징을 얻을 수 있게 된다. 이렇게 얻어진 특징을 fully-connected network을 통해 학습을 시키게 되면, 2차원 영상 정보로부터 receptive field와 강한 신호 선택의 특성을 살려, topology 변화에 강인한 인식 능력을 갖게 된다.
- 위 그림은 CNN의 구조를 설명하는 대표적인 그림 중 하나이며, 지금까지 설명한 것들을 종합하여보면, 입력 영상으로부터 convolution(즉, filter)을 통해, feature map을 만들다. 여러 개의 다른 특징을 추출하고 싶다면, 다른 특징을 추출할 수 있도록 convolution kernel의 계수를 설정하면 된다. 

- 다음 과정은 sub-sampling 과정이며, sub-sampling은 통상적 feature map의 크기를 줄여주면서, 이를 통해 topology invariance도 얻을 수가 있게 된다. 보통 1개의 feature map에 대해 1개의 sub-sampling 연산을 수행한다.
- 이렇게 해서 local feature를 얻었다면, 그 local feature에 대해 다시 convolution과 sub-sampling을 수행하며, 이 과정을 통해 좀 더 global feature를 얻을 수 있게 된다. 여러 단의 (convolution + sub-sampling) 과정을 거치면, feature map의 크기가 작아지면서 전체를 대표할 수 있는 강인한 특징들만 남게 된다. 이렇게 얻어진 global한 특징은 fully connected network의 입력으로 연결이 되며, 앞 서 살펴본 신경망의 특징처럼 학습을 통해 최적의 인식 결과를 낼 수 있게 된다.
- 정리하면 CNN은 여러 개의 layer로 구성이 되며, 그 주요 layer는 convolution layer, sub-sampling(pooling) layer, fully-connected layer이다. CNN의 구조가 단순한 fully connected network에 비해 복잡해 보이지만, 그 본질은 거의 유사하며, 신경망 학습에 사용했던 방법을 거의 비슷하게 적용할 수가 있다.
- 이번 class에서는 CNN에 대한 기본적인 구조에 대하여 알아보았다. 다음 class부터 CNN의 세부 layer에 대해 상세히 살펴 볼 예정이다.

### 4. Convolution Layer 1

- [Part Ⅳ. CNN] 1. CNN 개요 ~ 3. CNN의 구조 를 통해 CNN의 기본 개념과 기본 구조에 대해 살펴보았다. 이번 class에서는 핵심 블락인 convolution layer에 대해
  유명한 논문 2편의 구조를 통해 상세하게 살펴보고, 실제 CNN 설계시 고려할 사항에 대해 알아 볼 예정이다.
- 이전 class에서, convolution을 사용하면 영상이 갖는 공간적인 특성(local receptive field)을 최대한 활용하고, 전체 영상에 대해 가중치 및 바이어스를 공유(shared parameter)하여 자유 변수(free parameter)의 수를 줄임으로써 CNN 학습 시간을 줄이고, overfitting의 가능성을 줄일 수 있으며, 영상에서 잘 구별할 수 있는 특징(salient feature)을 얻을 수 있게 해준다는 사실을 알았다. 실제 구현 시에는 어떻게 되는지 다음에서 살펴보자.

#### 4.1. LeCun - Gradient-based learning applied to document recognition

- 아래 그림은 Lecun이 발표한 유명한 논문(4500번 이상 인용) "Gradient-based learning applied to document recognition"에 실린 블락도이다.
- LeCun은 CNN의 개념을 처음으로 만든 사람이며, 이 논문은 한동안 정체기에 빠진 신경망 연구에 돌파구가 되었다.
- LeCun은 이 논문을 통해, CNN이 필기체 인식에 있어 탁월한 효과가 있음을 입증하였으며, 이후 이 논문에 자극을 받은 많은 사람들이 CNN을 사용하여 다양한 분야에서 놀라울 만한 성과를 보여주었다.

##### 4.1.1. Lecun - CNN의 구조

- 이 논문을 보면, 32x32 크기의 입력 영상에 대해, 먼저 5x5 convolution을 적용하여 28x28 크기를 갖는 6개의 feature map을 생성하고, 이것을 다시 sub-sampling(pooling)을 거쳐 6개의 14x14 크기의 중간 영상을 만들어 낸다. 다시 이것에 대해 5x5 크기를 갖는 kernel을 사용해 convolution을 수행하고 총 16개 feature map을 갖는 10x10 영상을 만들어 내고, sub-sampling을 통해 총 16개의 5x5 영상을 얻는다.
- 그 중 120개를 선택하여 FC(Fully connected) NN에 연결하고 최종적으로 10개의 class를 분류할 수 있도록 하였다.

##### 4.1.2. Lecun - CNN 적용 결과

- 아래 그림은 LeCun이 개발한 CNN을 이용하여 실제로 필기체 숫자 인식 루틴을 수행했을 때, 각 단계별로 얻을 수 있는 결과들을 보여주는 것이다.
- C1 ~ F6까지의 각 단계별 중간 영상을 보면, 1단계와 2단계 convolution 과정을 거치면서 어떤 feature map을 얻어 낼 수 있는지 알 수 있으며, convolution 결과로 얻어진 다양한 feature map의 조합을 통해 잡음이 많이 끼어 있는 영상들에 대해서도 탁월한 결과를 내는 것을 볼 수가 있다.
- 위 그림에서 C1과 C3는 각각의 convolution을 거쳐서 얻어진 feature map 영상을 보여준다. C1 단계에서는 6개의 다른 feature map을 얻을 수 있도록 서로 다른 convolution kernel이 적용이 되었고, C3 단계에서는 16개의 다른 feature map을 얻을 수 있는 서로 다른 convolution이 적용 되었다.
- C1과 C2 convolution 단계 및 S2와 S4 pooling 단계를 거치면서, topology 변화에 강인한 특성을 갖게 되었으며, 결과적으로 인식 능력이 크게 개선되었다. 위 그림을 보면, 상단의 비교적 깨끗한 영상뿐만 아니라 하단처럼 상당한 잡음이 있는 경우에 대해서도, 각각의 convolution kernel이 각각 다른 특징을 추출해내는 것을 알 수가 있다.

#### 4.2. Krizhevsky - "ImageNet classification with deep convolution neural network"

- 아래 그림은 3800회 이상 인용이 된 Krizhevsky의 유명한 논문 "ImageNet classification with deep convolution neural network"에 나오는 또 다른 신경망의 구조를 보여준다. 이 논문에서 Krizhevsky는 ImageNet ILVRC-2010 120만개의 영상을 약 1000개의 class로 분류하는데 CNN을 사용하여 압도적인 결과를 얻을 수 있었다.
- 또한 GPU에 적용하여 괄목할만한 성과를 냈고, 관련 소스 코드까지 공개를 하여, 이후 많은 연구자들의 연구 결과에 큰 영향을 끼쳤다.

##### 4.2.1. Krizhevsky - CNN의 구조

- Krizhevsky는 224x224x3 크기를 갖는 color 영상에 대해  총 5단계의 convolution을 적용하였으며, 1단계에는 11x11 크기의 convolution kerney을 사용하고 건너뛰기(stride)4를 적용하여, 최종적으로 55x55 크기를 갖는 96개의 feature map을 얻을 수 있었다.
- 2단계에는 1단계 영상에 대해 normalization과 max-pooling 단계를 거치고, 5x5 크기를 갖는 kerney을 적용하여 총 256개의 feature map을 얻었으며, 3단계에선 3x3 크기를 갖는 kernel을 적용하여 총 384개의 feature map을 얻었으며, 4단계는 max pooling 단계없이 3x3 convolution 을 적용하여 다시 13x13 크기를 갖는 384개의 feature map을 얻었으며, 5단계에서는 비슷하게 256개의 feature map을 얻었다. 이후 과정은 비슷하다.

##### 4.2.2. Krizhevsky - 1 단계 convolutional kernel

- 아래 그림은 Krizhevsky가 1단계 과정에 적용한 96개의 kernel을 보여준다.
- 얼핏 보기에는 어떤 특징을 추출할 수 있는지 한눈에 파악이 어렵겠지만, 바로 이런 kernel을 갖는 여러 단계의 convolution 중첩을 통해 영상을 1000개의 class로 분류할 수 있는 특징을 추출할 수 있게 되었다. 영상 처리(Image processing) 이론을 배우면 convolution(영상으로부터 각각 다른 특징을 추출해 낼 수 있기 때문에 filter라고도 부름)을 통해 영상을 부드럽게 보이게 하거나 에지 등을 추출할 수  있는데, 이 때 대부분 필터의 계수가 고정이 되어 있다.
- CNN에서의 convolution은 이처럼 계수가 고정이 되어 있는 것이 아니라, 학습을 통해 계수 값을 정한다는 점이 다르다. 즉, CNN 알고리즘을 통해 처리하고자 하는 과제에 따라 최종  convolution kernel의 계수가 달라질 수 있음을 뜻하며, 동일 과제일지라도 학습에 사용하는 학습 데이터에 따라서도 달라질 수 있고, 설정한 hyper-parameter의 값에 따라서도 달라질 수 있다는 것을 의미한다. 계수의 값은 기존 신경망과 마찬가지로 gradient에 기반한 back-propagation에 의해 결정이 된다.
- 또 한가지 주목해야 할 점은 Krizhevsky의 논문은 LeCun의 논문과는 좀 구조가 다른 점이다.
- Krizhevsky의 논문에서 위/아래로 구조가 나눠져 있는 것은 2개의 GPU에 적용을 하기 위함이다. 이 논문에서는 stride라는 개념이 적용되었고, convolution을 거치고 max-pooling 단계 없이 곧바로 convolution을 적용하는 등 구조가 상당히 다르다는 것을 알 수 있다. 또한 convolution의 단계도 5단계나 된다.
- 이렇게 구조가 달라지고, 이것들이 무슨 의미를 갖고, 어떻게 정할 것인가에 대해서는 설명이 다소 길어질 수 있어 다음 class에서 상세히 살펴볼 예정이다.
- 이전 class에서는 LeCun과 Krizhevshy의 유명한 논문에 실린 CNN의 구조 비교를 통해, 수행하고자 하는 과제에 따라 convolutional layer의 구조가 달라진다는 것에 대해 알아보았다. 이번 class에서는 convolutional layer 구조를 결정하는 여러 시스템 변수(hyper-parameter)에 대해 살펴볼 예정이다.
- 우리는 [Part Ⅲ. Neural Networks 최적화] 8. Hyperparpameters Optimization 자료를 통하여 신경망의 거시 혹은 시스템 변수인 hyper-parameter에 대해서 학습을 했으며, 이 hyper-parameter를 어떻게 설정하느냐에 따라 학습의 성공 여부가 결정이 난다는 것을 알았다.
- 얼핏 보더라도 CNN은 기본 신경망(MLP)에 비해 복잡해 보이기 때문에 짐작을 했겠지만, CNN에서는 더 많은 hyper-parameter에 대해 신경을 써줘야 한다. 그 중 convolution layer에 관련된 중요한 hyper-parameter가 바로 필터의 개수와 필터의 형태이다.

#### 4.3. Filter의 개수

- Feature map의 크기(즉, convolutional layer의 출력 영상의 크기)는 layer의 depth가 커질수록 작아지기 때문에, 일반적으로 영상의 크기가 큰 입력단 근처에 있는 layer는 filter의 개수가 적고, 입력단에서 멀어질수록 filter의 개수는 증가하는 경향이 있다. 무턱대고 감으로 필터의 개수를 정할 수도 없는데, 과연 어떤 원칙으로 필터의 개수의 정할까?
- 먼저 CNN에서 convolution layer의 연산 시간을 살펴보자. 연산 시간은 대략적으로 아래의 수식으로 결정이 된다.
  $Tc = Np x Nf x Tk$
  Tc: 각 layer에서의 연산시간,
  Np: 출력 pixel의 수, Nf: 전체feature map의 개수, Tk: 각 filter 당 연산 시간

- 필터의 개수를 정할 때 흔히 사용하는 방법은 각 단에서의 연산 시간/량을 비교적 일정하게 유지하여 시스템의 균형을 맞추는 것이다. 이를 위해 일반적으로는 각 layer에서 feature map의 개수와 pixel 수의 곱을 대략적으로 일정하게 유지시킨다. 그 이유는 위 식에서 살펴본 것처럼 convolutional layer에서의 연산 시간이 픽셀의 수와 feature map의 개수에 따라 결정이 되기 때문이다. 보통 pooling layer를 거치면서 2x2 sub-sampling을 하게 되는데, 이렇게 되면 convolutional layer 단을 지날 때마다, pixel의 수가 1/4로 줄어들기 때문에 feature map의 개수는 대략 4배 정도로 증가시키면 될 것이라는 것은 감을 잡을 수가 있을 것이다.
- Feature map의 개수는 가중치와 바이어스와 같은 free parameter의 개수를 결정하며, 학습 샘플의 개수 및 수행하고자 하는 과제의 복잡도에 따라 결정이 된다.

#### 4.4. Filter의 형태

- 필터의 형태는 논문마다 상당히 다르다는 것을 알 수 있으며, 학습에 사용하는 학습 데이터 집합에 따라 적절하게 선택을 한다. 일반적으로 32x32나 28x28과 같은 작은 크기의 입력 영상에 대해서는 5x5 필터를 주로 사용을 하지만, 큰 크기의 자연 영상을 처리할 때나 혹은 1 단계 필터에 11x11이나 15x15와 같은 큰 크기의 kernel을 갖는 필터를 사용하기도 한다.
- LeCun의 논문에서는 5x5로 동일한 크기를 갖는 convolutional kernel을 사용했지만, Krizhevsky의 논문에서는 1단에서는11x11, 2단계에서는 5x5, 3단계 이상은 3x3 크기를 갖는 convolutional kernel을 사용했다. (Part Ⅳ. CNN_4. Convolution Layer 참고)
- 그렇다면, 큰 kernel 크기를 갖는 필터 1개을 사용하는 것이 좋을까? 아니면 작은 크기를 갖는 filter 여러 개를 중첩해서 사용하는 것이 좋을까? 예를 들어 7x7 크기를 갖는 filter 1개를 사용하는 것이 좋을까? 아니면 3x3 필터 3개를 중첩해서 사용하는 것이 좋을까?
- 결론부터 말하자면, 결과는 여러 개의 작은 크기의 필터를 중첩해서 사용하는 것이 좋다. 이는 작은 필터를 여러 개 중첩하면 중간 단계에 있는 non-linearity를 활용하여 원하는 특징을 좀 더 돋보이도록 할 수 있기 때문이다. 뿐만 아니라, 작은 필터를 여러 개 중첩해서 사용하는 것이 연산량도 더 적게 만든다.

#### 4.5. Stride 값

- Stride는 convolution을 수행할 때, 건너 뛸 픽셀의 개수를 결정한다. 쉽게 설명하기 위해, 1차원의 경우를 예를 들면 아래 그림과 같다.
- 위 그림에서는 1차원 convolution 커널이 {1, 0, -1}이고 입력 데이터가 {0, 1, 2, -1, 1, -3, 0}인 경우 왼쪽은 stride 1을 적용하여 출력 {-2, 2, 1, 2, 1}을 얻는 경우이고, 오른쪽은 한 픽셀씩 건너뛰면서 convolution을 수행하여 {-2, 1, 1}을 얻는 경우를 보여준다.
- 위 그림처럼, stride는 건널 뛸 픽셀의 수를 나타내며, 2차원인 영상 데이터에 대해서는 가로 및 세로 방향으로 stride에서 정한 수만큼씩 건너 뛰면서 convolution을 사용한다. 앞서 살펴본 LeCun의 논문은 stride를 1을 사용했지만, Krizhevsky의 논문에서는 1단계 convolution에서는 stride 값을 4를 사용하고, 나머지에서는 1을 사용했다. ([Part Ⅳ. CNN] 4. Convolution Layer 참고)
- Stride는 입력 영상의 크기가 큰 경우, 연산량을 줄이기 위한 목적으로 입력단과 가까운 쪽에만 적용을 한다. 그럼 stride를 1로 정하고, pooling을 적용하는 방식과stride 값을 크게 하는 방식에는 어떤 차이가 있을까? Stride를 1로 하면, 경계가 아닌 모든 입력 영상에 대해 convolution 연산을 수행하고 pooling을 하면서 값을 선택적으로 고를 수가 있지만, stride를 크게 하면 그런 선택의 기회가 사라지게 된다. 그러므로 통상적으로 보았을 때는 stride를 1로 하고, pooling을 통해 적절한 sub-sampling 과정을 거치는 것이 결과가 좋다. 하지만 Krizhevsky의 논문처럼, 큰 영상에 대해 CNN을 적용하는 경우는 연산량을 줄이기 위해 입력 영상을 직접적으로 처리하는 1단계 convolutional layer에서는 stride 값을 1이 아닌 값을 적용하기도 한다.

#### 4.6. Zero-padding 지원 여부

- 아래 그림은 유명한 LeCun의 논문에 나오는 CNN의 구조를 보여준다.
- 이 그림을 보면 32x32 입력 영상에 대해 kernel의 크기가 5x5인 convolution을 적용하면 feature map의 크기가 28x28로 줄어 들었고, 14x14 영상에 대해 다시 5x5 convolution을 수행하면 결과는 10x10으로 줄어드는 것을 볼 수가 있다. 보통 convolution 연산을 하게 되면, 경계 처리문제로 인해 출력 영상인 feature map의 크기가 입력 영상보다 작아지게 된다. zero-padding은 작아지는 것을 피하기 위해 입력의 경계면에 0을 추가하는 것을 말하며, FFT 등에서도 많이 사용이 된다. 만약에 입력 영상의 경계에 0을 2 픽셀씩 추가하여 36x36 크기의 영상을 만들었다면 feature map의 크기는 32x32가 된다. 이렇게 zero-padding을 지원하면 영상 크기를 동일하게 유지할 수 있다.
- 그럼 zero-padding을 하는 이유는 무엇일까?
- 단순하게 영상의 크기를 동일하게 유지하는 장점 이외에도, 경계면의 정보까지 살릴 수가 있어 zero-padding을 지원하지 않는 경우에 비해 zero-padding을 지원하는 것이 좀 더 결과가 좋다. 얻고자 하는 feature-map의 개수, 필터의 크기, stride 및 zero-padding은 CNN의 구조를 결정하는 중요한 hyper-parameter이다. 물론 위 parameter 이외에도 실제로 CNN을 설계할 때는 고려해야 할 요인들이 더 있지만, 이번 class에서는 CNN의 구성 요소 중 주로 convolutional layer의 구조를 결정하는 기본 hyper-parameter 들에 대해서 살펴보았다
