---
title: "[기계학습] 8. GAN"
excerpt: "Yann Lecun told us; the most interesting idea in the last 10 years in Machine Learning"
slug: "8-ml-GAN"
layout: post
category: "ml"
lang: en
tags: ["ml", "coursework"]
date: 2020-11-18T20:28:09-05:00
use_math: true
draft: false
---

## 8. GAN

### 1. Introductions to the GAN

#### differences with other networks

그간 이 블로그를 통하여 주로 살펴본 망은 classification이나 segmentation 처럼 대상을 판별하는 판별망(Discriminative Network) 이었다. 

- 즉, 아래 그림과 같이 고양이 그림을 보여주면 몇% 확률로 고양이라고 판별을 해내는 것과 같은 일을 수행해내는 망이었다.
- 이러한 판별망 혹은 판별 모델은 수학적인 관점에서 보면, 입력 데이터 $x$​에 대하여 레이블(label) $y$가 될 확률, 즉 “조건부 확률 $P(y|x)$​”을 구하는 것이 된다.

- 반면에 생성 모델(Generative Model)은 특정한 확률분포 $P_{data} (x)$를 갖는 학습 데이터를 이용해 학습을 시키면 이것과 유사한 분포 $P_{model}(x)$​​​ 를 갖는 데이터 혹은 샘플을 생성할 수 있는 모델을 말한다. 학습 데이터를 이용해 학습만 시켜주면, 거기에 특별한 레이블을 붙여주지 않더라도 스스로 데이터 속에 숨어 있는 중요한 정보(representation)를 찾아내고, 새로운 데이터를 만들어낼 수 있는 것은 상상만으로도 멋진 일이지만, 사용자들의 기대 수준을 만족시킬 수 있을 정도의 데이터를 생성한다는 것이 그렇게 쉬운 일이 아니었으며, 그간에도 많은 생성 모델들이 발표가 되어왔지만 그리 만족스럽지는 못하였다.

#### origin of GAN

- [Generative Adversarial Network](https://arxiv.org/abs/1406.2661) by Ian Goodfellow et al, 2014
- 그러다가 2014년 Ian Goodfellow가 GAN 생성모델을 발표하였는데, 기존에 발표된 어떤 모델보다도 연산 시간이 빠르면서도 꽤 준수한 성능을 얻을 수 있어, 그 이후 GAN에 근거한 엄청나게 많은 논문 및 분석 자료들이 발표되었다. 특히 Facebook AI 연구를 총괄하는 Yann Lecun 교수는 지난 10년 이래 deep learning 관련 연구 중에 최고라는 찬사를 보내기도 하였다.
- 비록 기존에 발표된 생성망들에 비하여 GAN의 성능이 뛰어나기는 하지만, 안정성이 떨어지기 때문에 학습이 제대로 되지 않거나 많은 다양한 분포에서 특정 분포만 갖는 결과를 내는(Mode collapsing) 문제점이 계속 제기되어 왔다. 그러나 GAN은 여러 설계자들을 자극하고 동기를 부여하기에 충분하였으며, 2014이후 꽤 괜찮은 논문들이 연속적으로 발표가 되면서 점점 문제점들이 개선되어 가고 있으며, 앞으로 얼마다 더 획기적인 연구 결과가 발표될 것인지 기대가 된다.
- GAN은 검토해볼 부분이 많고, 논문들도 많이 발표가 되었기 때문에 여러 class에 걸쳐 차근차근 살펴볼 예정이며, 이번 class에서는 가급적이면 수식보다는 개념을 위주로 설명을 진행할 예정이다.

### 2. 주요 개념

#### Naming as 'Generative Adversarial' Network

- 아래 그림과 같이 생성망(`Generator`)과 판별망(`Discriminator`)을 경쟁적으로 학습을 시켜 진짜인지 혹은 가짜인지 구별하기 힘든 수준까지 발전시키는 것이다. 망의 이름에 Adversarial(대립적인, 적대적인)이라는 단어가 들어가는 그런 이유에서이다.
- 블락도에서 `Discriminator`가 하는 일은 실제 학습 데이터(Real world images)와 `Generator`를 거쳐 만들어진 가짜 데이터를 이용하여 학습을 하며, **실제 샘플이 진짜인지 혹은 가짜인지 구별하는 역할**을 한다. 
- `Generator`는 `Discriminator`를 속일 수 있을 정도로 **진짜와 구별이 불가능할 수준의 가짜 데이터를 만들어내는 것**이 목표이다. 
  - 그래서 `Discriminator`를 판별을 잘 하는 방향으로, `Generator`는 `Discriminator`를 잘 속이는 방향으로 계속 학습을 하다 보면, 최종적으로는 `Generator`는 진짜인지 가짜인지 거의 구별이 불가능한 수준의 데이터를 만들어내고, `Discriminator` 역시 분별 능력이 점차 개선이 되게 되는 것이 목표이다.
    < 그림 출처: 스탠퍼드 Fei-Fei 교수 presentation >

##### Metaphor at [original paper](https://arxiv.org/abs/1406.2661)

- 위조지폐를 만들어내는 사기꾼과 위조지폐 여부를 판별하는 형사에 대한 비유가 매우 적절한 것 같다.
- 위조 사기꾼은 형사를 속이기 위하여 점차 정밀한 위조 기술을 발전시키고, 형사는 그런 위조범들의 기술에 대항하여 점차 식별기술을 발전시킨다. 시간이 지날수록 위조범들은 더욱 위조 기술을 발전시켜 진짜인지 가짜인지 모를 수준까지 발전을 시키고, 형사도 점점 교묘해지는 위조 지폐를 찾아내기 위해 감별기술을 발전시킬 수 있게 된다.

#### Learn with GAN

- 기존에 우리가 살펴본 대부분의 판별망에서는 대부분 cost function 혹은 loss function을 정의하고, 오차를 역전파(back-propagation) 시키면서 변수를 갱신하는 방식을 사용하였다. 특정 목적을 가진 망 1개를 학습시키는 것이기 때문에 지향점이 비교적 명확하다.
- 반면에 생성망과 판별망으로 이루어진 GAN은 기존 방법으로는 학습이 가능하지 않으며, 적절한 학습 방법이 필요하다는 것은 예상할 수 있을 것이다.
- 크게 보면, GAN에 대한 학습은 2단계로 이루어진다.
  - [첫번째 단계] 아래 그림과 같이, `Generator`는 고정을 시키고 **`Discriminator`를 학습**을 시킨다. `Discriminator`는 어느 것이 진짜이고 어느 것이 가짜인지 이미 알고 있기 때문에 기존에 보았던 판별망의 학습 방법과 동일하다.
  - [두번째 단계] `Discriminator`는 고정을 시키고 **`Generator`를 학습**을 시키는 것이다. `Generator`의 목적은 `Discriminator`를 속이는 것이기 때문에 `Discriminator`가 진짜로 착각할 수 있는 방향으로 학습을 한다.
  - 그리고 위 과정을 반복적으로 수행을 하다 보면, `Discriminator`와 `Generator`가 발전을 거듭하여 평형상태에 도달하게 된다.
- Ian GoodFellow의 논문에서는 개념적인 설명을 위하여 아래 그림을 사용하였다. 
  - 이 그림에서 검은 점선은 학습 데이터의 분포 $P_{data}(x)$​를 나타내며, 녹색 실선은 `Generator`에서 생성되는 데이터의 분포, 즉 모델 분포 $P_{model}(x)$​를 나타내며, 파란색 점선은 `Discriminator`의 출력을 나타낸다. 
  - (a)를 보면 아직은 실제 데이터 분포와 모델 분포가 다르기 때문에 `Discriminator`의 출력이 흔들리기도 하고 부분적으로만 정확하다. 하지만 점차 학습을 진행하면서 모델분포와 학습 데이터의 분포가 근접하게 되며, (d)에서는 정확하게 일치하면서 `Discriminator`가 최적의 평형 상태인 1/2에 도달하게 된다.

#### GAN 모델 학습이 잘될까?

- GAN의 학습 과정은 기본적으로 상호 협조적이지 않는 상대가 서로 최적의 상태에 도달하려고 노력하다 보면, 내쉬 평형(Nash Equilibrium) 상태에 이르게 된다는 것이며, 그래서 논문에서는 2 player mini-max 게임에 비유하였다.
  - Mini-max 게임:  [Minimax algorithm와 Alpha-beta pruning](http://nosyu.pe.kr/2172) 글을 참고하면 좋을 것 같다.
  - Nash equilibrium : 흔히 죄수의 딜레마(Prisoner’s dilemma) 비유와 함께 많이 사용이 되며, [What is the Nash equilibrium and why does it matter? @ The Economist](https://www.economist.com/the-economist-explains/2016/09/06/what-is-the-nash-equilibrium-and-why-does-it-matter) 를 참고하면 좋을 것 같다.
- 내쉬 평형(위 그림 참고)에 도달하려면, `Discriminator`와 `Generator` 사이에서 서로의 학습을 견인해줘야 하는데 그렇지 못한 경우는 최적에 학습 결과에 도달하기가 어려워지며, 안정성도 떨어지게 된다. 기존에 살펴보았던 망들은 단순하게 `Discriminator`의 cost function을 최소화하는 방향으로만 학습을 하면 되지만 (사실 이것도 아주 쉬운 것은 아님), GAN에서는 `Generator`까지 같이 고려를 해줘야 하기 더 어렵다는 것을 충분히 예상할 수 있다. 밑에서 자세히 수식으로 보자.
- 뿐만 아니라, 예를 들어 이미지를 생성한다면, 무엇을 기준으로 optimal point에 도달했는지 명확하게 판단할 근거도 부족하기 때문에, 아직은 사람의 개입(manual)이 필요하다는 점도 좀 골치 아픈 일이다.
- 하지만 그럼에도 불구하고 꽤 준수한 결과를 보인다는 점 때문에 많은 연구자들이 도전하고 있는 현실이다. 아래 그림은 [DCGAN, Alec Rudford et al, 2016](http://arxiv.org/abs/1511.06434 ) 에 실린 이미지로 침실 이미지를 학습을 시켰다니, 아래처럼 자연스러워 보이는 침실 이미지를 생성하였음을 보여준다.
  

이번 class에서는 가급적이면 수식을 배제한 상태에서 개념적으로 GAN에 대해서 설명을 하고자 하였다. 개념적으로는 서로 경쟁하는 망을 이용하여 서로의 성능을 끌어 올린다는 아주 멋진 개념에서 출발하였다는 점과, 기존에 발표되었던 복잡한(?) 생성 모델들에 비해 빠르고 뛰어난 성능을 보인다는 점에서 주목을 받을 가치가 있다. 다음 class에서는 이런 개념적 이해를 바탕으로 수식까지 살펴보면서 좀 더 깊게 살펴볼 예정이다.
​
​

---

### 3. Introduction to the GAN

- 지난 Class에서는 GAN의 동작 원리에 대하여 살펴보았다. 이번 class에서는 좀 더 이론적인 측면에서 GAN을 살펴보고자 한다.

#### Generative Model과 Maximum Likelihood Estimation

- Ian GoodFellow의 GAN tutorial(NIPS 2016 Tutorial: The Generative Adversarial Networks)에서는 생성 모델을 아래 표와 같이 분류를 하였다. 위 표에서 볼 수 있듯이, Maximum Likelihood에 기반을 두고 있는 대부분의 생성모델은 크게 보면, 모델에 대한 확률변수를 구하는 `explicit density` 방식과 구하지 않는 `implicit density` 방식으로 나눌 수가 있다.
- Explicit density 모델은 아래 그림의 예처럼, 왼쪽에 있는 데이터 분포로부터 오른쪽의 그림에서처럼, 확률 분포를 구한다.
  그런데 위 그림처럼, 1차원 데이터의 경우는 그리 어렵지 않게 모델 분포를 구할 수 있지만, 아래 그림과 같은 이미지들이 있다면 과연 모델 분포함수를 쉽게 구할 수 있을까?
- 그리고 만약에 그 모델에 대한 분포를 구하는 것보다는 모델을 통하여 얻을 수 있는 결과에 더 관심이 많다면 굳이 모델에 대한 분포를 구할 필요도 없으며, GAN은 여기에 해당된다고 볼 수 있다.
- m개의 학습 데이터를 모델 분포에 최적으로 근사시키는 변수 $θ$를 구하고자 한다면 식(1)처럼, 결합확률 함수 형태로 표현이 가능하며, 이것을 해석을 쉽도록 로그 공간으로 변형을 시키면 식(2)와 같은 형태가 되며, 다시 컴퓨터를 이용한 수치해석에 쉽도록 로그 함수의 특성을 이용하여 다시 정리해주면 식(3)의 형태가 된다.
- 학습 데이터의 분포와 모델 분포를 최적으로 근사시키는 것은 Kullback-Leibler 분산을 최소화시키는 것과 같은 의미이며, 이것은 다시 아래 수식으로 표현이 가능하게 된다.
- 즉, $P_{data}(x)$를 $P_{model}(x)$의 차가 최소가 되도록 해줘야 하는데, 여기서 $P_{data}(x)$와 $P_{model}(x)$​​을 직접 구하는 대신에, GAN은 위에서 살펴본 것처럼, 구체적인 확률분포를 구하는 대신에 `Generator`와 `Discriminator`라는 서로 경쟁하는 2개의 망을 이용해 학습 데이터의 분포와 모델의 분포를 일치시키는 또는 근사시키는 작업을 수행하며, 이 과정을 통하여 실제로는 Maximum Likelihood 추정을 하는 셈이 된다.

#### GAN의 생성망과 AutoEncoder의 유사성

- [문제 인식] 일반적으로 고차원을 갖는 데이터를 ML(Maximum Likelihood) 추정을 통하여 생성 하려면, 고차원 학습 데이터로부터 직접 확률 분포를 추정해내야 하는데 이 작업은 일반적으로 매우 어렵거나 불가능하다.
- 이것에 대한 해결책은 random noise와 같이 상대적으로 저차원이면서 단순한 구조를 갖는 입력으로부터 학습 데이터의 분포로 변형할 수 있는 방법을 사용하면 가능할 수도 있으며, 이런 곳에 신경망을 사용하면 효과를 볼 수 있다.
  - 아래 그림은 입력으로 잡음 z를 받은 후 생성망을 통하여, 학습 데이터의 분포와 비슷한 데이터를 만들어내는 것을 보여준다.
    입력으로 random noise z를 받아들이는 부분의 이해가 어렵다면, 아래의 AutoEncoder 경우를 떠올려보면 이해가 가능할 것이다. (AutoEncoder는 “읽기쉬운 머신러닝”편 참고)
- AutoEncoder를 사용하면, encoding 과정을 통하여 차원을 줄여 compressed representation을 만들어내고 다시 decoder를 통해 차원을 키우면 학습이미지를 복원할 수 있다. 여기서 compressed representation은 복원이미지 “2”를 만들어내기 위한 latent variable이라고 볼 수 있으며, 이 값을 잘 정하면 디코더를 통하여 “2”를 복원해낼 수 있게 된다.
  생각을 좀 바꿔보면, 어떤 random noise가 있더라도 디코더를 잘 학습시키면 (혹은 학습시키는 방법을 찾아낼 수 있다면) 원하는 분포를 갖는 데이터 생성할 수 있다는 이야기가 된다. 또한 이런 디코더가 만들어지면, 입력의 random noise 값을 바꾸면 다른 데이터를 생성할 수 있다는 것도 충분히 추론할 수 있게 된다.
- AutoEncoder에서는 차원이 줄었다가 다시 확장이 되더라도 입력과 출력을 갖게 만드는 방식으로 학습이 가능하였는데 인코더를 없다면, random input z를 원하는 분포를 갖는 데이터를 만들어 낼 수 있는 디코더는 어떻게 만들어낼 수 있을까?
- **2-player minimax game**
  - GAN은 블로그 part1에서 설명을 했듯이, 생성망(G)을 가이드하기 위한 방법으로 판별망(D)를 두었으며, 이 2개의 신경망이 서로 적대적으로 동작하면서 마치 강화학습처럼, 서로를 발전시키는 방식을 사용한다.
    드디어 Random 입력이 들어오더라도 이제는G를 학습시킬 수 있는 수단이 만들어진 것이다.
    D는 입력 데이터가 진짜이면 1을 출력하고 가짜이면 0이 출력하며 최대한 그럴듯한 가짜까지도 구별해낼 수 있어야 한다. G는 D를 속이는 방향으로, 즉 G에서 만들어진 데이터가 D에서 1이 되도록 학습을 해야 한다.
    GAN의 학습 과정은 기본적으로 상호 협조적이지 않는 상대가 서로 최적의 상태(Nash Equilibrium)에 도달하려고 노력하는 방식이라서, 목표를 달성하려면 적절한 objective function을 설정해야 하고, 학습을 시키는 방법도 매우 중요하다.

##### Objective Function 설정

- 학습의 목적함수(objective function)은 아래처럼 정할 수 있다. 
  - 먼저 우변의 첫번째 항은 D가 학습 데이터를 진짜라고 판단할 수 있는 능력을 말하며, 
  - 두번째 항은 D가 G에서 만든 데이터를 가짜라고 판정할 수 있는 능력을 말한다.
- 목적함수가 정해지더라도 G와 D의 학습 방향은 좀 다르다.
  - D는 가짜와 진짜에 대한 구별을 잘 하는 방향으로 학습을 진행해야 하기 때문에 위 함수를 크게 만드는 방향으로 학습을 진행한다.
  - G는 D를 속일 수 있을 정도로 진짜 같은 가짜를 만드는 방향으로 학습을 하며, 위 함수에서 첫번째 항은 G의 학습과 무관하기 때문에 두번째항을 작게 만드는 방향으로 학습을 시킨다.

#### G에 대한 학습 목표를 바꾸자

- 앞서 살펴본 내용을 보면 G의 경우, $1 - D(G(z))$를 최소화 시키는 방향으로 학습을 진행하여야 하는데, 여기에는 약간의 문제가 있다.
그 이유는 학습의 초기에는 G가 진짜 같은 가짜 이미지를 만들어낼 능력이 별로 없기 때문에 D가 쉽게 가짜를 구별해낸다. 이렇게 되면 gradient 값이 작아지기 때문에 학습이 잘 되지 않는 문제가 있다. 
- 이것은 아래 그림과 같이 $1 - D(G(z))$의 경우 파란색 곡선의 형태가 되는데, 0 근처에서의 gradient 값이 너무 작다. 그러므로 1 ? D(G(z)) 대신에 D(G(z))가 최대가 되는 방향으로 학습 방법을 바꾸면 녹색 곡선의 형태가 되어, 초기에 큰 gradient가 확보되면서 학습이 제대로 되지 않는 문제를 해결할 수 있게 된다.

##### 학습 방법

이렇게 학습 방향이 정해지더라도, 2개의 결합된 망을 학습을 시켜줘야 하기 때문에 적절한 학습 방법이 필요하다. 

논문에서는 학습 방법으로 2개의 망을 번갈아 가며 학습시키는 방법을 권장하였다. 

- 먼저 G를 고정시킨 상태에서 k-step 만큼 D를 학습을 시키며, 다음은 D를 고정시킨 상태에서 G를 1-step 만큼 학습을 시키며, 이 과정을 내쉬 평형 상태에 이를 때까지 반복한다. 논문에서는 k 를 1로 하였기 때문에, 실제로는 D와 G를 번갈아 가면 학습을 시키는 형태였다. k를 1로 할 것인지 아니면 1보다 큰 값을 설정할 것인지에 대한 golden rule은 아직 없기 때문에 설계자가 실험을 통해 결정을 해야 한다.
- 실제로 G가 진짜 같은 가짜 데이터를 잘 만들어 냈는지 자동으로 판단할 수 있는 방법이 별로 없기 때문에 학습 종료에는 사람의 개입이 필요하다.

이번 class에서는 좀 더 이론적인 관점에서 GAN에 대하여 살펴보았다. 생성 모델을 만들기 위해 학습 데이터의 확률 분포를 직접 구하는 대신에 random noise 입력으로부터 학습 데이터를 만들어 내는 개념, 2개의 적대적인 망을 결합을 시켜 game 이론에 의한 학습 방법은 매우 인상적이다.
하지만 GAN은 학습의 안정성이 떨어지고, 사람의 개입을 필요로 한다는 점에서 약점을 갖고 있다. 다음 class부터 GAN의 안정성을 개선시키고, GAN의 능력을 다양하게 발전시킨 다양한 종류의 GAN upgrade 버전에 대하여 살펴볼 예정이다.
​

---

### 4. DCGAN

#### Introduction

- AI에 관련된 가장 핫한 소식을 정리해서 전하는 주간 뉴스레터 “Deep Hunt”에서 발표한 자료에 따르면 2014년 Ian GoodFellow가 GAN에 대한 기본 개념을 발표한 이후, 거의 매주 GAN 관련 논문이 발표가 되고 있으며, 아래 그림과 같이 폭발적으로 증가하고 있는 것을 확인할 수 있다.
  <<그림 출처: https://deephunt.in/the-gan-zoo-79597dc8c347 >>
- 이렇게 쏟아지는 여러 GAN 관련 후속 논문들 중에서 단연 돋보이는 논문 중 하나는 Alec Radford 등이 2015년에 발표한 [DCGAN: (Unsupervised Representation Learning with Deep Convolutional GANs)](https://arxiv.org/pdf/1511.06434.pdf) 이며, 이 DCGAN은 Ian GoodFellow의 original GAN의 구조를 발전시켜 최초로 고화질 영상을 생성시킬 수 있기 때문에, 어떤 요인들이 이런 발전을 가능하게 했는지 꼼꼼히 살펴볼 필요가 있다. 워낙 유명한 논문이라서 불과 2년만에 1000회 이상 인용이 되었을 정도이다.
  

#### 기존 GAN의 문제점과 DCGAN에서 이루고자 하는 과제

Ian GoodFellow가 2014년 처음 발표한 original GAN의 문제점

- MNIST와 같이 비교적 단순한 이미지는 괜찮은 이미지를 생성하였지만, CIFAR-10과 같은 복잡한 영상에 대해서는 그렇게 좋은 이미지를 생성했다고 볼 수 없다. (아래 그림에서 a)는 MNIST, b)는 TFD(얼굴 데이타), c)와 d)는 CIFAR-10 데이터를 original GAN을 통해 생성한 경우)
- 이 문제뿐만 아니라, 학습 시 안정성이 떨어지는 심각한 문제가 있었다. 

DCGAN의 Solution

- DCGAN 논문의 저자들은 이 문제를 해결하기 위해, CNN을 적용하기 위해 다양한 구조에 대한 실험 끝에 최적이라고 생각되는 구조를 찾아냈으며, DCGAN 이후에 발표된 다른 대부분의 논문들에서는 이들이 발견한 구조를 사용하고 있다.
- 또한 이들은 입력 데이터로 사용하는 입력 잡음(input noise) z에 대한 의미를 발견하였다. `Generator`의 입력으로 들어가는 z 값을 살짝 바꾸면, 생성되는 이미지가 그것에 감응하여 살짝 변하게 되는 vector arithmetic의 개념을 찾아냈다. 이 부분은 뒤에서 좀 더 자세하게 살펴보기로 한다.

#### DCGAN의 구조적 특징

DCGAN 연구진들은Original GAN의 망을 단순하게 CNN 을 적용하는 것만으로는 충분히 좋은 결과를 얻을 수 없다는 사실을 실험을 통해서 확인을 하였으며, 최적의 결과를 내기 위해, 다음과 같은 5가지 방법을 적용하였다.

1. Max-pooling layer를 없애고, strided convolution이나 fractional-strided convolution을 사용하여 feature-map의 크기를 조절.
2. Batch normalization을 적용하기.
3. Fully connected hidden layer를 제거하기.
4. `Generator`의 출력단의 활성함수로 Tanh를 사용하고, 나머지 layer는 ReLU를 사용.
5. `Discriminator`의 활성함수로 LeakyReLU를 사용.

실험을 통하여 이들이 찾아낸 최적의 `Generator`구조는 아래 그림과 같다.

-  `Generator`는 random input을 받아들여, `Discriminator`에서 사용할 수 있는 image 데이터를 생성해내야 하기 때문에 최종 출력은 64x64 크기인 컬러 영상이 나와야 한다.
  - 입력으로 사용하는 latent variable이 “가로x세로” 형태의 image 형태가 아니기 때문에, 이것을 CNN 망과 연동을 시키기 위한 image feature-map 형태로 변형시킬 수 있는 “Project and reshape” 블락이 필요하다.
  - 이 출력은 convolution layer로 넘겨지게 되는데 feature-map의 크기를 키워야 하기 때문에 영상의 크기를 키워줄 수 있는 fractionally-strided convolution (dilated convolution)이 필요하다.
  - 일반적으로 strided convolution은 stride 값을 1 이상의 정수값을 사용하면 pooling과 마찬가지로 출력의 크기를 줄이는 효과를 얻을 수 있다. (본 블로그의 Convolution layer[2] 참고) 하지만 stride 값을 1보다 작은 분수를 사용하는 경우는 반대로 출력의 크기를 키울 수 있으며, 이 경우를 구별하기 위해 fractionally-strided convolution 이라고 부르기도 한다. (자세한 설명은 본 블로그의 dilated convolution 편 참고)
    `Generator`에 적용된 방법을 종합하면 아래 그림과 같은 결과가 된다.

DCGAN에서 사용한 `Discriminator`의 구조는 아래 그림과 같다.

-  `Discriminator`는 64 x 64 크기의 컬러 영상을 입력 받아 진짜 혹은 가짜의 1차원 결과를 출력한다.
-  활성함수로는 아래 그림에서 볼 수 있는 것처럼, LeakyReLU를 사용한다. 
  - LeakyReLU는 ReLU가 음수 영역에서 0을 출력하던 것과 달리, 약간의 기울기를 갖는 값을 출력한다.
- AutoEncoder에서는 차원이 줄었다가 다시 확장이 되더라도 입력과 출력을 갖게 만드는 방식으로 학습이 가능하였는데 인코더를 없다면, random input z를 원하는 분포를 갖는 데이터를 만들어 낼 수 있는 디코더는 어떻게 만들어낼 수 있을까?

##### 학습 결과

저자들은 DCGAN을 LSUN(Large-scale Scene Understanding), ImageNet-1K, Face dataset을 이용하여 학습을 시켰다. 이미지에 대해서는 별도의 전처리(pre-processing)을 하지 않았으며, 영상의 값을 [-1, 1]의 범위로 scaling 하여 사용하였다.

300만장이 조금 넘는 LSUN dataset을 이용하여 학습시키고 생성해낸 침실 영상은 하였다. 1-pass 후에 얻은 침실 영상은 아래와 같으며, 겨우 1-pass만을 진행했기 때문에 `Generator`가 학습 영상을 기억할 수가 없는 것은 확실하며, 영상에서 약간 조잡한 부분들이 눈에 띄기는 하지만 꽤 준수한 결과가 얻어짐을 확인할 수 있다.

다음은 5-pass를 거친 다음에는 한결 나아진 것을 확인할 수 있다.

Latent 변수 z의 값을 조금씩 변화를 시켜보면

학습을 마친 후 latent 변수 z 값을 조금씩 변화시켜보면, 아래 그림에서 볼 수 있는 것처럼 생성된 이미지가 조금씩 달라지는 것을 확인할 수 있다. 예를 들어 마지막 줄의 경우는 텔레비전이 있는 침실 영상이 서서히 바뀌면서 큰 창문을 가진 영상으로 바뀌는 것을 확인할 수 있다.

##### `Discriminator`에서 얻어진 특징들에 대한 시각화

많은 수의 학습 영상을 통해 얻어진 `Discriminator`의 특징을 시각화하였더니, 아래 그림처럼, 각각의 filter들이 침대나 창문과 같이 침실의 일부분을 학습할 수 있다는 사실을 확인할 수 있었다.


#### Vector arithmetic

기본 개념

- Tomas Mikolov 일행은 “Distributed Representation of Words and Phrases and their Compositionality” 논문을 통하여 단어들에 대한 representation을 벡터 공간에 매핑을 시키면 재미있는 linear arithmetic이 가능함을 보였다. 
  - 예를 들어, vec(“Madrid”) ? vec(“Spain”) + vec(“France”) 연산을 하게 되면, 다른 어떤 단어보다 vec(“Paris”)에 가깝다는 사실 같은 것을 말이다.

DCGAN 에서의 vector arithmetic 가능 확인 

- 비슷한 실험을 DCGAN 개발자들도 하였는데, 학습을 통해서 얻어진 z 값들을 이용한 vector arithmetic이 가능하다는 사실을 발견하였다. 
  - 아래 그림의 예처럼, vec(“안경을 낀 남자”) - vec(“안경을 끼지 않은 남자”) + vec(“안경을 끼지 않은 여자”) 연산을 수행하면, “안경을 낀 여자 영상”이 생성되는 것을 확인할 수 있었다. 이것을 보면 학습을 제대로 수행을 하게 되면, z값이 의미 없는 값이 아니라 각각의 영상이 갖고 있는 representation을 제대로 나타낸다는 것을 확인할 수 있다.
- 또 다른 재미 있는 결과는 아래 그림처럼, 왼쪽을 보고 있는 얼굴의 벡터와 오른쪽을 보고 있는 얼굴에 대한 벡터에 대한 interpolation을 수행하면, 마치 얼굴의 각도가 돌아가는 것 같은 영상을 얻어낼 수도 있다.

DCGAN의 구조는 어떤 이론적인 추론을 통하여 얻어진 것은 아니지만, 실제로는 꽤 괜찮은 결과를 얻을 수 있었으며, latent variable z의 의미도 파악할 수 있었다. 또한 unsupervised learning을 통하여 영상이 갖고 있는 정보(representation)들에 대하여 잘 파악할 수 있었으며, unsupervised learning을 통하여 학습된 `Discriminator`가 잘 동작한다는 것도 확인할 수 있었다. 밑에서는 Semi Supervised GAN을 알아보자.



### 5. Semi-Supervised GAN

#### Introduction : Data의 범람

- IDC(International Data Corporation)의 조사에 따르면, 2013년의 digital data의 총량은 4.4 zettabyte(10의 21제곱) 정도 수준이며, 2020년에 이르면 44zetabyte가 될 것으로 예상하고 있다. 이것을 보면, 엄청난 속도로 digital data가 늘어난다는 것을 알 수 있다.
  <그림 출처: https://www.seagate.com/files/www-content/our-story/trends/files/Seagate-WP-DataAge2025-March-2017.pdf >
- 이렇게 쏟아지는 데이터에 대하여 딥러닝을 통해 의미 있는 정보를 끄집어 내고 활용할 수 있다면 좋겠지만, 여기에는 labeling(데이터에 맞는 꼬리표 달아주기) 문제가 발생한다.

지도 학습(Supervised Learning)은 데이터 x에 대한 label y의 관계, 즉 $p(y|x)$를 파악하는 문제로 볼 수 있으며, 지금까지 잘 개발된 훌륭한 방법론이 존재하고 있다. 

- 하지만 데이터 x에 대한 꼬리표 y가 항상 필요하며, 데이터에 대한 꼬리표를 달아주는 작업은 시간도 오래 걸리고 비용도 많이 발생하는 심각한 문제가 있다. 
- 또한 엉뚱한 꼬리표가 붙게 되면, 학습이 틀어지는 문제가 발생하게 되고, 이로 인한 오류를 해결하는데 또 많은 시간과 노력을 필요로 한다.

그래서 비지도 학습(unsupervised learning)에 대한 필요나 기대가 점점 커지고 있는 상황이지만, 아직 모두가 만족할만한 수준의 비지도 학습 알고리즘은 개발이 되지 못한 실정이다.


#### Semi-Supervised Learning

- Semi-supervised learning이란 지도 학습과 비지도 학습의 중간 정도에 위치하는 학습 방법이라고 볼 수 있으며, 꼬리표가 달린 데이터를 이용한 지도 학습과 꼬리표가 없는 비지도 학습을 결합하여 좀 더 학습 능력을 개선하자는 목표로 많은 연구가 되어 있다.
- 위 그림은 Semi-supervised learning을 개념적으로 설명하기 위한 것으로, 빨간색과 녹색의 십자가가 꼬리표가 붙은 데이터고 나머지 회색점들이 꼬리표가 없는 데이터라고 했을 가정을 했을 경우, 위 그림과 같이 학습을 통하여 빨간색과 녹색 그룹을 구별할 수 있데 되는 것을 말한다.

활용

- Semi-supervised learning을 이용하면, 꼬리표가 붙은 supervised data와 그렇지 않은 unsupervised data를 결합하여 상호 보완을 통하여 좀 더 효율적인 학습이 가능해진다. 
- 소수의 supervised(labeled) data는 다수의unsupervised(unlabeled) data대한 guide 역할을 할 수 있으며, unsupervised data는 소수의 supervised data로는 충분하지 못한 데이터 속에 숨어 있는 유용한 정보를 좀 더 잘 끄집어낼 수 있게 해준다. (다수의 데이터로부터 자연스럽게 일반화가 가능해짐)
- Semi-supervised learning은 위와 같이 유용한 결과를 얻을 수 있기 때문에 그 동안 다양한 방식으로 연구가 되어 왔다.

#### Semi-Supervised GAN (SGAN)

2016년 Augustus Odena가 발표한 논문 “Semi-Supervised with Generative Adversarial Networks”에는 기존 GAN 구조와 달리 아래 그림과 같은 구조를 갖는다. 그들이 Semi-Supervised GAN(줄여서 SGAN이라고 불렀기 때문에 이후에는 SGAN)은 기존 GAN의 `Discriminator`가 단순하게 진짜 혹은 가짜만을 구별하던 것과 달리, 위 그림처럼 classification을 수행하여 class의 종류 및 가짜를 구별할 수 있다.

- original GAN은 `Generator`와 `Discriminator`를 통하여 서로를 보완하는 방향으로 학습을 하여 결과적으로는 훌륭한 `Generator`를 얻는 것이 좀 더 큰 목적이었다면, SGAN은 결과적으로는 `Generator`와 `Discriminator`를 통하여, 좀 더 좋은 `Discriminator`를 얻는 목적이 좀 더 크다고 볼 수 있다.original GAN의 `Discriminator`는 진짜와 가짜만을 구별하면 되었기 때문에 sigmoid 출력이었다.
- SGAN은 class를 구별해야 하기 때문에 Softmax 출력을 사용한다는 점을 제외하면 구조는 거의 동일하다. 참고로SGAN의 기본 구조는 original GAN보다 성능이 좋은 DCGAN 구조를 기본 구조로 사용하고 있다.
- `Discriminator`의 Loss는 original GAN과 달리 label이 있는 데이터에 대한 supervised loss와 label이 없는 데이터에 대한 unsupervised loss의 합으로 된다.

##### SGAN의 objective function (loss function)

- Orginal GAN은 진짜와 가짜 여부만을 구별하면 되기 때문에 앞서 살펴본 것처럼 비교적 단순한 구조의 objective function으로 충분했다.
  하지만 SGAN에서는 `Discriminator`에서는 label이 있는 데이터와 없는 데이터를 같이 사용하여 classification을 수행해야 하기 때문에 `Discriminator`의 loss 함수는 original GAN과 달라야 한다는 것은 추론이 가능하다.
  - 원래 논문에서는 loss 함수에 대한 자세한 설명이 없기 때문에, source code가 있는 다음 site를 참조하면 좀 더 상세한 설명이 있다.
  (https://github.com/nejlag/Semi-Supervised-Learning-GAN)
- k개의 class를 구별할 수 있는 `Discriminator`를 만들고 싶다면, `Generator`에서 만들어지는 이미지를 가짜로 구별을 해줘야 되기 때문에, `Discriminator`에서는 총 (k+1)개의 class를 구별할 수 있어야 한다.
- 만약에 x가 가짜라고 한다면 x의 class 확률은 아래와 같이 표현이 가능하다.
- 한편 label이 붙은 입력 학습 데이터라면, class 중 하나로 분류가 되어야 하기 때문에 아래와 같이 표현할 수 있게 된다.

`Discriminator` loss는 supervised loss와 unsupervised loss의 합으로 구성이 되며, supervised loss는 이미 class를 알고 있기 때문에, 아래 식과 같이 표현할 수 있다. Unsupervised loss는 학습 데이터에 대하여 제대로 분류를 수행해야 하는 것과(가짜라고 구별하지 않는 것), `Generator`에서 생성해낸 데이터를 가짜로 구별해야 하기 때문에 아래와 같이 loss 함수를 정의할 수 있다.

##### `Generator`의 loss function

이제는 `Generator`의 loss function에 대해서 살펴보자. 

- `Generator`의 loss function은 기본적으로 "`Generator`에서 생성한 데이터의 분포가 학습 데이타의 분포를 제대로 mapping을 시켜줘야 한다"는 개념에 기반하기 때문에, 이들은 `Discriminator` 내부의 activation을 비슷하게 유도하는 feature matching loss를 사용하였다.

- Source code를 보면, 아래와 같이 최종 class를 구별하기 위한 fully connected layer 이전 layer인 "flatten5" layer의 activation을 비슷하게 유지하려는 방향으로 loss 함수를 설정한 것을 알 수 있다.

- 추가적으로 `Generator`의 loss로 고려한 것이 있는데, 이것은 "`Generator`는 `Discriminator`를 잘 속이는 방향으로 학습"을 해야 하기 때문에, `Generator`에서 생성한 데이터에 대하여 아래와 같은 cross entropy를 loss 함수에 추가하였다.

  `Generator`의 최종 loss 함수는 아래와 같이 정의할 수 있다.

  

##### 결과

- MNIST 데이터에 대하여 10%만 label이 있는 데이터를 이용한 실험 결과는 92.55% 정도의 성공률을 보였으며, 100% label이 있는 데이터의 성공률이 94.5%인 것을 감안한다면, 매우 좋은 결과임을 알 수가 있다. 

- 또한 10% label이 있는 데이터를 이용한 경우의 `Generator`에서 데이터를 만들어 내는 성능도 아래와 같이 우수함을 확인할 수 있다. 이 그림에서 맨 왼쪽은 epoch가 0인 경우로 당연히 잡음만 있으며, 가운데는 150 epoch 정도 진행이 된 경우로 그런대로 글자가 비슷하게 생성되는 것을 확인할 수 있으며, 왼쪽처럼 1000 epoch 정도 진행이 되면 꽤 높은 수준의 영상이 생성되는 것을 확인할 수 있다.

이번 class에서는 Semi-supervised learning을 수행할 수 있는 SGAN에 대하여 살펴보았다. 다음 class에서는 거의 비슷한 시기에 다른 objective function을 사용하여 Semi-supervised learning을 수행한 일명 CatGAN(Categorical GAN)에 대하여 살펴볼 예정이다.

---

### 6. CatGAN(Categorical GAN)

#### Introduction

비지도 학습(unsupervised learning)은 학습 데이터 x 속에 뭔가 학습에 필요한 정보가, 즉 수학적으로는 p(y|x)의 관계를 파악할만한 정보, 내재 되어 있다는 사실에 기반을 두고 있다. 그래서 비지도 학습은 입력 데이터에 대한 분포 p(x)를 직접 모델링하는 Generative한 방법과, p(x)에 대하여 신경 쓰지 않고 분류된 데이터들 간의 거리를 최대로 하는 것을 주 목표로 하는 Discriminative한 방법으로 나눌 수 있다.

- Generative한 방법으로는 Gaussian mixture model, k-means와 같은 방식이 있고, 
- Discriminative한 방법으로는 Maximum margin clustering(MMC)이나 regularized information maximization(RIM)과 같은 방법이 있다.
  ?
- K-means clustering과 Gaussian mixture model은 아래 블로그에 설명이 잘 되어 있으니, 여기를 참조하면 좋을 것 같다.
  Machine learning 스터디 (13) Clustering (K-means, Gaussian Mixture Model) - ...
  우리에게 처음 주어진 것은 왼쪽 파란 데이터이다. 각각의 데이터에 대한 정보는 아무 것도 없는 상태에서 주어진 데이터들을 가장 잘 설명하는 클...
  sanghyukchun.github.io

비지도 학습만으론 만족스럽지 못한 성능이 나오는 경우에도, 비록 일부만이라도 label이 달려 있는 경우에는 semi-supervised learning 방법을 적용하면, 꽤 만족할 만한 성능을 얻을 수 있다는 것은 이미 지난 블로그에서 확인을 하였다.

이번 블로그에서는 지난 블로그에서 살펴본 SGAN(Semi-supervised GAN)과 거의 비슷하게 original GAN을 변형시켜 semi-supervised learning을 수행할 수 있는 CatGAN(Categorical GAN)에 대하여 살펴볼 예정이다.

CatGAN은 “Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks”라는 제목으로 ICLR 2016 컨퍼런스 논문으로 발표가 되었으며, 다음 사이트에서 확인을 할 수가 있다.
https://arxiv.org/pdf/1511.06390.pdf

#### RIM(Regularized Information Maximization)

본격적으로 CatGAN을 설명하기에 앞서, CatGAN이 Discriminative clustering을 위해 이론적 기반을 두고 있는 RIM(Regularized Information Maximization)에 대해서 간단하게 살펴보자.
RIM은 2010년 “Discriminative Clustering by Regularized Information Maximization”이라는 논문에 소개가 되었다.
https://pdfs.semanticscholar.org/ab70/609cea8dbf3eb54474590c1f153244fab931.pdf

##### RIM에서 주목하는 부분

- RIM에서 중요한 부분은 class separation과 class balance에 대한 trade off를 하는 것이다. Class separation 관점에서는 클래스를 나누는 경계(decision boundary)에 데이터가 없도록 해주는 것인데, unsupervised learning에서는 위 그림의 맨 왼쪽처럼 class의 숫자를 줄이는 방향으로 진행될 우려가 있기 때문에 이것만으로는 부족하다.

- 이 문제에 대한 보완적 개념으로 클래스가 고르게 배치될 수 있도록 class balance를 맞춰주는 방법인데, 이 경우에는 개별 데이터의 모두 별도의 클래스로 배치할 때 최적이 될 수 있기 때문에, 뭔가 규제를 해줄 수 있는 regularization 항목이 필요하게 된다. 위 그림에서 가운데가 class balance 관점에서 과도하게 클래스가 나눠지는 예이다.

- 그래서 RIM 개발자들은 regularization을 수행하는 R(W; x)를 도입하였으며, 이것은 어떤 p(y|x, W)를 선택하는지에 따라서 달라지게 되게, 이것이 위 가운데 그림처럼 과도하게 class가 나눠지는 것을 견제하는 역할을 하며, 그래서 object function은 아래 수식과 같은 형태가 된다.

#### Categorical GAN (CatGAN)

CatGAN의 구조는 이전 블로그에서 살펴본 SGAN과 구조적인 관점에서는 동일하기 때문에 개념적인 구성도는 아래 그림과 같이 동일하다. 단, CatGAN의 objective function은 SGAN의 objective function과 다른데, 이유는 개념의 출발이 다르기 때문이라고 생각한다.
CatGAN 개발자들은 자신들의 구조가 original GAN의 일반화 버전이거나 RIM의 확장 버전이라고 생각하였다. (그래서 뒤에서 살펴보겠지만, objective function을 정하는 부분이 SGAN과는 좀 차이가 있다)
original GAN의 `Discriminator`는 진짜와 가짜만을 구별하는 것을 목적으로 하였다면, CatGAN은 개념을 좀 더 확장하여 class까지 구별할 수 있어 GAN의 확장 또는 일반화 된 버전이라고 보았다. CatGAN의 `Generator`는 unsupervised 혹은 semi-supervised `Discriminator`의 성능을 보강해주는 역할을 하며, `Discriminator`는 흔히 알고 있는 classification network와 거의 동일한 구조를 갖지만 `Generator`와의 adversarial training을 통해 성능이 강화되게 된다.
CatGAN의 RIM의 확장 개념으로 볼 수도 있는데, 그 이유는 앞서 살펴본 것처럼, RIM에서 regularization 역할을 하는 함수가 추가가 되었듯이, CatGAN에 있는 `Generator`가 Classifier 역할을 수행하는 `Discriminator`에 대한 regularization 역할을 해주며, 이를 통해 더욱 강건한(robust) 분류기가 만들어지게 된다.
?
?

#### CatGAN의 objective function (loss function)

?
SGAN에서 original GAN과의 구조적인 차이를 고려하여 objective function을 만들었듯이, CatGAN 개발자도 독특한 objective function을 설계한다.
?
RIM에서 최적의 unsupervised classifier를 얻기 위한 objective function으로 엔트로피 개념을 사용하였듯이, CatGAN 개발자들은 자신들의 구조를 RIM의 확장판으로 생각을 하였기 때문에, unsupervised(혹은 semi-supervised) data를 각각의 class로 얼마나 잘 분류하는지를 알아볼 수 있는 “goodness of fit” 의 척도로 엔트로피 개념을 사용하였다.
위 그림에서 녹색은 `Generator` 부분을 나타내고, 보라색은 `Discriminator`에 해당이 된다.
`Discriminator`에서 학습 샘플은 반드시 특정 class에 속해야 하기 때문에, 위 그림의 (i)처럼 특정 class에 속할 확률이 높게 나와야 하며, 그렇기 때문에 엔트로피는 낮은 값이 된다. (ii)처럼, `Generator`를 거쳐서 만들어진 샘플은 특정 클래스에 속하지 않고, 가짜로 구별이 되어야 하기 때문에, 특정 클래스에 속할 확률은 모두 고르게 낮은 분포를 가져야 하며, 그렇게 때문에 엔트로피가 최대가 되는 방향이 되어야 한다. (iii)는 학습 샘플은 모두 특정 클래스에 속할 확률이 균일한 가정을 하였으며, 이렇다는 이야기는 결과적으로 입력 데이터 x에 대한 주변확률분포(marginal distribution)의 엔트로피가 최대가 되어야 한다는 뜻이다. 학습 데이터 모든 class에 걸쳐 균일하게 분포된다는 사실은 실제적으로는 힘든 조건이 될 수 있다. 그래서 논문의 부록을 보면 학습 데이터가 모든 class에 대하여 균일하지 않은 상황도 고려를 하고 있는데, 이 경우에도 주변확률분포에 대한 엔트로피를 최대화 시키는 방향으로 해주면 문제가 없다.
`Generator`에 대한 objective function도 비슷하게 구할 수 있다. `Generator`에서 만들어진 가짜 데이터는 `Discriminator`를 속이는 방향, 즉 class에 속하는 방향으로 학습이 이루어져야 하기 때문에 엔트로피를 최소화 하는 방향으로 학습이 되어야 하며, `Discriminator`의 경우처럼 생성된 샘플은 각 class에 고르게 속하도록 해줘야 하기 때문에 주변확률분포의 엔트로피가 극대화 되어야 한다.
그래서 결과적으로 얻어진 objective function은 아래의 식과 같으며, `Discriminator`와 `Generator`에서 sub-항목들에 대한 최대/최소의 방향이 다른 경우는 부호를 “-“로 하여 최소/최대 방향을 맞출 수 있도록 하였다.
위 식을 보면, marginal entropy 항목은 전체 데이터에 대하여 정의를 해주는 것이 맞지만, batch 모드로 수행을 하는 경우는 전체 데이터를 볼 수 없기 때문에 이 부분이 문제가 된다. 하지만 class의 개수가 batch 크기에 비하여 훨씬 작다면 이 문제는 크게 문제가 될 것이 없이 batch 대한 marginal entropy만을 고려해도 된다.
?
?

#### 결과

MNIST 데이터에 대한 실험 결과 unsupervised learning 만을 했을 경우에 에러율이 9.7% 수준인데 반면, 100개의 label이 있는 semi-supervised learning의 경우 에러율이 1.91% 정도 수준이고, 1000개의 label이 있는 경우는 에러율이 1.73% 수준이며, 전체 data에 대한 label이 준비된 supervised learning의 경우는 0.91%의 에러율이 나왔다. 결과는 아래 표와 같다. 단 100개 정도만 label이 있더라도 semi-supervised learning에서 괄목할만한 성능이 나온다는 사실을 확인할 수 있다.
위 데이터에서 “RIM + NN”은 CatGAN에서 `Generator`가 없는 경우이며, 이것을 보면 adversarial training이 매우 효과가 있다는 사실 역시 확인이 가능하다.
MNIST 학습에 사용한 망의 구조는 아래와 같다.

또한 합성 데이터에 대한 실험도 수행을 하였는데, 아래 그림과 같이 복잡한 경우에도 decision boundary를 잘 잡아서 구별을 잘 하는 것을 확인할 수 있다. K-means 방법이나 RIM+NN의 경우에는 오류로 인해 제대로 데이터를 구별할 수 없지만, CatGAN 방법을 사용하면, 두 개의 원을 잘 구별하며, generated data 역시 원 데이터에 근접함을 확인할 수 있다.

또한 `Generator` 역시 아래 그림처럼 꽤 준수한 이미지를 생성하는 것을 확인할 수 있다. 아래 그림에서 왼쪽은 MNIST 데이터를 이용하여 학습을 시킨 경우이고, 오른쪽은 CIFAR-10 데이터를 이용하여 학습을 시킨 경우이다.
이번 class에서는 Semi-supervised learning을 수행할 수 있는 CatGAN에 대하여 살펴보았다. 일부의 데이터에만 label이 있더라도 그것이 가이드가 되어 높은 성능을 얻을 수 있다는 사실을 확인할 수 있었다.
​
​

---

### 7. cGAN

#### Introduction

GAN은 `Generator`와 `Discriminator`의 상호 견제를 통하여, 기존 `Generator` model처럼 복잡한 확률 계산 없이도, 기존 어떤 `Generator` 보다 훨씬 실제와 비슷한 이미지를 만들어낼 수 있기 때문에 발표되자마자 급속하게 확산이 되었다.

GAN을 사용하면, 학습에 사용한 이미지와 흡사한 가짜 이미지를 만들 수 있지만, original GAN 모델에서는 어떤 이미지를 만들어낼지는 제어할 수가 없었다. 이것이 가능해지면, GAN의 활용도가 더 올라갈 것이라는 것은 분명하기에, 2014년 몬트리올대의 “Mehdi Mirza” 팀이 Conditional GAN (일명 cGAN) 이라는 논문을 발표한다.
논문 제목: Conditional Generative Adversarial Nets
링크: https://arxiv.org/pdf/1411.1784.pdf
cGAN은 2014년 11월에 발표가 되었음에도 불구하고, 624회나 인용이 될 정도의 이후 많은 연구자들의 영감을 자극하였으며, 대표적인 것으로는 Pix2Pix, CycleGAN 등이 있다.

#### Latent Variable

`Generator`는 아래 그림과 같이 random noise 혹은 latent variable이라고 불리는 ‘z’를 입력으로 받아 학습 영상의 분포를 반영한 실제와 비슷한 가짜 영상을 출력으로 내보내게 된다.
여기서 latent variable에 대하여 잠깐 살펴보자. Latent variable(내재 변수 혹은 잠재 변수)은 직접적으로 관측되는 변수가 아니라, 관측이 가능한 다른 변수들로부터 추론이 가능한 변수라고 위키에서는 정의하고 있다.
좀 더 이해가 쉽도록 구체적인 예를 들어 살펴보자. “건강”이라는 추상적인 변수는 직접적으로 측정할 수 없는 “내재 변수”이지만 몸이나 정신 상태를 표현하는 말로 자주 사용이 된다. 직접 계측할 방법은 없지만, 혈압, 맥박, 혈당, 체온, 체중, 허리둘레 등 “(많은) 관측이 가능한 외부 변수”로부터 추론이 가능하며, 이런 외부 변수로부터 추상적인 건강이라는 내재 변수를 도출하려면 수학적(?)인 혹은 경험적인 모델이 필요하다.
그럼 건강이라는 내재 변수는 왜 사용할까? 좀 전에 살펴본 각각의 변수를 떼어서 해석하면 몸 상태를 파악하기가 어렵지만, 이것들을 건강이라는 내재 변수(추상적 개념)로 엮어내면 해석이 쉬워지기 때문이다.
Autoencoder에서는 입력 데이터를 받은 후 encoding 과정을 거쳐 압축된 결과(compressed representation)을 만들어 내는데, 이것이 바로 내재 변수가 된다. 이미지는 색의 삼원소를 갖는 픽셀들의 배열로 구성이 되지만, 이것들을 encoder를 통해서 압축시키면 결과적으로 “z”가 얻어지며, 이렇게 얻어진 “z”를 다시 decoder 망을 통과 시키면 복원된 데이터가 나오게 된다. 여기서 뒷부분만을 보면, `Generator`의 과정과 매우 유사하다는 것을 확인할 수 있다.
GAN에서는 어떤 사람은 “z”를 noise로 표현하고, 어떤 사람은 latent variable이라고 한다. noise라고 쓰는 이유는 사전에 정의할 수 없음에 의미를 둔 표현이고, latent variable이라고 하는 것은 autoencoder처럼 이것으로부터 출력 영상을 만들 수 있기 때문에 이 표현도 쓴다.
이 latent variable에 뭔가 이미지 생성에 관련된 condition을 부가하고, 이것을 통해서 이미지 생성을 가이드 할 수 있다면, 이제는 원하는 방향으로 이미지를 생성할 수 있게 되며, 다양한 분야에 응용이 가능해진다. 몬트리올대 팀은 이 점에 착안하여 cGAN이라는 망을 고안하였고, 이후 많은 사람들이 더 다양한 시도를 해볼 수 있는 단초를 마련하였다.

Original GAN

이전 블로그에서 GAN은 2 player min-max game을 통하여 학습한다는 것을 알았다. 즉, 학습 과정에서 상호 협조적이지 않는 상대(`Generator`와 `Discriminator`)가 서로 최적의 상태(Nash Equilibrium)에 도달하려고 노력하는 방식이며, 이 목표를 달성하기 위한 objective function으로 아래와 같은 식을 사용한다.

#### Conditional GAN(cGAN)

Original GAN에 어떤 condition을 가하는 방법은 의외로 간단하다. `Generator`와 `Discriminator`에 특정 condition을 나타내는 정보 y를 가해주면 된다.
여기서 y의 형태는 특별하게 정해진 것이 아니기 때문에 다양한 형태를 가질 수가 있다. 예를 들어, 숫자 필기체를 인식하는 MNIST에서 원하는 숫자를 generation 하고 싶다면, 숫자의 class에 해당하는 label(논문에서는 one-hot encoding을 하였음)을 추가로 넣어주면 된다. 만약 MNIST를 one-hot encoding을 하면 10-bit가 필요하기 때문에 y는 10비트가 된다. 그렇지만 꼭 이렇게 class의 label 뿐만 아니라, 다양한 형태(multi-modal)도 가능하다. (Multi-modal condition은 나중에 pix2pix를 다루면서 자세히 설명할 예정이다.)
그림으로 이해하는 것이 훨씬 쉬울 것 같은데, 아래 그림처럼 `Generator`의 기존 latent variable에 y를 더해주면 되고, `Discriminator` 역시 마찬가지이다.
Original GAN에 부가 조건 y가 들어가는 점을 제외하면 동일하다는 사실을 확인할 수 있다.
이제는 cGAN을 풀어주기 위한 objective function을 구해야 하는데, y가 조건부로 들어간 것을 제외하면 동일하기 때문에, objective function 역시 아래처럼 거의 동일하다.

#### cGAN 결과

논문에서는 MNIST data에 대하여 조건 y를 one-hot encoding 시킨 class label을 사용했으며, 이 실험을 통하여 아래 그림처럼, 원하는 숫자가 condition y대로 생성되는 것을 확인할 수 있었다.
Conditional GAN은 생성 조건을 제어해줄 수 있는 condition y를 추가하는 간단한 동작만으로, 원하는 데이터를 생성할 수 있다는 것을 알았다. 이후 pix2pix와 같은 좋은 conditional GAN 계열의 논문이 나오는 촉매 역할을 하였는데, 이 망들에 대하여 차례로 살펴볼 예정이다.
​
​

### 8. Pix2Pix

#### Introduction

이전 블로그에서 살펴본 cGAN은, original GAN 모델에서 생성되는 이미지를 마음대로 조절할 수 없는 문제를 해결하여, GAN을 `Generator`로 사용할 때의 효용을 높였다. 즉 cGAN에서는 입력 잡음(z)에 원하는 방향으로 영상을 생성시키기 위한 일종의 가이드(y)를 같이 인가하여 학습을 진행하는 간단한 방법을 사용한 후, y값만 조작하면 생성되는 영상을 특정 class로 만들 수 있었다.
하지만, cGAN에서 가이드 y는 보통은 one-hot encoding 방식을 사용한 크지 않은 벡터이기 때문에, 비교적 소수 class의 이미지는 생성하는 것은 문제가 없지만, 마음대로 대충 그린 스케치(혹은 edge-map)에서 특정 대상을 마음대로 생성하거나, 흑백 영상을 마음대로 칼라 영상으로 바꾸는 것은 거의 불가능하다. (cGAN 논문에서 사용한 MNIST 데이터는 0에서 9까지 class가 10개뿐이라서 별 문제가 없었다.)
??
위 그림처럼, 주간 영상만으로 야간 영상을 만들거나 그 반대도 가능하고, 창문, 출입구, 베란다의 위치만 대충 잡아주면 건물의 정면 영상을 생성하고, 열화상 이미지를 보여주면 칼라 영상으로 변환하는 등의 복잡한 이미지 변환이나 생성을 해줄 수 있는 `Generator` 모델이 있으면, 매우 유용할 것이라 생각된다.
이런 것들을 가능하게 해주는 일종의 conditional GAN 모델이 2016년 버클리 AI Research Lab의 Phillip Isola 팀이 발표를 하였으며, 영상 변환이나 생성이 가능한 범용 툴 개념으로 Pix2Pix라는 이름을 붙였다. 발표하고 2년 정도인데 인용횟수가 무려 1065회나 되고, 인터넷을 통해 Pix2Pix라는 키워드만 입력해도 영상 변환에 관련된 데모 결과나 블로그 등을 쉽게 찾아볼 수 있다. 논문의 제목은 “Image-to-Image Translation with Conditional Adversarial Networks” 이며, 아래 링크에서 찾아보면 된다.
https://arxiv.org/pdf/1611.07004.pdf
Pix2Pix를 소개하는 대부분의 블로그 등은 구조적인 내용보다는 주로 변환의 결과나 변환을 통하여 얻어지는 기괴한 영상을 보여주는 등 흥미나 효과 위주로만 되어 있어, 좀 아쉬운 느낌이 있다. 본 블로그는 4부(part)로 구성될 예정이며, 1부(part1)에서는 주로 개념적인 설명을 하고, 2부와 3부에서는(part2/part3)에서는 내부 구조에 대하여 좀 더 자세히 살펴보고, 4부(part4)에서는 또 다른 `Generator` model인 Variational Auto-Encoder(VAE)와 비교도 해볼 예정이다.

#### Pix2Pix

원 논문은 Isola 팀이 쓰고 소스 코드를 공개하였지만, “Christopher Hesse” 가 만든 데모 프로그램이 좀 편리하며, 아래 링크에 많은 예제들이 있다.
https://affinelayer.com/pixsrv/
그 중 하나인, 대략적으로 외곽선만 그려주면 무조건 고양이 영상으로 변환을 해주는 "edges2cat"은 그리 많지도 않은 2천장 정도의 이미지로 학습을 시켰는데 놀랍게도 아래와 같은 준수한 결과를 낸다.
하지만 아래와 같이 고의적으로 이상하게 스케치 하는 경우는 기괴한 형태의 영상을 만들어내기도 하며, 이런 영상들은 인터넷을 검색하면 쉽게 얻을 수 있다.

#### Pix2Pix를 학습시키기

Pix2Pix의 기본 구조는 아래와 같으며, 여기서는 외곽선 영상(edge-map) x와 학습 목표 영상 y를 이용하여 학습시키는 예를 보여준다.
Pix2Pix를 학습시키려면, 원영상 x와 변환하고자 하는 목표 영상 y의 쌍 {x, y}가 필요하며, 이런 데이터를 ?(가급적이면 많이) 준비하여 학습을 시키면 된다. 아래 그림에서 G(x)는 망을 학습시킨 후 망을 통하여 얻은 결과 영상 중 하나이다.
Pix2Pix는 다른 일반적인 CNN과 달리, 비교적 1천장 수준의 적은 수의 이미지만으로도 꽤 준수한 결과를 얻을 수 있다. 아래 그림처럼, 충분한 수의 {신발 스케치 그림, 실제 신발 영상} 쌍을 이용하여 학습을 시키면, 출력과 같은 형태의 신발 영상을 생성할 수 있다.
?
논문이나 블로그에서 볼 수 있는 것처럼, 신발뿐만 아니라 핸드백의 경우도 꽤 준수한 결과를 낸다는 사실을 확인할 수 있다.
?
학습을 위한 {x, y} 영상을 준비할 때에, 많은 수작업이 필요하다. 하지만, 위 그림과 같이 가방의 영상으로부터 외곽선(edge) 영상을 만들 때는 canny edge 검출기와 같은 외곽선 검출 알고리즘을 사용하면 수고를 덜 수 있다.
Pix2Pix가 다른 알고리즘보다 뛰어난 성능을 보이는 이유
Pix2Pix가 나오기 전에도 꽤 많은 이미지 변환이나 생성 알고리즘이 있었다. 그것들 중 CNN에 기반한 방법들은 그 성능이 뛰어난 편이었고, 특히 style transfer는 영상을 유명한 화가의 스타일로 변환시키는데 빼어난 성능을 보였다.
[style transfer 소개2분 동영상] https://www.youtube.com/watch?v=Uxax5EKg0zA
[논문 제목] Image Style Transfer Using Convolutional Neural Networks
하지만, 기존에 발표된 많은 알고리즘은 Pix2Pix와 같이, 영상 변환의 형태와 무관하게 범용으로 적용하기는 적합하지 않아서, 변환하고자 하는 방향에 따라 적당한 알고리즘을 선택해야 한다.
반면에 Pix2Pix는 학습에 사용할 {x, y} 데이터 쌍을 1천장 정도만(많으면 더 좋겠지만) 준비를 하고 학습을 시키면, 변환의 종류와 관계 없이 동일한 framework을 사용할 수 있다는 점에서 획기적이다. 물론 변환되는 영상의 질에 다 만족할 수는 없겠지만, 한계를 이해하고 잘 활용한다면 매우 유용한 툴이 될 것이다. 논문에서 이야기한 것처럼, 어떤 의미에서는 예술가나 디자이너들의 상상력을 자극하는 도구로도 활용이 가능할 것 같다.
다른 CNN을 사용하는 방법에 비해 우수한 또 다른 이유는 loss function이 다르다는 점이다. 대부분의 CNN 방식에서는 L1 혹은 L2 loss를 최소화 시키는 방식을 사용하기 때문에 변환할 영상의 결과가 사람이 보기에 그럴듯하게 보이는 것과는 무관하게 보통은 유클리디안(Euclidean) 거리를 최소화 하는 쪽으로 진행하며, 이렇게 되면 나올 수 있는 결과의 평균을 취하는 쪽으로 가기 때문에 영상이 선명하지 못하고 blur 해지는 경향이 있다. 하지만, Pix2Pix는 GAN에 기반하기 때문에 loss에 대한 학습을 통하여 보다 선명한 결과를 얻을 수 있다.
대부분의 기존 이미지 변환 알고리즘들이 pixel-to-pixel 변환에 초점을 맞춘다. 이 경우에, pixel과 인접한 pixel 간에는 서로 영향을 끼치지 않고 독립적이라는 가정(unstructured loss)이 숨어 있는데, 실상은 특정 pixel과 인접한 pixel 간에는 상관 관계가 매우 높기 때문에 올바른 가정이라고 보기 어렵다. 반면에, Pix2Pix에서는 pixel과 pixel 간의 연관 관계(joint configuration)까지 고려한 structured loss 개념이 자동으로 적용되기 때문에 다른 변환 알고리즘들에 비해 비교적 좋은 결과를 얻을 수 있다.
이번 블로그에서는 Pix2Pix의 기본적인 개념들에 대하여 살펴보았다. 다음 part2에서는 내부 구조를 좀 더 세밀하게 살펴볼 예정이다.
​
​
[Part VIII. GAN]

6. Pix2Pix
Generative Adversarial Networks ? “Pix2Pix(Part 3)”
Part2에서, Pix2Pix 설계자들은 성능 확보를 위하여 어떤 고려를 하였는지 살펴보았다. 최적화를 위한 objective function을 설계할 때에 cGAN loss뿐만 아니라, L1 reconstruction loss를 함께 고려해줘야, 보다 선명한 이미지를 생성해낼 수 있었다. G(`Generator`)는 기본적으로는 전체적인 맥락 이해 및 복원을 위해 encoder-decoder 형태를 취하지만, 최종적으로 영상을 복원했을 때, 화질이 선명하지 못한 문제를 해결하기 위해 U-Net 구조를 취했다.
Part3에서는 D(`Discriminator`)에서 사용한 PatchGAN 구조 및 전체적인 성능에 대하여 살펴보고, Part4에서는 대표적인 생성 모델 중 하나인 VAE(Variational Auto-Encoder)를 간단히 살펴보고, 비교도 해볼 예정이다.
`Generator`와 `Discriminator` 망의 구조
Pix2Pix 망의 세부 구조에 대한 그림은 원 논문에는 없지만, 스탠포드 대학교의 Yuanfang Li가 Pix2Pix를 사용하여, 풍경에 대한 sketch에 자연색을 입히는 응용을 소개한 논문 “Automatic Sketch Colorization(2017)” 에서 망의 세부구조를 아래 그림과 같이 깔끔하게 그려주는 수고를 하였다.
[논문 링크]: http://cs231n.stanford.edu/reports/2017/pdfs/403.pdf
`Generator`는 앞서 Part 2에서 설명한 것처럼 U-Net 구조를 하고 있는데, 전형적은 U자형 형태로 표현한 것보다는 encoder와 decoder를 나란히 배치하고 중간에 있는 skip connection이 직관적으로 잘 드러나도록 하였다. `Generator`는 입력 영상을 받아서 변환을 시켜 출력 영상을 생성하는 것이 주 목표다. encoding 과정에서는 그림의 맥락을 이해하기 위해, feature-map의 크기를 줄여가면서(compression) 핵심 representation만 추출한다. 이후 decoding 과정을 통해 원영상을 복원하는데, 복원하는 과정에서의 compression 과정에서 날아간 정보를 보강하면서 선명성 확보를 위해 skip connection을 사용하였다.
`Discriminator`는 뒤에 자세하게 설명하겠지만, PatchGAN 구조를 사용하였다.
PatchGAN 기반의 `Discriminator`
Part 2에서 최적화의 목표함수는 아래와 같은 수식이라는 것을 이미 확인하였다.
`Generator`의 최적화 목표 함수에는 L1 loss, 즉 reconstruction loss가 포함이 되어 있는데 reconstruction loss는 원영상과 생성영상 사이의 유클리드 거리를 최소화 하는 방향이기 때문에 통상적으로 영상의 평균 성분, 즉 저주파에 집중하는 경향이 있다.
L1 loss가 저주파 영역을 감당하기 때문에 `Discriminator`에서는 고주파 성분(detail)에 집중하여, 진짜(real)/가짜(fake) 여부를 판단하면 된다. 통상적인 GAN 구조에서는 영상 전체에 대하여 score를 구하는 방식이었다. 원 논문에서는 이것을 ImageGAN이라고 불렀다.
과연 전체 전체 영상에 대하여 진짜/가짜를 구별하는 방식이 효과적일까?
전체 영역이 아니라, 특정 크기의 patch 단위로 진짜/가짜를 판별하고, 그 결과에 평균을 취하는 방식이 PatchGAN이다. 픽셀들 간의 연관성(correlation)은 거리에 비례하여 작아지는 경향이 있으며, 일정한 거리를 넘어서게 되면 상호 간에 별 의미가 없다. 그러므로 특정 크기의 patch에 대하여 진짜 같은 이미지를 생성할 수 있다면, 그리고 그런 patch의 수가 많아지는 방향으로 학습을 하게 된다면, `Generator`의 성능은 더 올라갈 수 있을 것이라는 사실을 추정할 수 있다.
Pix2Pix 팀은 위 가정을 증명하기 위해, 전체 이미지에 대하여 진짜/가짜 score를 판정하는 ImageGAN, 다양한 patch 크기에 대한 결과를 확인하기 위해patch 크기를 70x70인 경우와 16x16 경우, patch 크기를 pixel 단계까지 줄인 1x1 patch (이것을 PixelGAN 이라고 명명)에 대하여 실험을 하였는데, 그 결과는 아래 그림과 같다.
?
L1 loss만을 사용한 경우에는 blur 현상이 매우 심해 이미지 변환 툴로 쓰기에 부적합하며, 1x1 patch를 사용하는 pixel 간의 경우는 컬러 표현이 좀 낫기는 하지만, L1 과 오십보백보 수준이다. Patch 크기를 16x16으로 한 PatchGAN에서는 L1 보다 확실히 결과가 좋지만, 작은 patch 크기로 인해 patch의 경계 근처에서 타일(tile) 형태의 artifact가 생기는 것을 확인할 수 있다. Patch 크기가 70x70인 경우는 선명한 이미지를 얻을 수 있었다. 전체 이미지를 사용하는 ImageGAN은 대충 보기에는 70x70 PixelGAN과 비슷해 보이기는 하지만, FCN-score를 이용해 수치적으로 비교하면 70x70 보다 수치가 낮게 나오는 것을 확인할 수 있다.
?
이 실험 결과를 바탕으로 결론을 내리면, 전체 이미지에 대하여 진짜/가짜를 가리는 것보다는 correlation 관계가 유지되는 범위에 해당하는 적절한 크기의 patch를 정하고 그 patch들이 대부분 진짜인 쪽으로 결과가 나오도록 학습하는 것이 효율적이라고 할 수 있다.
Pix2Pix 팀은 patch 크기가 70x70일 때 가장 효과적이라고 주장을 하였지만, 다른 연구 개발팀에서는 patch 크기가 좀 더 큰 경우가 더 효과적이라는 주장도 있다. Patch 크기는 전체 이미지 크기 및 영상 전체에서 특정 픽셀과 다른 픽셀들 간의 연관 관계가 미치는 적절한 범위를 고려하여 hyper-parameter 성격으로 정하는 것이 바람직할 것 같다.
논문에서 Markovian `Discriminator`라는 용어를 사용한 이유도 Patch 간에는 상호 연관성이 떨어져 독립적으로 볼 수 있기 때문에 영상을 MRF(Markov Random Field)로 모델할 수 있기 때문이다.
PatchGAN의 장점
앞서 살펴본 것처럼, 서로 겹치는 patch로 나누고 각 patch 들에 대하여 진짜/가짜를 구별하는 방식으로 `Discriminator`의 최적화 목표 함수를 정하게 되면, 좀 더 선명한 영상을 얻을 수 있게 된다.
논문에서는 256x256 크기의 입력 영상과 256x256 크기의 unknown 영상을 concatenation 시킨 뒤 최종적으로 30x30x1 크기의 결과를 얻었다. 최종 feature-map의 1-pixel은 입력 영상의 70x70 영역(receptive field)에 해당한다.
PatchGAN의 개념을 깔끔하게 그리면 아래 그림과 같다. 아래 그림은 `Discriminator`로 PatchGAN과 ImageGAN을 사용하여 inpainting 기법을 성공적으로 구현한 논문에서 가져왔다.
[논문 제목]: Patch-Based Image Inpainting with Generative Adversarial Networks
[논문 링크]: https://arxiv.org/pdf/1803.07422.pdf
PatchGAN을 사용하게 되면, 전체 이미지가 아니라, 작은 patch에 대하여 sliding 방식으로 연산을 수행하기 때문에 파라미터의 개수가 훨씬 적으며, 이로 인해 연산 속도가 더 빠르며, 전체 이미지의 크기에 영향을 받지 않기 때문에 구조적인 관점에서도 유연하다고 볼 수 있다.
이런 장점이 있기 때문에 Pix2Pix 이후에 발표되며, GAN을 사용하여 영상을 변환하는 논문들에서는 대부분 PatchGAN이나 PatchGAN의 변형을 사용한다.
?
?
Pix2Pix 성능 평가 방법
영상을 생성하는 툴에 대한 평가는 정성적인 방법과 정량적인 방법을 사용할 수 있다. 기존에 많이 사용하던 원영상과 만들어진 영상의 픽셀간의 “mean-square error” 방식을 쓰기에는 문제가 많다.
먼저 정성적인 방법으론 딥러닝 연구에서 label이나 평가에 많은 기여를 한 아마존의 Mechanical Turk(AMT)를 사용하였다. (물론 AMT에서 일하는 사람들의 처우 문제는 여기서 논외로 한다)
영상을 봤을 때, 전체적인 느낌이 컬러도 풍부하고, 선명하며, 사람들이 보기에 그럴 듯해 보이는 것을 판단의 기준으로 삼았다. 약 50명 정도의 AMT 검사자들에 영상을 1초 정도 보여주고 판단을 하도록 하였다.
생성된 이미지가 사실처럼 보이는지에 대한 정량적인 기준을 만드는 것은 매우 어려운 일이다. 그런 의미에서 볼 때, AMT를 사용하면 주관적이기는 하지만, 실제로 사람들이 판단을 하기 때문에 다른 어떤 정량적인 기준보다 정확도가 높을 수 있다. 하지만 테스트 절차를 어떻게 구성하느냐, 검사자들에게 어떤 동기(인센티브)를 부여하느냐에 따라 달라지고, 검사자들이 실수를 하는 경우에 어디가 틀렸는지 알려주면, 그 때부터 검사 결과가 크게 달라질 수 있다는 점이다.
사람의 시각 시스템은 영상에 있는 대상이 무엇인지를 파악하는데 집중하는 경향이 있기 때문에 대상 인지에 필요하지 않은 국부적인 특징들은 상대적으로 소홀히 보는 경향이 있다. 그런데 실수에 대하여 지적 받게 되면, 사람이 갖고 있는 순응(adaptation) 능력으로 인해 국부적인 부분도 집중해서 보게 되기 때문에 일반적으로는 짧은 기간을 보게 하고, 어디를 집중할지도 사전에 알려주지 않는다.
이러한 사람의 시각 시스템의 특성을 고려한다면, 생성되는 이미지를 정량적으로 판단할 수 있는 기준도 (전체적으로는 진짜인지 가짜인지 판단하는 기준을 만들기 어려우므로) 영상 속에 존재하는 대상을 얼마나 잘 구별할 수 있는지로 판단하는 것도 그리 나쁜 방법이 아닐 것이다.
그래서 Tim Salimans 팀이 발표한 논문 “Improved Techniques for Training GANs”에 따르면, “Inception score”을 사용하여 생성된 이미지를 평가하였으며, AMT 검사자들의 결과와의 상호연관성(correlation)을 따져보니 상당히 높은 연관 결과가 나타나 classifier를 이용해 진짜/가짜 여부를 가리는 방식이 나름대로 의미가 있다는 결론을 내렸다.
[논문 링크]: https://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf
마찬가지로 Pix2Pix 개발자들은 Inception score 대신에 “FCN-8s score”를 사용하였는데 그 개념의 근간은 거의 유사하다고 볼 수 있다.
?
?
Pix2Pix 결과
Pix2Pix 결과는 앞서 Part 1에서 살펴본 것처럼, 상당히 뛰어나다. 먼저 사람들을 얼마나 실수하게 만드는지, 아래 그림처럼 “지도 -> 항공사진”과 “항공 사진 -> 지도”의 경우에 실험을 수행하였다.
?
실험 결과는 아래와 같으며, 지도를 항공사진 영상으로 바꾸는 경우에는 AMT 검사들을 무려 18.9%나 실수하게 만들었다. “L1”과 “cGAN” loss를 결합한 방식이 꽤 효과적임을 확인할 수 있다.
?
또한 영상을 보여주고, 그 영상에 대하여 label을 달아주는 semantic segmentation에 대한 실험도 하였는데, 아래 결과와 같이 Cityscape 영상을 사용하였다.
실험 결과는 아래와 같은 결과를 얻을 수 있었다. 영상에 대하여 label을 달아주는 실험이기 때문에 아무래도 저주파(평균)만을 취한 L1이 결과가 좋지만, detail 정보를 모두 갖고 있는 “L1+cGAN”이 거의 근접한 결과를 보이는 것을 확인할 수 있다. 이들은 기존 segmentation 방법과 달리, cGAN 방법을 이용하여 semantic segmentation을 수행한 첫번째 시도라고 주장하고 있다.
논문에는 앞에서 열거한 것 이외에도 많은 결과 영상들이 있으니 그것을 참고하면 될 것 같다. 어쨌든 Pix2Pix 개발자들의 주장처럼, Pix2Pix는 영상 변환의 범용 framework를 제공했다는 점에서 매우 의미가 있다고 볼 수 있다.
?
이번 블로그에서는 Pix2Pix의 내부 구조 중 Disrcriminator에서 사용하는 PatchGAN의 구조 및 전체적인 성능에 대하여 상세하게 살펴보았다. 다음 part4에서는 VAE와도 비교를 통하여 Pix2Pix 에 대하여 좀 더 이해하는 시간을 가질 예정이다.
​
​​
