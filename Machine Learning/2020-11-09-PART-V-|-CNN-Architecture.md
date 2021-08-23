---
title: "[기계학습] 5. CNN Architecture"
excerpt: "합성곱 신경망 Architecture"
slug: "ml"
layout: post
category: "5-ml-cnn-arch"
lang: en
tags: ["ml", "coursework"]
date: 2020-11-07T20:28:09-05:00
use_math: true
draft: false
---

PART V / CNN Architecture
ⓒ [라온피플](https://blog.naver.com/laonple), Stanford cs231n

### 1. Overview

- [Part Ⅳ. CNN] 1. CNN 개요 ~ 4. Convolution Layer [2] 를 통해 CNN의 개념과 구조에 대하여 살펴보았다. CNN에 대한 이해를 좀 더 깊게 하기 위해 유명한 CNN 구조(architecture)들에 대하여 하나씩 차례로 살펴볼 예정이며, 우선 그것들에 대해 간단히 정리하면 다음과 같다. 아래에서 살펴볼 최고의 구조는 대부분 대량 영상 인식 경연대회인 ILSVRC에서 우승을 하거나 준우승을 한 것들이다.

#### 1.1. LeNet

- Yann LeCun이 1990년대에 발표한 구조로, 처음으로 CNN이라는 개념을 성공적으로 도입하였으며,
- 주로 우편 번호나 숫자 등을 인식하기 위해서 개발이 되었다. 이미 앞선 class를 통해, 어느 정도 그 구조에 대하여 살펴보았으며, CNN역사에 있어 古典이 되고 있다.

#### 1.2. AlexNet

- 컴퓨터 비전 분야에서 CNN이 널리 적용되도록 한 계기가 된 구조로, CNN 분야에서 유명한 Krizhevsky와 Hinton 등에 의해 개발이 되었다.
- 2012년 ImageNet ILSVRC 대회에서 2위와 큰 성능차(AlexnNet의 에러율은 16%이고 2위의 에러율은 26%)를 보이며 우승한 것으로 유명하다.
- AlexNet의 구조는 LeNet과 유사하지만, 보통 convolutional layer 다음에 pooling(sub-sampling) layer가 오는 기본 구조와 달리, convolutional layer 바로 뒤에 convolutional layer가 온 점이 특이하다.

#### 1.3. ZF Net

- ILSVRC 2013년 대회에서 우승을 한 구조로 뉴욕대의 Matthew Zeiler와 Rob Fergus에 의해서 개발이 되었다.
- Zeiler와 Fergus의 앞 글자를 따 ZF Net이라고 알려지게 되었다.
- AlexNet의 hyper-parameter를 수정하여 성능을 좀 더 개선을 하였으며, 특히 중간에 있는 convolutional layer의 크기를 늘렸다.

#### 1.4. GoogLeNet

- ILSVRC 2014년 대회에서 우승을 한 구조로 구글의 Szegedy 등에 의해서 개발이 되었다.
- “Inception Module” 개념을 도입했으며, 이것을 통해 망의 파라미터 수를 대폭 줄일 수 있게 되었다. (참고로 AlexNet은 60M 파라미터가 있지만, GoogLeNet의 경우는 4M 임)

#### 1.5. VGGNet

- ILSVRC 2014년 대회에서 2등을 한 구조로 영국 옥스포트 대학교의 Karen Simonyan과 Andrew Zisserman에 의해서 개발이 되었다.
- 대량의 이미지를 인식함에 있어, 망의 깊이(depth)가 정확도에 어떤 영향을 주는지를 보여주었다. 망의 시작부터 끝까지 동일하게 3x3 convolution과 2x2 max pooling을 사용하는 단순한(homogeneous) 구조에서 depth가 16일 때 최적의 결과가 나오는 것을 보여줬다.
- GoogLeNet에 비해 분류 성능은 약간 떨어졌지만, 다중 전달 학습 과제에서는 오히려 더 좋은 결과가 나왔다. 그러므로 CNN 연구 그룹에서는 VGGNet의 구조를 좀 더 선호하는 경향이 있다.
- 단점이라면, 메모리 수와 파라미터의 수가 크다는 점이다.

#### 1.6. ResNet

- ILSVRC 2015년 대회에서 우승을 한 구조로 마이크로소프트의 Kaiming He 등에 의해서 개발이 되었다.
- 기존 DNN(Deep Neural Network)보다 layer 수가 훨씬 많은 Deeper NN에 대한 학습(training)을 쉽게 할 수 있도록 해주는 residual framework 개념을 도입했다. 
- 앞으로 몇 회에 걸쳐 앞서 간단하게 언급한 CNN의 구조에 대해서 상세하게 살펴볼 예정이다.
- 참고로 ImageNet과 ILSVRC에 대해 간단히 소개하면 다음과 같다.

#### 1.7. 부록 

##### 1.7.1. ImageNet

- 세계 최대의 영상 데이터 베이스이며, 마치 사람이 보고 판단할 수 있는 것처럼, 컴퓨터 비젼을 연구하는 사람들이 벤치마크로 사용할 수 있는 영상 데이터 베이스이다.
- 현재 약 22000종류로 분류가 가능한1500만장의 인터넷 기반의 영상이 확보되어 있다. 구성은 WordNet의 계층 구조를 따라 만들어졌으며, 1개의 synset(synonym set)에 대해서 평균 1000장 이상의 영상이 확보되어 있으며, 사람들이 각각의 영상에 대해 label을 붙여놨기 때문에, 머신 러닝 기반의 컴퓨터 비젼 프로그램을 개발하거나, 자신이 개발한 프로그램의 성능을 평가할 때 사용하기에 적합하다.
- ImageNet에 확보된 영상에 대해서는 저작권을 주장하기 않기 때문에 누구나 편안하게 다운로드 하여 사용할 수가 있다. 이 영상 데이터베이스는 2009년 스탠포드 대학교의 Li Fei-Fei 교수가 중심이 되어 확보가 되었으며, 현재는 여러 대학교 및 구글이나 마이크로소프트 같은 기업에서도 참여를 하고 있다.

##### 1.7.2. ILSVRC

- ImageNet Large Scale Visual Recognition Challenge의 약어로, ImageNet 영상 데이터베이스를 기반으로 하여 컴퓨터 비젼 분야의 성능의 우열을 가리기 위한 대회이며, 2010년부터 매년 열리고 있다.
- 이 대회를 통해 놀라운 성능을 보이는 논문들이 많이 발표가 됨에 따라, ImageNet은 점차 대량 영상 인식분야에서 표준 벤치마크로 자리를 잡아가고 있다.
- 대회의 분야는 image classification, single-object localization 및 object detection으로 나뉜다. Image classification과 single-object localization은 그림에 여러 개의 object가 있더라도 1개를 확실하게 구별할 수 있으면, 맞은 것으로 처리한다. 단 localization의 경우는 위치까지 찾아내야 한다.

### 2. LeNet

- CNN(Convolutional Neural Network) ? “LeNet5”
- 지난 시간에 예고 했듯이 이번 class부터는 best CNN architecture에 대해서 살펴볼 예정이다.

- Convolutional neural network라는 개념을 최초로 개발한 사람은 프랑스 출신의 Yann LeCun이며, 그는 현재 뉴욕대 교수와 Facebook 인공 지능 연구소의 기술이사를 겸하고 있다. 원래 우편번호와 수표의 필기체를 인식하기 위한 용도로 개발을 시작했으며, 그 연구 그룹이 최종적으로 발표한 구조가 바로 유명한 LeNet5 이다.
- 이번 class에서는 LeCun이 1998년에 발표한 “Gradient-based learning applied to document recognition” 논문과 기타 다른 참고 자료 등을 기반으로 고전적인 CNN 구조, LeNet5에 대하여 살펴볼 예정이다. 이미 앞선 class에서 LeCun의 LeNet5의 구조에 대해서 부분적으로 살펴보았고, 또한 구조 자체가 그리 복잡하지 않기 때문에 이해에 큰 무리가 없을 것이라 생각되며, 이 구조를 잘 이해하면, 다음부터 차례로 살펴볼 다른 CNN의 구조는 LeNet5와 어떻게 다른지, 어디에 초점을 맞춘 것인지 등등을 살펴보면서 CNN의 구조에 대한 이해가 깊어질 것으로 예상한다.

#### 2.1. LeNet ? Convolutional Neural Network의 古典

- Yann LeCun과 그의 동료 연구원들은 기존의 fully-connected neural network이 갖고 있는 한계를 잘 이해하고 있었으며, 이것들을 개선하기 위한 연구에 돌입했다.
- 기존 신경망의 문제점은 이미 [Part Ⅳ. CNN] 1. CNN 개요 를 통해 살펴보았듯이, fully-connected neural network이 machine learning에 손색이 없는 알고리즘이기는 하지만, topology 변화에 대응이 어렵다는 점이다.
- 이 연구진들은 [Part Ⅳ. CNN] 2. Why CNN? ~ 3. CNN의 구조 자료에서 살펴보았던 것처럼, local receptive field, shared weight 및 sub-sampling의 개념을 결합한 CNN(Convolutional Neural Network)의 개념을 개발하게 된다.
- 그들이 최초로 개발한 LeNet-1은 1990년에 발표하게 되는데, 이것이 LeNet-5의 前身(전신)이라고 볼 수 있으며, 크기가 작을 뿐이지 현재 LeNet-5의 모습을 거의 갖추고 있음을 알 수 있다. (아래 그림 참고)
- 처음 LeNet-1을 발표 했을 당시, 입력 영상의 크기는 28x28로 LeNet-5에 비해서 작으며, 1단계 convolution을 통해 얻은 feature-map의 개수도 4개로 작으며 2 단계 convolution에서 얻은 feature-map의 크기도 12개로 작았다. 하지만 sub-sampling을 적용하여 feature-map의 크기를 줄이고, 여러 단계의 convolution을 거치면서 작은 feature에서 좀 더 global한 feature를 얻어가는 과정이나, 의미 있는 global feature를 얻게 되면, 이것을 fully-connected neural network을 classifier로 사용하는 점도 LeNet-5와 동일하다.
- 여기서 5x5 convolution kernel을 통해 local receptive field의 개념을 적용하였고, 전체 이미지에 대해서 같은 kernel을 적용함으로써 shared weight 개념이 적용되었으며, 가장 큰 자극만을 취하기 위한 max-pooling 을 적용함으로써 sub-sampling이 적용되게 된다. 이들은 이 구조를 완성하고 비교를 통해, fully-connected network나 기타 다른 machine learning 알고리즘을 적용하는 것보다 훨씬 결과가 좋다는 사실에 고무되어, 이 구조를 점차 발전시키게 된다.
- 이후 이 연구 그룹은 현재의 LeNet-5와 구조적으로 훨씬 유사한 LeNet-4를 거쳐, 최종적으로 LeNet-5를 발표하기에 이른다.

#### 2.2. LeNet-5

- LeNet-1이 처음 개발될 때는 아무래도 당시의 컴퓨팅 능력이 떨어졌기에 파라미터의 수가 작은 망을 개발했으리라 생각이 된다. 개발 후 결과를 보니, 입력 이미지의 크기 및 convolution kernel의 개수를 늘리고 최종단에 있는 fully-connected layer의 크기도 키우는 방향으로 자연스럽게 연구 흐름이 이어진 것으로 보인다.
- 앞서 살펴보았지만, LeNet-5의 구조는 아래 그림과 같다. LeNet-1의 구조와 비교를 해보면, 구조가 상당히 유사하지만 크기가 달라졌다는 것을 알 수가 있다. LeNet-1에서는 16x16의 크기로 샘플 이미지의 크기를 줄인 후 그것을 28x28 크기의 중앙에 위치하도록 하였지만, LeNet-5에서는 MNIST의 28x28 test image를 32x32 영상의 중앙에 위치시켜 처리를 하였다. 아무래도 좀 더 큰 크기의 영상을 사용하기 때문에 down-size 영상에서보다 영상의 작은 부분(detail)에 대한 고려가 훨씬 많아지고, 그 결과로서 더 우수한 성능을 얻을 수 있게 된 것이다. LeNet-1의 경우는 약 10만번 정도의 multiply/add가 필요하지만, free parameter의 숫자는 3000개 이하로 작다. 이는 20x20 크기의 영상을 입력으로 받고, 300개의 hidden unit을 갖는 400-300-10 fully connected neural network에서 free parameter의 개수가 12만개 이상인 것에 비해 훨씬 작다는 것을 알 수 있으며, 실제 망을 통한 학습의 결과도 LeNet-1이 더 좋다. (논문에서 성능 비교의 수치가 나옴) LeNet-5는 앞서 언급한 것처럼 성능 향상을 위해, LeNet-1보다 큰 망을 설계하였다. 전체적으로 34만개의 connection이 있으며, free parameter의 개수도 약 6만개에 달한다.
- 하지만 성능 면에서는 놀랄만한 결과를 보인다. 실제로 논문에서 여러 알고리즘에 대해 비교한 결과를 보면, fully-connected NN에 비해 LeNet-1의 성능이 훨씬 뛰어나고, LeNet-5의 결과는 그것보다 더 좋다는 것을 알 수 있다. 아래 표에서 최고의 결과를 보인 Boosted LeNet-4는 LeNet-4에 대해 3가지 다른 방법으로 학습을 실시하여, 결과적으로 인위적인 distortion을 통해 training data가 많아지는 것과 같은 효과를 얻게 되었다.
- LeNet-5의 구조를 좀 더 상세하게 살펴보자. 
  - 구조를 나타내는 그림에는 Cx와 Sx 및 Fx가 나온다. 여기서 C는 convolution, S는 sub-sampling, F는 Fully-connected layer를 의미하며, 대문자 알파벳 다음에 오는 소문자 x는 layer의 번호를 나타낸다. 가령 C1은 첫번째 layer이고 convolution을 수행하는 layer 임을 나타낸다. 비슷하게 F6라면, 6번째 layer이고 fully connect NN 임을 의미한다. LeNet-5의 구조는 총 3개의 convolution layer, 2개의 sub-sampling layer 및 1개의 fully-connected layer를 갖고 있다는 것을 알 수가 있으며, convolution뒤에 sub-sampling이어지면서, 영상의 크기가 1/4로 줄어들게 된다. C1은 convolutional layer이며 32x32 영상을 입력으로 받아, 28x28 크기의 feature-map 영상을 만들어 낸다. 5x5 kernel을 사용하고, zero-padding을 지원하지 않기 때문에 boundary 정보가 사라지면서 28x28 크기의 feature-map 영상이 나오게 된다. Convolution kernel의 계수(parameter)는 사전에 결정되는 것이 아니라, 최적의 결과를 낼 수 있도록 학습을 통해 결정이 된다.
  - C1 단계는 각 convolution kernel에서 (총 26 = 25 +1)의 자유 파라미터가 있고, 그런 커널이 6개 있기 때문에 총 156개의 자유 파라미터가 있다.
  - S2는 sub-sampling을 수행하며, 2x2 크기의 receptive field로부터 average pooling을 수행하기 때문에, 결과적으로28x28 크기의 feature-map 영상을 입력으로 받아, 14x14 크기의 출력 영상을 만들어 내며, 각각의 feature map에 대해 1개의 대응하는 sub-sampling layer가 있다. Average pooling을 수행하기 때문에 weight 1 + bias 1 로 각각의 sub-sampling layer는 2개의 파라미터를 갖고, 자유 파라미터의 개수는 총 12개 있다. 
  - C3는 C1과 동일한 크기의 5x5 convolution을 수행하며, 14x14 입력 영상을 받아 10x10 크기의 출력 영상을 만들어 낸다. 6개의 입력 영상으로부터 16개의 convolution 영상을 만들어 내는데, 이 때 6개의 모든 입력 영상이 16개의 모든 출력 영상에 연결이 되는 것이 아니라, 아래의 테이블과 같이 선택적으로 입력 영상을 골라, 출력 영상에 반영이 될 수 있도록 한다. 이렇게 하는 이유는 연산량의 크기를 줄이려는 이유도 있지만,
    결정적인 이유는 연결의 symmetry를 깨줌으로써, 처음 convolution으로부터 얻은 6개의 low-level feature가 서로 다른 조합으로 섞이면서 global feature로 나타나기를 기대하기 때문이다. 이 단계의 자유 파라미터의 개수는 1516개이다. 이는 25(커널) x 60(S2의 feature map과 C3의 convolution에 대한 연결) + 16(bias)의 결과이다.
  - S4는 S2와 마찬가지로 sub-sampling 단계이며, 10x10 feature-map 영상을 받아 5x5 출력 영상을 만들며, 이 단계의 자유 파라미터의 개수는 32(2x16)개 이다. C5는 단계는 16개의 5x5 영상을 받아, 5x5 kernel 크기의 convolution을 수행하기 때문에 출력은 1x1 크기의 feature-map이며, 이것들을 fully connected의 형태로 연결하여 총 120개의 feature map을 생성한다. 이전 단계에서 얻어진 16개의 feature-map이 convolution을 거치면서 다시 전체적으로 섞이는 결과를 내게 된다. 
  - F6는 fully-connected이며 C5의 결과를 84개의 unit에 연결을 시킨다. 자유 파라미터의 개수는 (120+1)x84 = 10164가 된다. 입력 영상에 대해, 각각의 단계별 영상은 아래 그림과 같다. 단계별 영상을 보면, LeNet-5의 각각의 layer에서 어떤 결과를 도출해내는지 알 수가 있으며, topology 변화나 잡음에 대한 내성이 상당히 강함을 알 수가 있다.
- 이번 class는 LeNet-5의 구조에 대해서 알아보았다. 그리 복잡하지 않은 망을 이용하여 놀랄만한 성과를 얻을 수 있음이 확인이 되었다. 다음 class에서는 AlexNet에 대해 상세하게 살펴보기로 한다.

### 3.  AlexNet [1]

- ImageNet 영상 데이터 베이스를 기반으로 한 "ILSVRC(ImageNet Large Scale Visual Recognition Challenge) - 2012" 우승은 캐나다의 토론토 대학이 차지한다. 당시 논문의 첫번째 저자가 바로 Alex Khrizevsky이며, 그의 이름을 따, 그들이 개발한 CNN 구조를 AlexNet이라고 부른다.
- 아래 그림의 SuperVision은 AlexNet을 의미한다. 결과에서도 알 수 있듯이 압도적인 성능으로 1위를 하게 되어 크게 주목을 받게 되었고, 그 이후 매년 참가자들은 AlexNet의 결과에 고무되어 더 좋은 성능을 보이게 되며, 당장 그 이듬 해에 치러진 ILSVRC-2013에서는 대부분의 참가자들이 AlexNet보다 좋은 결과를 낸다. AlexNet은 구조적 관점에서 보았을 때, LeCun의 LeNet5와 크게 다르지는 않지만, 높은 성능을 얻기 위해, 꽤 여러 곳에 상당한 고려가 들어가 있으며, 특히 GPU를 사용하여 매우 의미 있는 결과를 얻었기 때문에 이후 연구자들은 CNN 구조를 설계할 때 GPU를 사용하는 것이 대세가 되었다. 이세돌 9단과 세기의 대결을 벌인 구글(딥마인드)의 알파고(distributed model) 역시, 1920개의 CPU와 280개의 GPU를 병렬적으로 사용한다.
- 이번 class에서는 2012년에 발표된 “ImageNet Classification with Deep Convolutional Neural Networks” 논문을 기본으로, AlexNet이 성능을 올리기 위해 구조적으로 어떤 점을 고려하였는지 살펴보기로 한다.

#### 3.1. AlexNet 기본 구조

- AlexNet의 기본 구조는 아래 그림과 같으며, 전체적으로 보면 2개의 GPU를 기반으로 한 병렬 구조인 점을 제외하면, LeNet5와 크게 다르지 않음을 알 수 있다.
- AlexNet은 총 5개의 convolution layers와 3개의 full-connected layers로 구성이 되어 있으며, 맨 마지막 FC layer는 1000개의 category로 분류를 위해 활성 함수로 softmax 함수를 사용하고 있음을 알 수 있다.
- AlexNet은 약 65만개의 뉴런, 6000만개의free parameter 및 6억3000만개의 connection으로 구성된 방대한 CNN구조를 갖고 있으며, 이렇게 방대한 망에 대한 학습을 위해 2개의 GPU를 사용하고 있다. 당시에 사용한 GPU는 엔비디아 社에서 나온 GTX580을 사용했다. GTX580은 3GB의 메모리를 갖고 있기 때문에, 망의 구조를 결정하는데도 3GB의 메모리 한계에 맞춰 하였다.
- 이후에 발표된 GTX Titan X 같은 GPGPU processor를 사용하면, 더욱 더 발전된 구조를 설계할 수 있었을 것 같다. 아래 그림은 엔비디아 사가 밝힌 자신들의 GPU 상대 성능을 비교한 것이며, GTX580 대비 높은 성능을 보이는 모델이 많이 있음을 알 수 있다.

#### 3.2. AlexNet Details

- AlexNet의 블록도를 이해하려면, CNN은 아래 그림과 같이 3차원 구조를 갖는다는 것을 이해해야 한다. 영상의 크기를 나타내는 width와 height 뿐만 아니라 depth를 갖는다. 보통 color 영상의 경우는 R/G/B 3개의 성분을 갖기 때문에 시작이 3 이지만, convolution을 거치면서 feature-map이 만들어지고 이것에 따라 중간 영상의 depth가 달라진다. 이것을 이해하면, AlexNet의 블록도에 있는 숫자의 의미가 파악이 가능하다.
- LeNet의 경우는 입력 영상의 크기가 32 x 32로 작았기 때문에 모든 convolutional layer는 동일하게 5 x 5 크기를 갖는 kernel을 사용했다. 입력 영상도 흑백 영상이기 때문에 최초의 depth는 1이고, 이것이 convolution을 거치면서 depth가 증가하게 된다. 하지만 AlexNet의 경우는 입력 영상의 크기가 227 x 227 x 3으로 영상의 크기가 매우 크기 때문에, 첫번째 convolutional layer의 kernel의 크기가 11 x 11 x 3 크기로 비교적 큰 receptive field를 사용하고 있다.
  - 첫번째 convolutional layer에서는 stride의 크기를 4를 적용하였으며, 96개의 feature-map을 생성하기 때문에 결과는 55 x 55 x 96이 된다. AlexNet의 첫번째 convolutional layer는 55 x 55 x 96 = 290,400개의 neuron으로, 각 kernel은 11 x 11 x 3 = 363개의 weight 및 1개의 bias를 변수로 갖기 때문에  kernel 당 364개의 parameter이고, kernel이 96개이므로 364 x 96 = 34,944의 free parameter (LeNet 전체의 절반이상), connection의 숫자도 290,400 x 364 = 105,750,600으로 첫번째 layer에서만 1억개 이상의 connection이 만들어진다. 첫번째 convolutional layer를 거치게 되면, 96개의 feature-map을 얻을 수 있으며, GPU-1에서는 주로 컬러와 상관없는 정보를 추출하기 위한 kernel이 학습이 되고, GPU-2에서는 주로 color에 관련된 정보를 추출하기 위한 kernel이 학습이 된다.
  - 두번째 convolutional layer는 5 x 5 x 48 크기를 갖는 kernel을 사용하고 있으며, 첫번째 convolutional layer 뒤에 바로 두번째 convolutional layer가 오는 점이 다르다. 이는 [Part Ⅳ. CNN] 3. convolution Layer [1] 에서 살펴본 것처럼, 첫번째 convolutional layer에서 연산의 수를 줄이기 위해 stride를 1로 하지 않고, 4로 적용을 했기 때문에 자연스럽게 pooling을 한 것처럼 영상의 크기가 줄어 들었기 때문이다. 세번째 convolutional layer 연산을 하기 전에 response normalization과 pooling 과정을 거쳐 영상의 크기를 27 x 27 x 256으로 줄어들게 되며, 3 x 3 x 256 크기를 갖는 kernel을 사용하여 convolution연산을 수행하고 384개의 feature-map을 얻는다. 이 때 GPU-1과 GPU-2의 결과를 모두 섞어 사용을 한다. 그 결과에 대해 response normalization과 pooling을 거쳐 13 x 13 x 384크기의 영상을 얻으며, 그 결과에 대해 4번/5번 convolution이 블락도와 같이 적용이 된다.
- 최종 convolution을 거친 영상은 pooling 과정을 거쳐 총 4096개의 fully connected net에 연결이 되고, 최종단은 1000개의 category에서 결과를 낼 수 있도록 softmax 함수가 적용이 된다.

#### 3.3. AlexNet 성능 개선을 위한 고려 (ReLU)

- AlexNet이 성능 향상을 위해 좀 더 고려한 부분은 ReLU, overlapped pooling, response normalization, dropout 및 2개의 GPU 사용이라고 볼 수 있다. 이들에 대한 상세한 내용은 내용이 길어질 것 같아, 이번 class에서는 ReLU에 대해서만 살펴보고, 나머지는 다음 class에서 살펴볼 예정이다.
- 앞서 신경망을 처음 설명하면서, 신경망의 활성함수로는 biological neuron을 모델링하기 위한 nonlinear 활성함수로 sigmoid 함수를 사용한다는 것을 설명하였다.
- 하지만 sigmoid 함수는 hyperbolic-tangent 함수인 tanh에 비해서 학습 속도가 느리며, hyperbolic-tangent 함수 tanh 역시 학습 속도가 느린 문제가 있다. 망의 크기가 작을 때는 그 차이가 심각하지 않지만, AlexNet과 같이 망의 크기가 엄청나게 큰 경우는 학습 속도에 치명적인 영향을 준다. Xavier Glorot 등이 쓴 논문 “[Deep Sparse Rectifier Nerual Network](https://deeplearningworkshopnips2010.files.wordpress.com/2010/11/nipswrkshp2010-cameraready.pdf)”에 자세한 내용이 있으니 참고하면 좋을 것 같다.
- 그래서 논문에서는 활성함수로 ReLU(Rectified Linear Unit)을 사용했다. 이는 전자공학에서 반파정류 회로와 비슷한 개념이며, 0에서 미분이 안 되는 문제가 있기는 하지만, 학습속도가 탁월하고 back-propagation 결과도 매우 단순하기 때문에 요즘 Deep Neural Network에서는 거의 ReLU를 선호하고 있다. 논문에서도 실제 실험 결과가 나오는데, 학습속도가 sigmoid나 tanh를 사용했을 때에 비해 학습 속도가 6배 정도 빨라지는 것을 볼 수 있다. ImageNet에는 엄청난 수의 학습 이미지가 있기 때문에 고속으로 학습을 하기 위해서는 ReLU를 선택하였다. 이후에도 여러 논문들에서 자신들의 CNN 망에서 1장의 이미지를 학습하는데 걸리는 시간을 밝히고 있다. 1장의 영상을 학습시키는데 1ms 정도씩만 더 걸리더라도, 수천만장의 영상을 학습시킬 때 걸리는 시간은 엄청나게 벌어질 수 있다. 그러므로 활성 함수의 선택은 매우 중요하다.
- 이번 class는 AlexNet의 기본 구조에 대해서 살펴보았다. 다음 class에서는 AlexNet에서 성능을 개선하기 위해 적용된 부분이 어떤 의미가 있는지 상세하게 살펴볼 예정이다.

### 3.  AlexNet [2]

- AlexNet의 설계자가 성능 향상을 위해 좀 더 고려한 부분은 ReLU, overlapped pooling, response normalization, data augmentation, dropout 및 2개의 GPU 사용이라고 볼 수 있다. 지난 class에서는 위항목들 중 ReLU에 대해서만 살펴보았다. 이번 class에서는 나머지 부분에 대해 살펴볼 예정이다.

#### 3.4. Overlapped Pooling

- CNN의 구조에서 일반적으로 pooling은 convolution을 통해 얻은 feature-map 영상의 크기를 줄이기 위한 용도로 사용하며, average pooling 또는 max pooling을 사용한다.

- LeNet-5에서는 average pooling 방식을 사용했지만, AlexNet에서는 max pooling을 사용하였으며, 아래 그림의 화살표 영역이 pooling layer에 해당된다. 

  ![그림]()

  - average pooling은 pooling window 내에 있는 픽셀들의 평균을 취하는 방식
  - max pooling은 window에서 최대값을 갖는 픽셀을 선택한다. 
    - max pooling은 average pooling에 비해 최대값을 구해야 하기 때문에 연산량이 더 많지만, 최대 크기를 갖는 자극만 전달한다는 관점에서 보면, 생물학적인 특성과 좀 더 유사하다고 볼 수 있다. 

- 통상적으로 pooling을 할 때는 

  - 겹치는 부분이 없게 하는 것이 대부분이며
  - pooling window의 크기도 2x2를 주로 사용하고
  - 건너뛰기(stride)도 2를 사용하기 때문에 -> 출력 영상의 크기가 가로/세로 각각 1/2로 줄어들게 된다.

- 하지만 AlexNet은 2x2 윈도우 대신 3x3 window를 선택하고, 건너 뛰기를 2로 하는 overlapped pooling 방식을 사용했다. 

  - 이 overlapped pooling 방식을 통해 top-1과 top-5 에러율을 각각 0.4%와 0.3% 줄일 수 있었으며, overfitting에 빠질 가능성도 더 줄일 수 있다고 주장을 하고 있다.

#### 3.5. Local Response Normalization

- 활성함수로 sigmoid나 tanh를 쓰는 것보다 ReLU를 쓰게 되면,
  - (1) **학습속도를 줄일 수 있고 back propagation이 간단**해지는 이점에 대해서는 지난 class에서 설명하였다.
  - sigmoid나 tanh의 경우는 saturation 되는 구간이 있기 때문에, overfitting을 피하기 위해 입력에 대한 정규화(normalization)을 수행한다.
- 하지만 ReLU를 사용했을 때, 덤으로 얻을 수 있는 효과가 하나 더 있는데, 이것은 **입력의 normalization이 필요가 없다**는 점이다.
- 하지만, 출력은 ReLU 특성 곡선에도 알 수 있듯이 입력에 비례하여 그대로 증가가 된다.
  - 여러 feature-map에서의 결과를 normalization을 시키면, 생물학적 뉴런에서의 lateral inhibition(강한 자극이 주변의 약한 자극이 전달되는 것을 막는 효과)과 같은 효과를 얻을 수 있기 때문에 generalization 관점에서는 훨씬 좋아지게 된다.
  - 논문에서는 첫번째와 두번째 convolution을 거친 결과에 대하여 ReLU를 수행하고, max pooling을 수행하기에 앞서 response normalization을 수행하였다. 이를 통해 top-1과 top-5 에러율을 각각 1.4%와 1.2%를 개선하였다.

#### 3.6. Overfitting에 대한 해결책

- AlexNet의 free parameter의 개수는 6000만개 수준이기 때문에 이렇게 많은 free parameter를 처리하다 보면 overfitting때문에 고생한다.
- 이 문제에 대한 해결책으로 학습 영상의 수를 늘리기 위한 data augmentation과 망의 일부를 생략하면서 학습을 진행하는 dropout 방법을 사용하였다. 
  - Overfitting에 대한 개념과 일반적인 해결법에 대한 설명은 [Part Ⅲ. Neural Networks 최적화] 1. Overfitting ~ 4. Dropout  에 자세하게 하였으니, 좀 더 상세한 설명은 그 부분을 참고하면 된다.

##### 3.6.1. Data Augmentation

- Overfitting 문제에 대한 대표적인 해결책 중 하나가 학습에 사용할 데이터의 양을 늘리는 것이다.
  - 그러나 학습 데이터를 늘리는 것이 쉽지 않으며, 학습 데이터가 늘어나면 학습 시간이 길어지기 때문에 효율성을 반드시 고려해야 한다.
- AlexNet에서는 작은 연산만으로 학습 데이터를 늘리는 2가지 방법을 적용하였다.
- 특히 GPU가 이전 이미지(학습 데이터)를 이용하여 학습하고 있는 동안에, CPU에서는 이미지를 늘리기 때문에 디스크에 저장할 필요가 없도록 하였다.
- 첫번째 방법은 ILSVRC의 256x256 크기의 원영상으로부터 무작위로 224x224 크기의 영상을 취하는 것이다.
  - 이렇게 되면, 1장의 학습 영상으로부터 2048개의 다른 영상을 얻을 수 있게 된다.
  - test를 할 때는 5개의 224x224 영상(상하좌우 코너 및 중앙으로부터)과 그것들을 수평으로 반전한 이미지 5개, 총 10개로부터의 softmax 출력을 평균하는 방법으로 택하였다.
- 두번째 방법은 각 학습 영상으로부터 RGB 채널의 값을 변화시키는 방법을 택하였다.
  - 이를 위하여 학습 이미지의 RGB 픽셀 값에 대한 주성분 분석(PCA)를 수행하였으며,
  - 거기에 평균은 0, 표준편차는 0.1 크기를 갖는 랜덤 변수를 곱하고 그것을 원래 픽셀 값에 더해주는 방식으로 컬러 채널의 값을 바꾸어 다양한 영상을 얻게 되었다.
- 이렇게 data augmentation을 통하여 overfitting을 줄이고 top-1 error에 대하여 1% 이상 에러율을 줄일 수 있게 되었다.

##### 3.6.2. DropOut

- Dropout에 대한 논문이 발표된 이래, 요즘 대부분의 CNN 구조에서는 Dropout을 적용하여 학습시간을 단축시키고 overfitting의 문제도 해결한다.

- Dropout은 [Part Ⅲ. Neural Networks 최적화] 4. Dropout 에서 살펴본 것처럼, voting 효과 및 co-adaptation을 피하는 효과를 얻을 수 있기 때문이다.

- Krizhevsky와 공동 저자인 Hinton이 첫번째 저자로 발표한 2012년의 논문 “Improving neural networks by preventing co-adaptation of feature detectors” 논문도 한번 살펴보면 좋을 것 같다.

- 아래의 그림은 Hinton의 논문에서 dropout을 hidden layer에 대해서만 적용할 때와 입력단과  hidden layer 양쪽에 dropout을 적용했을 때를 비교한 것이다.

  ![그림]()

- Dropout은 그 성격상 fully-connected layer에 대하여 행하기 때문에 

  - AlexNet에서는 fully connected layer의 처음 2개 layer에 대해서만 적용을 하였다.
  - 또한 dropout의 비율은 50%를 사용하였다.

- 이번 class에서는 지난 class에 이어 AlexNet에서 성능을 개선하기 위해 적용된 부분이 어떤 의미가 있는지 살펴보았다. GPU를 사용한 부분에 대해서는 아무래도 좀 더 상세하게 살펴보는 것이 좋을 것 같아, 다음 class에서 살펴볼 예정이다.

[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

### 3. AlexNet [3]

- 지난 [Part Ⅴ. Best CNN Architecture] 3. AlexNet [1] 와 3. AlexNet [2] 을 통해,
  - AlexNet의 구조 및 성능 향상을 위해 AlexNet의 설계자들이 고려한 부분에 대해서 살펴보았다.
- 이번 class에서는 성능 향상을 위해 고려한 마지막 부분으로 multi-GPU 사용에 대한 내용을 살펴볼 예정이다.

#### 3.7. AlexNet의 성공과 GPU에 대한 관심 증가

- 2012년 AlexNet의 성공은 CNN/DNN 연구자들의 관심을 GPU 쪽으로 돌리는 계기가 되었다.  그리고 이는 대량 연산이 필요한 범용 프로세서를(GPGPU) 확산시키고자 하는 GPU 설계 업체, 특히 엔비디아에게는 아주 반가운 소식이 된다.
- 엄청난 연산량을 충족하기 위해 
  - GPU와 같이 대량 연산 장치를 갖고 있어야 하고,
  - parallelism을 절대적으로 필요로 하는 application이 만들어졌고,
  - 그것도 아주 매혹적인 분야에서 놀랄 만한 연구 결과가 나왔기 때문에,
  - -> GPU를 이용한 DNN(Deep Neural Network)에 대한 연구가 불붙게 된다.
- 엔비디아가 중심이 되어 DNN 분야에서 GPU를 효율적으로 사용하기 위한 cuDNN(CUDA Deep Neural Network) 라이브러리를 발표하였으며, 많은 연구자들을 포럼으로 끌어들였다.
- 아래 그림은 cuDNN 블로그 사이트에 있는 그림으로, 이것을 보면 CNN 분야의 연구 동향을 알 수가 있다.
- ![그림]()
- 위 그림을 보면, ILSVRC 참가자들 중 GPU를 이용한 참가자는 2012년에는 AlexNet 설계자들 뿐이었지만 (녹색 막대 그래프), 2013년 2014년을 보면 대부분의 참가자들이 GPU를 사용한 것을 알 수 있으며, GPU 사용으로 인해 알고리즘의 성능을 나타내는 에러율도(꺽은선 그래프) 엄청나게 줄어들게 된다.

#### 3.8. Why GPU ?

- Q. 그런데 왜 GPU가 필요한 것일까?

- A. 답은 연산량 때문이다.

- CNN을 이용하여 좋은 결과를 내려면, 엄청난 학습 데이터를 필요로 한다. ILSVRC에 있는 수천만장의 영상을 학습하려면, 엄청난 연산 능력이 필수적이다. 

  - 이세돌과 바둑 대결을 펼친 알파고가 엄청나게 많은 碁譜(기보)를 학습하고 빠른 연산을 위해 1920개의 CPU와 280개의 GPU를 쓰는 이유도 그것 때문이다.

- AlexNet의 주 개발자인 Krizhevsky는 토론토 대학을 졸업하고 구글에 입사하였으며, 그가 2014년에 발표한  “One weird trick for parallelizing convolutional neural networks” 논문을 보면, 요즘의 Deep CNN은 크게 2개로 구성이 되었다고 볼 수 있다.

  > 전체 연산량의 90~95%를 차지하지만, free parameter의 개수는 5% 정도인 convolutional layer
  > 전체 연산량의 5~10%를 차지하지만, free parameter의 개수는 95% 정도인 fully connected layer

- 여기서, convolutional layer는 구조적으로 아주 좋은 병렬적인 특징(parallelism)을 갖는다.

  - 앞선 class에서도 살펴보았듯이, 입력 이미지에 대해서 픽셀의 위치를 옮겨가면서 반복적으로 matrix multiplication이 필요하며
  - 여러 개의 입력 feature-map으로부터 동일한 parameter를 갖는 filter 연산을 해야 되기 때문에

- 또한 fully-connected network는 data 관점에서의 병렬성은 아니지만 model 관점에서의 병렬성이 좋다. 서로 다른 연산 유닛이 다른 모델을 수행할 수 있다는 뜻이다. 그래서 CNN의 layer에 따른 병렬성을 구별하면 아래와 같다.
  ![그림]()

- Convolutional layer에서는 여러 개의 입력 feature-map으로부터 여러 개의 filter 연산을 수행하기 때문에

  - 높은 수준의 data parallelism 이 존재하며,

- filter 연산 하나만 놓고 보더라도 행렬 곱(matrix multiplication)을 해야 하는데,

  - 이 때 행렬의 여러 component들은 연산 장치만 충분히 있다면 동시에 개별 곱을 구하면 연산을 더 효과적으로 할 수 있다.

- 즉, low-level 연산 parallelism도 존재하며, 입력 데이터의 크기가 커지거나 convolution kernel의 크기가 커지면 연산의 수는 기하급수적으로 증가하게 된다.

#### 3.9. CPU vs GPU ?

- 아래 그림은 엔비디아가 발표한 자료에 있는 내용으로, CPU와 GPU를 사용할 때의 극명한 성능 차이를 보여준다.
- ![그림]()
- 최근 사용되는 GPU 들은 
- 대부분 Shader 기반 구조를 갖고 있고,
  - 엄청난 수의 병렬 연산 유닛을 갖고 있으며,
  - floating point 연산 장치도 훌륭하고,
  - 특히 메모리 억세스의 BW도 높고 속도도 엄청 빠르기 때문에
  - -> 이런 low-level parallel 연산은 범용 CPU와 비교했을 때는 엄청난 성능 차이를 보이게 된다.
- 그럼 CPU만으로는 CNN을 구현할 수 없는 것일까?
  - 그렇지는 않다.
  - 굳이 GPU를 사용하지 않더라도, 학습 시간이나 연산시간이 그리 길지 않은 작은 모델인 경우는 CPU를 써도 무방하다.
- 하지만, 학습 데이터의 수가 증가하게 된다면, 최적의 학습 결과 도출에 걸리는 시간상의 문제로 인해 GPU를 사용하는 것이 바람직하다.
- ![그림]()
  - 위 그림을 보면 GPU로 하루 걸리는 일이 CPU로 14일이 걸릴 수도 있으며, 만약에 GPU로 일주일이 걸린다면, CPU로는 3개월 이상이 걸릴 수도 있다. 앞서 study 했듯이, CNN에서의 최적의 구조 결정을 위한 hyper-parameter가 매우 많다는 점을 고려하면,
    대용량 데이터를 CPU만으로 최적의 학습 결과를 낸다는 것은 기대하기 어려울 것 같다.

#### 3.10. AlexNet에서의 GPU 활용

- AlexNet은 2개의 GPU를 사용하였다. 
  - 당시는 GTX580 GPU를 사용했는데, 지금은 더 고성능의 GPU들이 더 많이 나와있으며, 8개 이상의 GPU를 사용하는 연구들도 많이 볼 수 있다.
  - AlexNet 개발 당시 사용한 GTX580은 3GB의 메모리를 갖고 있었기 때문에 모델의 구조에 제약이 있을 수밖에 없어 보인다.
  - 이는 그때 당시 학교에서 진행하는 연구였기 때문에 쉽게 구할 수 있는 플랫폼을 선택했을 것으로 추정하며,
    2개의 GPU를 사용하는 구조를 설계하면서 많은 고민이 있었을 것으로 보인다.
  - 또한 제한된 플랫폼이기 때문에 GPU를 위/아래로 나눠쓰는 방식을 취했지만, 이후 연구들에서는 이런 식의 구조를 취한 사례가 별로 없다.
    이는 더 고사양의 플랫폼을 쓰면 되기 때문인 것으로 추정된다.
- 아래 그림은 AlexNet의 구조를 GPU 사용의 관점에서 본 것이다.
- ![그림]()
- GPU의 개수나 GPU에 있는 메모리 크기를 고려하지 않고 구조를 결정한다면
  - 첫번째 convolutional layer에서 1개의 GPU에는 color와 무관한 48개의 feature-map을 추출하고,
  - 다른 GPU에서는 color와 관련된 48개의 feature-map을 추출하는 등의 제한을 두지는 않았을 것이다.
- 또한 inter-GPU와 intra-GPU connection 역시 크게 신경을 쓰지 않았을 것으로 보이며,
- resource를 고려하지 않은 구조 설계를 했다면, 더 좋은 결과가 나왔으리라고 추정이 된다.
- 실제로 이후 연구들이 여러 개의 GPU를 사용하거나 더 高仕樣의 GPU를 사용하여 성능을 크게 개선시킨 논문이 뒤에 발표가 되었다. 
- AlexNet 연구자들은 2개의 GPU를 사용하여, 1개의 GPU를 사용했을 때보다 top-1과 top-5에러를 각각 1.7%와 1.2% 절감할 수 있었다고 주장을 하고 있다.

#### 3.11. AlexNet - 결과

- AlexNet의 실험 결과를 보면, 아래 그림처럼, 상당히 결과가 괜찮은 것을 알 수가 있다.
- 그림에서 올바른 결과는 그림 바로 밑에 적혀 있고, 그 밑에 있는 5개의 후보는 AlexNet이 추정한 것이다.
  - Mite(진드기)는 한쪽으로 치우쳐 있어도 잘 구별을 하는 것을 알 수가 있으며 leopard에 대한 추정 5개도 거의 비슷하게 보이는 것들이기 때문에 엉터리는 아니다.
  - 또한 추정이 틀린 경우에도 중앙에 있는 것들을 어떻게 볼 것인지에 따라 충분히 추정이 가능한 답을 했음을 알 수 있다.
  - ![그림]()
  - 이런 결과가 SIFT(Scale Invariant Feature Transform)와 같은 feature extractor를 사용하지 않고, CNN 학습을 통해서 얻어진 결과라는 점이 더욱 고무적이다.
  - 2012년의 대회에서 실제로 SIFT 등을 사용한 다른 참가자들의 경우, 성능이 떨어졌다. 
  - 이는 CNN을 아주 복잡한 일반 영상에 대해서도, 학습의 양이 충분하다면, 그리고 좋은 CNN 구조를 갖는다면,충분히 좋은 결과를 낼 수 있다는 가능성을 보여줬다는 점에서 매우 큰 의미를 갖는 것 같다.

#### 3.12. 첨언

- GPU를 사용한 CNN은 아직 뜨겁게 연구가 되고 있는 분야라서, 짧은 지면으로 전부 소개를 한다는 것은 말이 안 되는 것 같다. 
- 하지만, 지금은 Best CNN 구조가 어떻게 변하고 있는지를 살펴보는 단계이니, 이 정도로 간단히 소개를 하고, CNN 구조에 대한 비교가 끝나면, 따로 자세히 GPU, CUDA, cuDNN, GPU를 이용한 병렬 컴퓨팅 등을 살펴보기로 하자. 다음 class에서는 ZFNet의 구조에 대하여 살펴볼 예정이다.

### 4. ZFNet [1]

#### 4.1. AlexNet 이후의 후속 연구와 ZFNet

- 2012년 ILSVRC를 우승한 AlexNet 에 대한 소식은 관련 분야의 연구자들에게는 엄청난 자극과 희망을 가져다 주게 된다. 
  - 이것은 마치 인류의 한계라고 여겼던 100m 달리기 기록이 누군가에 의해 한번 깨지게 되면, 그 다음에는 그 기록을 깨는 사람들이 많이 나오는 것과 유사하다.
- 그리하여 AlexNet은 이후 새로운 구조나 방식을 개발하는 연구자들이 성능을 비교할 때 사용하는 일종의 reference 역할을 하게 된다.
  - 2012년 ILSVRC에서 AlexNet은 classification error 율이 16.4%로 우승을 했지만, 
  - 2013 년 ILSVRC에서는 
    - Clarifai가 11.7%의 에러율을 보이며 우승을 하게 된다. 
    - 그리고 AlexNet 보다 좋은 결과를 보이는 팀도 총 8개가 된다.
    - Clarifai?
      - Clarifai는 뉴욕대에서 박사 학위를 받은 Matthew Zeiler가 2013년에 세운 회사
      - 영상 인식이나 인공지능 분야의 솔루션을 공급
      - Matthew Zeiler
        - 뉴욕대를 다니는 동안 CNN을 보다 잘 이해할 수 있는 “Visualizing 기법”을 최초로 개발
        - 2011년에 “Adaptive Deconvolutional Networks for Mid and High-level Feature Learning” 이라는 논문을 발표
        - 이후 2013년에 유명한 논문 “Visualizing and Understanding Convolutional Networks”라는 논문을 발표
        - 2013년 ILSVRC에도 참가하여 classification error율 13.5% 를 보인다.
        - 아마도 학교에 있을 때, 
          - 이미 Visualizing 기법을 활용한 CNN 구조 연구에 이미 정통해 있었고,
          - 2012년 Krizhevsky가 AlexNet을 발표하자, 그 연구 결과에 자신들의 연구를 적용하여 더 좋은 구조를 찾아낼 수 있게 되었고, 그 다음 해에 ILSVRC에 참가한 것으로 보인다.
        - Matthew Zeiler는 결과적으로 2개의 팀으로 참가한 것이나 마찬가지 상황이 되지만, 이미 AlexNet을 능가할 수 있는 아이디어는 꾸준히 발전을 시켰을 것이라 추정이 되고, 이에 고무되어 회사를 설립하고 ILSVRC에는 자신의 회사도 참여하여 우승을 하게 된다.
      - Clarifai의 구조에 대하여 상세 소개한 자료는 거의 없다.
    - Zeiler의 ZFNet
    - Clarifai의 구조에 대해 상세히 소개한 자료는 거의 없는 반면 Zeiler가 발표한 ZFNet은 소개가 잘 되어 있으며, 
    - 개념적으로도 같은 연구자가 개발하여 脈(맥)이 같을 것으로 추정된다. 
    - CNN의 좋은 구조를 찾아가는 것이 우리 class의 목적이기 때문에, 이번 class에서는 ZFNet에 대하여 살펴보기로 한다.

#### 4.2. ZFNet이란 ?

- CNN, 특히 layer가 여러 개인 DNN의 경우, 
  - 무슨 원리로 그렇게 좋은 결과를 낼 수 있는지,
  - CNN의 구조를 결정하는 hyper-parameter는 어떻게 설정을 할 것인지,
  - 또는 자신들이 개발한 구조가 과연 최적인지 판가름하는 것은 너무 어렵다.
- AlexNet의 경우 2개의 GPU를 사용하여 학습을 하는데 일주일 이상이 걸리는데, 10개가 넘는 hyper-parameter의 조합 중 최고를 찾아낸다는 것아 과연 가능한 일일까?
- 좋은 결과는 얻었지만, 
  - 이론적으로 설명하기가 어렵고,
  - 최적의 구조인지도 확신할 수도 없는 상황이기 때문에 뭔가 CNN을 보다 잘 이해할 수 있는 수단이 필요했으며, Matthew Zeiler는 이것을 “Visualizing 기법"을 사용하여 해결하려는 시도를 하였다.
- 이것에 관련된 주요 개념은 2011년에 이미 발표를 하였으며, 2012년 AlexNet의 결과가 reference로 만들어지자, 2013년 논문 발표를 통해 자신들의 연구 결과가 효과적임을 입증하였다. 이후 많은 연구자들이 Zeiler의 논문에 자극을 받아 “Visualizing 기법”을 변형/발전 시키는 논문을 발표하게 된다. 
- 이 중 2개만 소개하면 아래와 같다.

>1. Visualizing Convolutional Neural Networks for Image Classification (2015, Danel Brukner)
>2. Delving Deep into Convolutional Nets (2014, Ken Chatfield)

- 어찌보면, ZFNet은 특정구조를 가리키는개념이 아니라, CNN을 보다 잘 이해할 수있는기법을 가리키는개념으로 이해하는 것이 좋을것 같다.
- 실제 논문에서도 새로운 구조를 발표했다기 보다는 AlexNet의 기본 구조를 자신들의 Visualizing 기법을 통해 개선할 수 있음을 보여줬다.
- 결과적으로 보면, 많은 수의 hyper-parameter를 최적화 함에 있어 시간이나 기타 다른 이유로 인해, AlexNet 연구자들이 더 좋은 구조를 설계할 수 있음을 보임으로써 자신들의 연구의 탁월함을 보여준 셈이 된다.

#### 4.3. ZFNet 속으로 ? Deconvolution을 이용한 Visualization

- CNN의 동작을 잘 이해하려면, 중간 layer에서 feature의 activity가 어떻게 되는지를 알 수 있어야 한다.
- 그런데 중간 layer의 동작은 보기가 어려우니, 이 activity를 다시 입력 이미지 공간에 mapping을 시키는 기법이 바로 “Visualizing 기법”의 핵심이 된다.
- CNN 구조는 
  - 여러 개의 convolutional layer를 기반으로 하고 있는데, 
  - 중간 단계 layer에서는 이전 feature map으로부터 데이터를 받아,
  - 그것에 대해 filtering(convolution)을 수행하여 현단계의 feature-map을 만들어내고,
  - 정류(ReLU) 장치를 통과 시킨 후, 
  - Pooling을 수행하는 아래 그림과 같은 방식을 취한다.
- 특정 feature의 activity가 입력 이미지에서 어떻게 mapping 되는지를 이해하려면 위 과정을 역(reverse)으로 수행하면 된다. 그런데 여기에서 가장 문제가 되는 부분이 max-pooling 에 대한 역(reverse)를 구하는 것이다.
  - Max-pooling 단계는 주변에서 가장 강한 자극만을 다음 단으로 전달을 하기 때문에, 역방향으로 진행을 할 때는 어떤 위치에 있는 신호가 가장 강한 신호인지 파악할 수 있는 방법이 없다.
- 여기에서 ZFNet 개발팀의 번득이는 아이디어가 들어가게 된다. 이 사람들은 “switch”라고 명명한 개념을 생각해냈다.
  - 이 switch는 가장 강한 자극의 위치 정보를 갖고 있는 일종의 꼬리표(flag)라고 보면 된다. 가장 강한 자극의 위치 정보를 갖고 있기 때문에, 역으로 un-pooling을 수행할 때는 switch 정보를 활용하여 가장 강한 자극의 위치로 정확하게 찾아갈 수 있다.
    - 원래는 가장 강한 자극 이외의 값들도 있었지만, max-pooling을 수행하면서 사라졌기 때문에, 이 방법의 단점은 가장 강한 자극(feature)에 대한 영향만 파악이 가능하다는 점이다.
  - 아래 그림을 보면, 녹색 화살표가 가리키는 자극은 강하지 않기 때문에 max-pooling을 수행하면서 오른쪽 빨간색 화살표처럼 사라진다.
    - ![ㅇ]()
  - 다음 과정은 ReLU의 역과정을 수행하여야 하는데, ReLU의 경우 앞선 클래스에서도 살펴봤지만, 음의 값을 갖는 것들은 0으로 정류(rectify) 하고 0이상 값들은 그대로 pass  시키기 때문에, 역과정을 수행할 때 양의 값들은 문제가 없고, 0 이하의 값들만 정류되었기 때문에 복원할 방법이 없다. 그렇지만 이들의 연구 결과에 따르면 영향이 미미 하다고 한다.
  - 다음 과정은 filter에 대한 inverse filter 연산을 수행하는 것이다. 이 과정은 가역이 가능한 과정이기 때문에 특별히 문제가 될 것이 없다.
  - 이렇게 Deconvolution을 수행하면, 정확하게 입력과 같은 상태로 mapping이 되는 것은 아니지만 
    - 가장 강한 자극(feature)가 입력 영상에 어떻게 mapping되는지를 확인할 수 있으며,
    - 이렇게 결과를 시각적으로 확인을 할 수 있기 때문에 CNN에 대하여 더 잘 이해를 할 수 있고,
    - 한 걸음 더 나아가 어떤 구조가 최적의 구조인지를 좀 더 잘 결정할 수 있게 된다.

#### 4.4. Feature Visualization

- 그럼 실제로 어떻게 나타나는지 좀 더 살펴보자.
- 학습 데이터를 이용하여 training을 마치게 되면, 각 layer에서의 자유 변수가 결정이 된다.
- AlexNet처럼 5개의 convolutional layer를 갖는 CNN 구조를 생각해보자.
- 각각의 단계의 feature-map에서 가장 강한 특징 9개만 보여주고, 그 9개의 feature에 대해 원본 영상을 같이 쌍으로 보여주기로 한다.
- 아래 그림은 Layer1과 Layer2를 보여주는 것으로서, 주로 영상의 코너나 edge 혹은 컬러와 같은 low level feature를 시각화 한 것이다.
  - ![ㅇ]()
- 다음은 Layer3에 대한 그림이다.
  - ![ㅇ]()
  - 이 그림을 보면, Layer1/Layer2에 비해 좀 더 복잡한 (상위 수준) 항상성(invariance)을 얻을 수 있거나
    비슷한 외양(texture)를 갖고 있는 특징을 추출할 수가 있다.
  - Layer4에서는 사물이나 개체의 일부분을 볼 수 있으며, 아래 그림의 경우처럼 개의 얼굴(1열, 1행)이나 새의 다리(4열 1행) 등을 볼 수 있다.
    - ![ㅇ]()
  - Layer5에서는 위치나 자세 변화 등까지 포함한 사물이나 개체의 전부를 보여 준다.
  - 이렇게 시각화(Visualizing)를 수행하게 되면, 
    - 특정 단계에서 얻어진 feature-map 들이 고르게 확보되어 있는지 혹은 특정 feature 쪽으로 쏠려 있는지,
    - 개체의 pose나 기타 invariance 등을 잘 반영하는지 등등을 파악할 수 있기 때문에,
    - 학습의 결과에 대한 양호 여부를 판단할 수도 있고, 결과적으로 CNN의 구조가 적절했는지 등을 판단하기에 좋다.
- 이번 class에서는 Matthew Zeiler 연구팀이 발표한 “Visualization 기법”의 필요성과 방법 및 기본 개념에 대하여 살펴보았다. 
  - CNN의 중간 layer를 시각화 하기 위해 Deconvolution을 사용하였으며, 가장 큰 난관인 max-pooling 결과에 대한 un-pooling 문제를 해결하기 위해 switch라는 개념이 적용되었음을 알았다.
  - 각각의 intermediate 단에서의 feature가 원영상에서 어떤 영역에 해당하는지, local feature등이 layer가 올라감에 따라 global feature로 바뀌는 것에 대해서도 좀 더 잘 이해할 수 있게 되었다.
  - 이 Visualization 기법이 만능 툴은 아니지만, 적어도 CNN 구조의 적합성 여부를 판단하는데 좋은 툴이 될 수 있음은 분명하다.
  - 다음 class에서는 이점에 대해 좀더 자세히 살펴보고, 이것을 통해 과연 AlexNet의 성능을 끌어올릴 수 있는지도 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

### 4. ZFNet [2]

- 지난 [Part Ⅴ. Best CNN Architecture] 4. ZFNet [1] 에서는 “Feature Visualization 기법”의 개념에 대하여 간략하게 살펴보았다.
  CNN은 여러 단계(convolution + pooling)의 과정을 거쳐 학습을 하게 되는데,
  각각의 단계(layer)에서 도대체 어떤 특징을 학습해 가는지를 시각화하여 볼 수 있다면,
  좀 더 나은 구조를 갖는 CNN을 설계할 수도 있다.
  오늘은 “Feature Visualization 기법”이 실제로 어떤 의미를 갖는지 좀 더 자세하게 살펴볼 예정이며,
  그간 feature extraction 알고리즘으로 최고의 성능을 갖는다고 알려진 SIFT와의 간략한 비교를 통해,
  CNN의 우수성(이미 Krizhevsky의 AlexNet이 2012년 ILSVRC에서 입증한 사실이지만)을
  다시 한번 살펴볼 예정이다.
  이번 class는 ZFNet의 설명에서 약간 벗어나기는 하지만,
  결과적으로 CNN을 좀 더 잘 이해할 수 있게 하는 것이 목적이니, 나름대로 의미를 갖는다.

#### 4.5. 이미지상의 Feature ?

만약에 디카로 찍은 영상들을 자동 분류(auto-tagging)를 하고 싶다면, 어떻게 해야 할까?
답은 영상을 분석하여 영상을 대표할 수 있는 “그 무엇(representation)”을 끄집어 내야 한다.
그렇다면 영상을 대표할 수 있는 “그 무엇(feature)”의 후보는 어떤 것들이 될까?
영상을 대표할 수 있는 특징은 크게 3가지로 분류가 가능하다.
l  Low-level feature: edge, corner, color
l  Mid-level feature: edge junction
l  High-level feature: object의 일부분이나 전체 object
?
영상을 대표할 수 있는 feature로는 흔히 사용하는 영상의 경계(edge)를 꼽을 수 있다.
영상의 경계는 우리가 알고 있는 Sobel, Laplacian, Canny 등
edge 검출 필터를 사용하면 쉽게 검출이 가능하지만, 그것이 무엇을 뜻하는지 파악하기란 매우 어렵다.
그렇기 때문에 많은 feature 검출 알고리즘들은 단순한 edge가 아니라
좀 더 강인한 특징을 갖는 edge를 추출하고, 거기에 histogram이나 기타 다른 방법을 통해,
좀 더 효과적으로 구별하려는 시도들을 한다.
좀 있다가 살펴볼 SIFT도 비슷하다.
아래 그림을 한번 보자.
이것은 지난 class에서 살펴본 그림의 일부이며, Zeiler가 자신의 논문에서 사용한 그림이다.
예를 들어, 아래 그림을 보면 서로 다른 개가 나오는데
이것을 edge에 기반한 feature 만으로 제대로 분류가 가능할까?
또한 다양한 각도에서 바라보는 자전거를 잘 분류해 낼 수 있을까?

ILSVRC 2012년 결과를 보면,
SIFT 기반의 결과와 CNN을 사용한 결과는 10% 이상의 성능차이를 보였으며,
2013년의 Zeiler의 Clarifai는 다시 2012년 결과를 5% 이상 개선을 하였기 때문에
이제는 CNN과 low-level feature 기반의 알고리즘을 비교 한다는 것 자체가 별로 의미가 없어졌다.

#### 4.6. SIFT(Scale Invariant Feature Transform)

- 그래도 한동안 대세로 자리를 잡았던 low-level feature 기반 영상 비교 알고리즘의 선두 주자였던 SIFT에 대해서 잠깐 살펴보자.
- SIFT는 
  - Scale Invariant Feature Transform의 약어로,
  - 크기의 변화나 회전에 영향을 받지 않는 영상의 feature를 검출하고 그것으로 꽤 괜찮은 성능을 보이는 알고리즘이다.
  - SIFT는 크게 보면, 다음 4개의 step을 거쳐 특정 object를 대표할 수 있는 descriptor를 생성한다.
    - 1. Scale-space extrema를 찾아낸다.
      2. Key-point localization & Filtering
      3. Orientation 할당
      4. Descriptor를 생성
  - SIFT 알고리즘을 설명할 때, 가장 많이 사용되는 그림이 바로 아래 그림이다.
  - ![pic]()
  - 영상에서 단순하게 경계를 검출하면, 잡음을 포함한 의미 없는 edge도 많이 검출하기 때문에
    - 먼저 Gaussian filter를 여러 레벨로 적용(σ 값을 조절)하여 영상을 blur 시키고,
    - 그것들 간의 차(difference)를 구하면, 
    - 잡음에 강인한 feature가 얻어진다. (DoG: Difference of Gauassian).
  - 이렇게 현재 scale에 대한 작업을 마치면, scale 변화에 대응이 가능하도록 영상의 scale을 바꿔가면서 동일한 작업을 수행하여, 서로 다른 scale(크기)에서의 강인한 feature(extrema)를 구한다. 
  - DoG를 통해 비교적 강인한 extrema를 추출했을지라도 아직 쓸모 없는 정보들도 많이 포함이 되어 있기 때문에 이것을 key-point localization 작업과 filtering을 통해 의미가 있는 정보들만 추출해낸다. 
  - 다음은 gradient의 방향에 따른 histogram을 구한다.
  - 다음은 4x4 gradient window에서 8개의 방향으로 4x4x8 = 128 총 128차원의 descriptor를 만들고, 이 descriptor를 비교하는 방식으로 영상의 정합성 여부를 판단하게 된다.
  - 결과적으로 보면, 영상의 mid나 high-level 특징을 보는 대신에 low-level gradient가 scale-space 에서 어느 방향으로 향하는 것인가에 대해서만 살피기 때문에 태생적으로 한계를 가질 수밖에 없다.
  - 조명의 변화에 대해서도 민감하며, 영상을 취득할 때 saturation 되는 부분에 따라서도 많은 영향을 받게 된다.
    SIFT를 소개하는 자료는 인터넷 상에 워낙 많기 때문에 이 정도로 간단하게 정리를 한다.

#### 4.7. CNN과 Feature Visualization

- CNN이 SIFT등 기존 알고리즘과 다른 점은 : 
  - 여러 단계의 convolution과 pooling을 거치면서 mid 및 high-level feature를 이용한다는 점이다.
  - 그리고 처리를 해야 할 영상의 형태나 내용과 관계 없이 동일한 방식을 따르는 것이 아니라, 데이터에 따라 convolution에 사용할 filter의 parameter를 학습을 통해 최적값을 결정한다는 점이 다르다.
- 하지만 최적의 결과를 얻으려면, 
  - 적절한 hyper-parameter의 설정이 필요하며, 
  - 이것을 직관적으로 이해하기가 쉽지 않다.
- “Feature Visualizing 기법”
  - CNN에서 최적의 구조를 결정하는데 크게 도움이 되는 수단을 제공한다.
    - 이러한 수단이 없다면 그냥 경험이나 직관에 의존을 해야 되지만, 
    - Feature Visualization 기법을 이용하면, 
      - 중간 단계의 filter 연산에서 적절한 feature들이 추출이 되었는지를 확인할 수 있으며,
      - 이를 통해 현재 실험 중인 hyper-parameter의 조합이 최적의 결과인지도 확인해낼 수 있다.
  - 논문에 있는 실험 결과를 살펴보자.
    - 그림에서 (a)는 Krizhevsky의 논문에 기반하여 얻은 첫번째 layer의 특징이며, 특정 feature가 많이 나타나는 것을 볼 수가 있다. 이는 filter kernel이 벡터 관점에서 보면 대표성이 떨어진다고 볼 수 있다.
    - (b)는 Zeiler가 좀 더 개선된 결과를 얻어내기 위해 Krizhevsky의 1단계 구조에서 stride와 filter kernel의 크기를 바꿔 개선한 것이다. (b)는 (a) 보다 훨씬 다양한 feature를 검출할 수 있다는 것을 확인할  수 있다.
    - 아래 그림에서 (c)는 Krizhevsky 논문의 두번째 layer를 Visualization 기법으로 본 결과이며, (d)는 ZFNet의 결과이다. 대충 봐도 알 수 있듯이, (d)는 (c) 보다 훨씬 선명하고, anti-aliasing artifact도 없음을 알 수 있다.
  - 이처럼, Feature Visualization 기법을 잘 활용하면, 최적의 architecture를 결정하는데 큰 도움을 받을 수 있다.좋은 feature는 topology나 scale 변화에 무관해야 한다.
  - 다시 처음에 사용한 그림으로 돌아가보자.
    - 다섯번째 layer에서 보면, 여러 형태 여러 위치에서 개(dog)를 잘 검출할 수 있다.
      - 이 또한 Visualization 기법을 통해 쉽게 확인이 가능
    - 또한 마지막 convolution 단계인 5단계를 거치면서 edge나 edge의 junction과 같은 low-level feature가 아니라 개의 얼굴이나 자전거의 바퀴처럼, object의 일부분이나 전체를 효과적으로 검출이 가능하다는 것을 확인할 수 있다.
  - 이번 class에서는 CNN이 다른 low-level feature 기반 알고리즘과 달리 장점을 갖는 이유를 Visualization 기법을 통해서 확인하였고, 또한 Visualization 기법을 통해 최적의 architecture를 결정할 때도 잘 활용할 수 있음을 살펴보았다. 다음 class에서는 Visualization 기법을 활용하여, Krizhevsky의 AlexNet의 성능을 끌어올린 구조에 대하여 살펴보고, 기타 Visualization 기법의 특징에 대하여 살펴볼 예정이다.

### 4. ZFNet [3]

- [Part Ⅴ. Best CNN Architecture] 4. ZFNet [1] 와 4. ZFNet [2] 을 통해 Feature Visualization의 개념을 살펴보고, CNN 구조를 잘 이해할 수 있게 되었다. [Part Ⅴ. Best CNN Architecture] 4. ZFNet [2] 에서 잠깐 언급을 하였지만, Zeiler는 Visualization 기법을 활용하여 Krizhevsky의 AlexNet 성능을 3% 이상 개선을 하였다.
- 이번 Class에서는 Zeiler가 개선한 ZFNet의 구조에 대하여 살펴볼 예정이며, 지난 Class에서 언급하지 않았던 Visualization을 이용한 CNN의 이해에 관련된 일부분을 살펴볼 예정이다.

#### 4.8. AlexNet 대 ZFNet

- 앞서 살펴본 AlexNet의 구조는 아래 그림과 같다.

  ![ㅇ]()

  - AlexNet :  
    - 2개의 GPU를 활용하는 구조
    - 필요에 따라 inter-GPU/intra-GPU 통신 구조를 취하고 있다.
  - Zeiler는 Visualization 기법을 활용하여, AlexNet의 구조의 문제점을 지적한다.
    - AlexNet 설계자들은 2개의 GPU에 구현하면서 많은 고민을 했을 것이라 생각되지만, 결국 Zeiler에 의해서 그들의 구조가 최적이 아니었음이 증명이 된다.
  - 위 그림의 (b)와 (d)는 AlexNet의 1/2단계 feature를 보여주는 것이고, (c)와 (e)는 ZFNet의 결과를 보여준다.
  - AlexNet보다는 ZFNet의 결과가 훨씬 더 좋다는 것은 그림 만으로도 쉽게 이해가 간다.
    - AlexNet은 일부 feature에 몰리는 현상도 있었고, aliasing 문제도 발생을 하였지만, ZFNet은 다양한 feature가 고르게 나타나는 것을 볼 수 있다. 당연히 AlexNet보다는 ZFNet을 이용한 결과가 좋게 나올 수밖에 없다.

- 그럼 아래 그림을 한번 보자. 아래 그림은 ZFNet의 구조이다.

- ![ㅇ]()

  - ZFNet도 AlexNet과 동일하게 GTX580 GPU를 사용하였다. 단, AlexNet과 달리, 2개의 GPU를 사용하지 않고, 1개의 GPU를 사용하였다. ILSVRC 영상을 할 때 70 epoch에서 멈추었으며, 1개의 GPU를 사용하여 총 12일 동안 학습을 시켰다고 한다.
  - ZFNet은 자신들의 Feature Visualization 기법의 효과를 보이는 것에 집중을 했기 때문에 전반적으로 AlexNet과 비슷한 구조를 취했다. 단, 첫번째와 두번째 layer는 구조가 다르다는 것을 알 수 있다. 아래 그림은 확대한 그림이다.
    - ![ㅇ]()
  - AlexNet은 첫번째 convolution layer에 11x11 크기에 stride를 4로 정했는데, ZFNet은 filter 크기를 7x7로 하고, stride를 2로 하였다. 그렇게 얻은 110x110 크기의 feature map에 대하여 3x3 overlapped max pooling을 수행하였으며, stride를 2로 하여 55x55 크기를 얻었다. 이 점은 AlexNet과 확연하게 다른 부분이다. 
  - [Part Ⅳ. CNN] 4. Convolution Layer [2] 에서 Hyper-parameter를 정하는 방법에 대해 소개를 했듯이, filter의 크기를 크게 하고 stride 값을 크게 하는 방식은 요즘 추세처럼 filter의 크기를 작게 하고 stride 크기를 작게 설정하는 것보다 결과가 좋지 못하다.
  - Zeiler는 Feature Visualization을 통한 이 부분을 밝혔으며, 더 좋은 결과를 얻을 수 있도록 AlexNet의 구조를 변형시켰다. 이후 과정은 AlexNet과 거의 대동 소이하다. 하지만 AlexNet의 경우처럼, 2개의 GPU에 mapping 시키기 위해 인위적으로 나눠서 처리하는 부분을 제거했다.
  - 그 결과 두번째 layer에 대한 Feature Visualization 결과 그림에서도 알 수 있듯이, 두번째 단계의 결과도 AlexNet보다 ZFNet이 좋다는 것을 확인할 수 있다. ZFNet을 통해서 증명을 했듯이 AlexNet의 구조처럼 2개의 GPU에서 다른 처리를 하는 시도는 그 후 다른 어떤 연구에서도 찾아 볼 수가 없다.

#### 4.9. Feature Visualization을 통해 확인 CNN에 대한 추가 이해 포인트

- Zeiler는 그의 연구를 통해 CNN에 대한 몇 가지 사항을 밝힌다.
- 우선 layer 별로 feature를 습득하는 시간이 다르다.
  - 아래 그림에서 볼 수 있는 것처럼, 앞쪽 layer들은 몇 번의 학습 epoch 만에 feature 들이 수렴하는 것을 볼 수가 있다.
  - 하지만, 
    - 뒤쪽 layer로 갈수록 feature 습득에 오랜 시간이 걸리며, 
    - 40~50 epoch 이상이 되어야 feature들이 보이기 시작한다.
  - 이 점은 CNN 학습의 epoch 수를 정할 때 참고하면 좋을 것 같으며, 실제로 ZFNet을 학습할 때는 70 epoch까지 진행을 했다고 밝히고 있다.
  - CNN의 개념 설명 당시 소개했던 것처럼, 크기, 위치 변화 및 회전 변화에 따른 invariance가 확보된다.
  - 아래 그림은 크기, 위치 및 회전 이동에 사용할 테스트 이미지 들이다.
    - 작은 변화에 대해서도 앞쪽 layer에서는 눈에 띌 정도의 변화가 있지만, 뒤로 갈수록 invariance를 얻을 수 있음을 실험으로 파악하였다.
    - 또한 위치 이동과 크기 변화에 대해서는 거의 linear한 특성을 얻을 수 있었다.
  - 다음 그림은 위 이동에 대한 결과이다.
    - 실험에 사용한 5 종류의 영상들에 대해서 각 layer에서의 유클리드 거리를 구한 것을 보여준다.
    - Layer가 뒤쪽으로 갈수록 invariance한 성질이 있음을 알 수 있다.
  - CNN이 영상 속에 있는 object(개체)의 위치까지 파악이 가능한가 아니면 주변에 있는 것들과의 관계 속에서 개체를 알아내는지도 실험을 통해 파악할 수 있다.
  - 결론부터 말하면, 개체의 정확한 위치까지 파악을 한다.
    이것을 확인하기 위해, 개체의 일부 영역을 사각형 형태로 회색 칠을 하고 결과를 확인하였다.
    물체를 가리면, 그 물체를 제대로 분류해낼 가능성이 많이 떨어지는 것을 확인할 수 있었다.
    아래 그림의 결과를 확인해보자.
    (a)는 실험에 사용한 그림들을 보여주며, 회색 사각형 영역을 옮겨가면서 실험을 한다.
    (b)와(c)는 layer5에서 가장 강한 activity에 대한 feature map이고, (d)와 (e)는 classifier의 출력이다.
    (b)는 가리는 부분에 따라 layer5 activation이 어떻게 달라지는가를 보여준다.
    Promeranian 그림의 경우 당연히 개의 얼굴을 가렸을 때 activation이 떨어지게 나타난다.
    (c)에서 검은 사각형으로 있는 그림은 가장 강한 activation을 input 공간에 mapping 시켰을 때를 보여주며,
    나머지 그림은 다른 이미지에서의 예이다.
    (d)는 classifier의 출력에서 특정 위치를 가렸을 때, 제대로 검출이 가능한지를 확률로 표현한 것이다.
    역시 예상대로 Pomeranian 이미지의 경우 가운데를 가리면 검출 능력이 현저히 떨어지는 것을 확인할 수 있다.
    (e)는 가리는 영역에 따라서 ZFNet이 분류가 어떻게 달라지는지를 보여준다.
    Pomeranian 영상에서는 가운데를 가리는 경우가 아니면, 대부분 제대로 분류를 해낸다.
    다른 2개의 예제 영상은 그림이 좀더 복잡하기 때문에 양상이 조금 다르게 나온다는 것을 확인할 수 있다.
    결과적으로 CNN의 경우는 분류나 인식을 원하는 개체의 위치에 따라
    반응이 달라진다는 것을 확인할 수가 있다.

- [Part Ⅴ. Best CNN Architecture] 4. ZFNet [1] ~ 4. ZFNet [3]은
  Matthew Zeiler의 Visualization 기법을 기반으로 한 ZFNet에 대하여 알아보았다.
  Visualization 기법을 통해 CNN의 동작 원리를 잘 이해할 수 있게 되었으며,
  최적의 구조 여부를 판단할 때도 Visualization을 활용을 잘 하면 큰 도움을 받을 수 있다는 것도 알게 되었다.
  다음 Class에서는 2014년 ILSVRC에서 우승한 구글의 GoogLeNet의 구조에 대하여 살펴볼 예정이다.

### 5. GoogLeNet [1]

2014년 ILSVRC는 구글의 GoogLeNet이 차지하였고,
아주 근소한 차이로 옥스퍼드 대학교의 VGGNet이 2위를 차지한다.
그런데 여기서 주목해야 할 것은 2014년부터 CNN의 구조에 큰 변화가 나타나기 시작한다는 점이다.
AlexNet이나 ZFNet 그리고 원조격인 LeNet5는 2014년 구조에 비하면 아주 단순한 편이며,
전형적인 형태를 취하고 있고, 망의 깊이도 10 layer 미만이다.
2014년 변화의 특징은 한마디로 "deeper"라고 표현을 할 수 있다.
GoogLeNet이나 VGGNet은
2012년 Krizhevsky의 AlexNet에서 촉발된 에너지를 바탕으로 새로운 변화를 모색하게 되었으며,
CNN을 통한 학습 능력이 훨씬 더 커지게 되었음을 입증하였다.
이번 Class부터 GoogLeNet의 구조에 대하여 설명할 예정이다.
이해를 돕기 위해 여러 part로 구성이 될 예정이며,
VGGNet은 GoogLeNet에 대한 Class를 마치고 따로 자세히 설명을 할 예정이기는 하지만,
중간 중간에 필요에 따라 VGGNet에 대한 설명도 곁들일 예정이다.
망은 깊어진다 (deeper and deeper) !!
CNN의 성능을 향상시키는 가장 직접적인 방식은 망의 크기를 늘리는 것이다.
여기서 망의 크기를 늘린다는 것은 단순하게 망의 layer 수(depth)를 늘리는 것뿐만 아니라,
각 layer에 있는 unit의 수(width)도 늘리는 것을 의미한다.
특히 ImageNet 데이터와 같이 대용량 데이터를 이용해 학습을 하는 경우는 거의 필수적이라고 할 수 있다.
아래 그림은 image classification의 성능 향상을 위해
CNN의 구조가 어떻게 바뀌고 있는지를 명쾌하게 보여주는 그림이다.
2013년까지는 CNN 망의 깊이가 10 layer 미만이었지만,
2014년의 대표주자인 GoogLeNet과 VGGNet은 각각 22 layer와 19 layer로 깊어지게(deeper) 된다.
물론 top-5 에러율도 각각 6.7%와 7.3%로 낮아지게 된다.
AlexNet의 결과가 나온 뒤 불과 2년 만에 에러율을 약 10% 정도 낮추는 쾌거를 이루게 된다.
2015년 우승을 한 ResNet은 망의 깊이가 152 layer 로 더욱 깊어지게 되며,
top-5 에러율도 3.57%로 더욱 낮아지게 된다.
망이 깊어지면, 부작용(side effect)는 없나?
망의 크기를 늘리면 성능을 더 높일 수 있지만,
적절하지 못하면 다음 2가지 중대한 문제를 만날 수도 있다.
우선 망이 커지면 커질수록 자유 파라미터(free parameter)의 수가 증가하게 되며,
이렇게 되면 특히 학습에 사용할 데이터 양이 제한적인 경우에 더 심각한 문제가 되지만,
망이 overfitting에 빠질 가능성이 높아진다.
(즉, 학습 데이터에만 특화된 결과가 만들어져, 실제 테스트 set에 적용하면 만족할 만한 결과가 나오지 못할 수 있다.)
그리고 대량의 데이터에 사람이 일일이 label을 달아주는 것도 쉬운 일이 아니다.
또 다른 문제는 망의 크기가 커지면 그만큼 연산량이 늘어나게 된다.
예를 들어 필터의 개수가 증가하게 되면, 연산량은 제곱으로 늘어나게 된다.
연산 능력이 뛰어난 GPU를 사용하더라도 연산량의 증가는 심각한 문제가 된다.
그리고, ZFNet을 학습하면서 살펴보았던 것처럼,
학습이 잘못되어 filter의 kernel이 특정한 무리로 쏠리게 된다면,
기껏 망의 크기를 늘렸음에도 불구하고, 최적의 결과를 얻지 못할 수도 있다.
GoogLeNet보다 layer 수가 작은 AlexNet 경우를 살펴보자.
[Part Ⅴ. Best CNN Architecture] 3. AlexNet [1] 에서 살펴본 것처럼, AlexNet은 엄청난 연산을 필요로 한다.
AlexNet은 자유 파라미터의 개수가 6000만개이고 약 6억 3000만개의 connection으로 이루어져 있으며,
엔비디아의 GTX580 dual-GPU를 이용하여도 학습 시간이 일주일 넘게 소요되었다.
단순하게 망을 깊게 만든다면, 자유 파라미터의 개수가 더욱 많아질 것이고,
connection도 엄청나게 많아지면서, 학습에 필요한 시간도 더욱 길어지게 된다.
또한 parameter 값이 정해지더라도 실제 연산을 할 때의 연산량 역시 무시할 수 없게 되고 말 것이다.
그리고 모바일이나 embedded 시스템에서 CNN을 활용하고자 한다면,
연산 능력이나 메모리 사용 등에서 PC 를 사용할 때보다 훨씬 제한될 수 밖에 없기 때문에,
단순히 망을 깊게 만든 것이 아니라, 뭔가 구조적인 고민이 필요하다.
GoogLeNet과 Inception
크리스토퍼 놀란 감독의 영화 인셉션(Inception)을 보면,
남에게 어떤 생각(꿈)을 주입하거나, 남의 생각을 읽어내는 내용이 나온다.
구글의 연구팀들은 그 영화에서 컨셉을 따와 인셉션이라는 이름을 갖는 CNN모듈을 만들어 낸다.
구글의 소개 자료를 보면 항상 다음과 같은 그림이 등장한다.
이는 더 깊은 CNN 구조를 사용하면 더욱 좋은 성능을 얻을 수 있다는 것 때문인 것 같고,
또한 인셉션의 내용이 남의 생각을 읽어내듯,
DNN을 이용한 데이터로부터 중요한 정보를 얻어내는 것을 연상하여 지은 이름 같다.
구글이 발표한 인셉션의 기본 구조는 아래와 같으며, 대충 봐도 무지 복잡해 보인다.
같은 layer에 서로 다른 크기를 갖는 convolutional filter를 적용하여 다른 scale의 feature를 얻을 수 있도록 했다.
그림에서처럼, 1x1 convolution을 적절히 사용하여 차원을 줄이고(reduce dimension) 망이 깊어졌을 때
연산량이 늘어나는 문제를 해결하였다.
이 부분은 다음 class에서 자세하게 설명할 예정이다.
(비록 복잡해 보이지만, 그 개념과 원리를 이해하면 충분히 공감이 갈 구조이다.)
GoogLeNet은 구글의 연구팀들이 인셉션 모듈을 고안한 뒤에 2014년 ILSVRC에 참가하기 위한 버전으로 내놓은 것이며,
인셉션의 구조는 다양한 형식으로 적용이 가능하다.
AlexNet과 GoogLeNet을 비교한 그림은 아래와 같다.
놀라운 부분은 망의 깊이는 훨씬 깊은데 free parameter의 수는 1/12 수준이고 전체 연산량의 숫자도 AlexNet에 비해 적다는 것을 알 수가 있다.
GoogLeNet과 인셉션에 대한 설계 철학을 정확하게 이해를 하면 그 이유를 알 수 있으며,
역시 설명할 부분이 많아 다음 Class 에서 다룰 예정이다.
참고로 GoogLeNet에는 총 9개의 인셉션 모듈이 적용되어 있다.
정리하자면, 구글의 연구팀들은 망을 더 깊게 하여 성능 향상을 꾀하면서도,
연산량의 증가를 늘리지 않는 CNN 구조를 개발하기 위해 많은 연구를 하였다.
결과적으로 초기 CNN 구조가 적합하지 않다는 것을 발견하였으며,
효과적으로 차원을 줄이면서 망을 깊게 할 수 있는 방법으로 인셉션 모듈을 개발하였다.
그 후에도 구글의 개발팀들은 인셉션을 더욱 발전시켜 자신들의 CNN연구의 기본으로 삼았으며,
관련 논문들도 많이 발표를 하였다.
이번 Class에서는 구들의 GoogLeNet이 이전 CNN 구조와 많이 다르다는 것에 대하여 간단하게 살펴보았다.
다음 Class부터는 그들의 설계 철학과 왜 이런 구조를 발전시키게 되었는지에 대해
여러 번에 걸쳐 상세하게 살펴볼 예정이다.

### GoogLeNet [2]

CNN(Convolutional Neural Network) ? “GoogLeNet (part2)”
지난 Class에서 구글의 GoogLeNet에 대하여 간단하게 살펴보았다.
그 동안 살펴본 LeNet-5, AlexNet, ZFNet 등은 그런대로 이해하는데 무리가 없을 것이라 생각되지만,
다소 괴상하게(?) 생긴 GoogLeNet은 왠지 부담스러울 것 같다.
GoogLeNet에서는 망의 깊이 및 넓이가 모두 커지고,
중간에 분기되는 부분도 있고, “인셉션”이라는 생소한 모듈이 등장한다.
하지만, 이 모든 것들이 구글의 연구팀들이 최초로 개발한 것은 아니며,
이들 역시 타 연구자들의 연구와 자신들의 연구를 융합하여 발전시킨 결과이다.
구글은 자신들의 구조를 설계함에 있어 크게 2개의 논문을 참조하고 있으며,
그 중 인셉션 모듈 및 전체 구조에 관련된 부분은 싱가포르 국립 대학의
“Min Lin”이 2013년에 발표한 “Network In Network” 구조를 더욱 발전 시킨 것이다.
이번 Class에서는 NIN(Network In Network)의 구조를 살펴보고,
이후 구글의 인셉션의 구조를 살펴볼 예정이다.
NIN(Network In Network) 구조와 설계 철학
NIN은 말 그대로 네트워크 속의 네트워크를 뜻한다.
일반적인 CNN 구조는 feature extraction 부분(conv + pooling layer)과
classifier 부분(fully connected neural network)으로 구성이 된다.
Convolutional layer와 pooling layer를 번갈아 사용하는 layer 군 여러 개를 사용하여 feature를 추출하고,
최종 feature vector를 classifier 역할을 하는 fully-connected neural network을 이용하여 처리 하였다.
NIN 설계자는 CNN의 convolutional layer가
local receptive field에서 feature를 추출해내는 능력은 우수하지만,
filter의 특징이 linear하기 때문에 non-linear한 성질을 갖는 feature를 추출하기엔 어려움이 있으므로,
이 부분을 극복하기 위해 feature-map의 개수를 늘려야 하는 문제에 주목했다.
필터의 개수를 늘리게 되면 연산량이 늘어나는 문제가 있다.
그래서 NIN 설계자는 local receptive field 안에서 좀 더 feature를 잘 추출해낼 수 있는 방법을 연구하였으며,
이를 위해 micro neural network를 설계하였다.
이들은 convolution을 수행하기 위한 filter 대신에 MLP(Multi-Layer Perceptron)를 사용하여
feature를 추출하도록 하였으며, 그 구조는 아래 그림과 같다.
CNN은 filter의 커널을 입력 영상의 전체 영역으로 stride 간격만큼씩 옮겨가면서 연산을 수행한다.
반면에 NIN에서는 convolution 커널 대신에 MLP를 사용하며,
전체 영역을 sweeping 하면서 옮겨가는 방식은 CNN과 유사하다.
MLP를 사용했을 때의 장점은 convolution kernel 보다는 non-linear 한 성질을 잘 활용할 수 있기 때문에
feature를 추출할 수 있는 능력이 우수하다는 점이다.
또한 1x1 convolution을 사용하여 feature-map을 줄일 수 있도록 하였으며,
이 기술은 GoogLeNet의 인셉션에 그대로 적용이 된다.
1x1 convolution은 뒤에 좀 더 자세하게 설명을 할 예정이다.
NIN 이름이 붙은 이유는 망을 깊게 만들기 위해, mlpconv layer를 여러 개를 쌓아 사용을 하기 때문이며,
그래서 네트워크 안에 네트워크가 있다는 개념이 만들어지게 되었다.
GoogLeNet에서도 인셉션 모듈을 총 9개를 사용하기 때문에 개념적으로는 NIN과 맥이 닿아 있다고 볼 수 있다.
위 그림은 mlpconv layer 3개를 사용한 구조이다.
NIN 구조가 기존 CNN과 또 다른 점은
CNN의 최종단에서 흔히 보이는 fully-connected neural network이 없다는 점이다. (위 그림 참조)
Fully-connected NN 대신에 최종단에 “Global average pooling”을 사용하였다.
이는 앞에서 효과적으로 feature-vector를 추출하였기 때문에,
이렇게 추출된 vector 들에 대한 pooling 만으로도 충분하다고 주장을 하고 있다.
Average pooling 만으로 classifier 역할을 할 수 있기 때문에 overfitting의 문제를 회피할 수 있고,
연산량이 대폭 줄어드는 이점도 얻을 수 있다.
CNN의 최종단에 있는 fully-connected NN은 전체 free parameter 중 90% 수준에 육박하기 때문에,
많은 파라미터의 수로 인해 overfitting에 빠질 가능성이 아주 높으며,
이를 피하기 위해서는 앞선 Class에서 이미 살펴 본 것처럼,
“dropout” 기법을 적용해야 하지만,
NIN의 경우는 average pooling 만으로 classification을 수행할 수 있기 때문에
overfitting에 빠질 가능성이 현저하게 줄어들게 된다.
물론 GoogLeNet에서도 global average pooling이 적용이 되어 있다.

- 1x1 Convolution이란?
  Convolution은 local receptive field의 개념을 적용하기 때문에
  7x7, 5x5, 3x3과 같이 주변 픽셀들의 정보를 같이 활용을 한다.
  그런데 괴상한(?) 이름의 1x1 convolution 이라는 개념이 나온다.
  1x1 convolution을 하는 결정적인 이유는 차원을 줄이는 것이다.
  GoogLeNet 소개 논문에 나오는 것처럼,
  Hebbian principle(Neurons that fire together, wire together)에 의해 차원의 줄일 수 있다.
  1x1 convolution을 수행하면, 여러 개의 feature-map으로부터 비슷한 성질을 갖는 것들을 묶어낼 수 있고,
  결과적으로 feature-map의 숫자를 줄일 수 있으며,
  feature-map의 숫자가 줄어들게 되면 연산량을 줄일 수 있게 된다.
  또한 연산량이 줄어들게 되면, 망이 더 깊어질 수 있는 여지가 생기게 된다.
  위 그림에서 “C2 > C3”의 관계가 만들어지면,
  차원을 줄이는 것과 같은 효과를 얻을 수 있기 때문에,
  GoogLeNet을 포함한 최신 CNN 구조에서는 1x1 convolution을 많이 사용한다.
  1x1 convolution은 처음에는 개념적으로 쉽게 와닿지 않는다.
  논문이나 설명 글을 참고할 때 1x1 convolution을 1-layer fully-connected neural network이라고도 하는데,
  그 이유는 1x1 convolution이 fully-connected와 동일한 방식이기 때문이다.
  만약에 입력 feature-map c2의 갯수가 4이고, 출력 feature-map c3의 갯수가 2인 경우를 가정해보면,
  1x1 convolution은 아래 그림과 같이 표현할 수 있다.
  결과적으로 보면 4개의 feature-map으로부터 입력을 받아, 학습을 통해 얻어진
  learned parameter를 통해 4개의 feature-map이 2개의 feature-map으로 결정이 된다.
  즉, 차원이 줄어들게 되며, 이를 통해 연산량을 절감하게 된다.
  또한, neuron에는 활성함수로 RELU를 사용하게 되면, 추가로 non-linearity를 얻을 수
  있는 이점도 있다.
  이상으로 1x1 convolution에 대한 간단한 설명을 마치고,
  나중에 VGGNet을 설명할 때,
  convolution을 어떻게 구현할 것인가 및 차원을 어떻게 줄여갈 것인가에 대하여 자세히 설명할 예정이다.
  구글의 인셉션(Inception)
  구글은 인셉션에 대한 개발을 하면서 NIN 구조를 많이 참고하였다.
  Local receptive field에서 더 다양한 feature를 추출하기 위해
  여러 개의 convolution을 병렬적으로 활용하려고 하였다.
  원래 1x1 convolution, 3x3 및 5x5 convolution, 3x3 max pooling을 나란히 놓는 구조를 고안하였다.
  다양한 scale의 feature를 추출하기에 적합한 구조가 된다.
  하지만 곧 문제에 부딪치게 된다. 3x3과 5x5 convolution은 연산량의 관점에서 보면,
  expensive unit(값 비싼 대가를 치러야 하는 unit)이 된다.
  망의 깊이가 깊지 않을 때는 큰 문제가 아니나,
  망의 깊이와 넓이가 깊어지는 GoogLeNet 구조에서는 치명적인 결과가 될 수도 있다.
  그래서 아래의 그림과 같이 3x3 convolution과 5x5 convolution의 앞에,
  1x1 convolution을 cascade 구조로 두고, 1x1 convolution을 통해 feature-map의 개수(차원)를 줄이게 되면,
  feature 추출을 위한 여러 scale을 확보하면서도, 연산량의 균형을 맞출 수 있게 된다.
  GoogLeNet의 22 layer까지 깊어질 수 있는 것도 따지고 보면,
  1x1 convolution을 통해 연산량을 조절할 수 있었기 때문에, 이렇게 깊은 수준까지 망을 구성할 수 있게 된 것이다.
  3x3 max pooling에 대해서도 1x1 convolution을 둔다.
  NIN에서는 MLP를 이용하여 non-linear feature를 얻어내는 점에 주목을 했지만,
  MLP는 결국 fully-connected neural network의 형태이고, 구조적인 관점에서도 그리 익숙하지 못하다.
  반면에 구글은 기존의 CNN 구조에서 크게 벗어나지 않으면서도 효과적으로 feature를 추출할 수 있게 되었다.
  요약하면, 인셉션 모듈을 통해 GoogLeNet 은 AlexNet에 비해 망의 깊이는 훨씬 깊은데
  free parameter의 수는 1/12 수준이고 전체 연산량도 AlexNet에 비해 적다는 것을 알 수가 있다.
  참고로 GoogLeNet에는 총 9개의 인셉션 모듈이 적용되어 있다.
  이제 인셉션 모듈에 대한 이해를 하였으니,
  다음 Class부터는 GoogLeNet의 전체 구조에 대한 설계 철학과
  왜 이런 구조를 발전시키게 되었는지에 대해 상세하게 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

### GoogLeNet [3]

2014년에 발표된 GoogLeNet과 VGGNet 등에서 가장 주목할 만한 특징은 망이 깊어지고 넓어졌다는 점이며, 이후 CNN 구조의 대세를 이루게 된다. 그럼 왜 이렇게 망은 넓어지고 깊어지는 것일까? 답은 깊어진 망을 이용하게 되면 문제 해결능력, 즉 학습 능력이 증가하기 때문이다.
하지만, 망이 깊어지게 되면 앞서 살펴보았듯이, 적절한 hyper-parameter의 설정이나 초기값 설정이 없다면 overfitting 문제에 빠질 가능성이 훨씬 증가하게 되며, 연산량의 문제로 인해 아주 성능 좋은 시스템을 사용하여 학습을 할지라도 시간이 너무 오래 걸리고, 그로 인해 최적의 결과를 얻어내지 못할 수도 있게 된다. 또한 스마트 폰이나 embedded system 에서는 아주 제한된 하드웨어 자원으로 인해, 망이 깊어지는 것에 대해서는 꿈도 꾸지 못할 수도 있다.
GoogLeNet 설계자들은 단순하게 hyper-parameter나 초기값을 적절하게 잘 설정하는 것만으로는 한계가 있다는 점을 잘 이해하였기 때문에 전체 구조적인 측면을 주목 하였으며, 다양한 CNN 구조 실험을 통해, 연산량을 유지하면서 망의 깊이와 넓이를 증가시킬 수 있는 구조를 얻게 되었다.
그 결과를 2014년 ILSVRC 대회 우승을 통하여 입증을 하였으며, 2015년에 발표한 그들의 또 다른 논문을 살펴보면, 좀 더 구조적인 개선을 통하여, 그 성능이 다시 2배 정도 좋아지는 쾌거를 이루게 되었다.
지난 2개의 Class를 통해, GoogLeNet의 핵심인 1x1 convolution, NIN(Network-in-Network), 인셉션 모듈의 기본 개념에 대하여 살펴보았기 때문에, 이번 Class에서는 그 개념들을 결합하여 GoogLeNet의 전체 구조에 대하여 살펴볼 예정이다.
GoogleLeNet의 핵심 철학 및 구조
GoogLeNet 의 핵심 설계 철학은 주어진 하드웨어 자원을 최대한 효율적으로 이용하면서도 학습 능력은 극대화 할 수 있도록 깊고 넓은 망을 갖는 구조를 설계하는 것이다.
이를 위해 [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [1] 와 5. GoogLeNet [2] 에서 살펴본 것처럼, 인셉션이라는 독특한 모듈을 만들어 낸다. 인셉션 모듈에 있는 다양한 크기의 convolution kernel(그림에서 파란색 부분)을 통해 다양한 scale의 feature를 효과적으로 추출하는 것이 가능해졌다. 또한 인셉션 모듈 내부의 여러 곳에서 사용되는 (위 그림의 노란색 부분) 1x1 convolution layer를 통해, 연산량을 크게 경감시킬 수 있게 되어, 결과적으로 망의 넓이와 깊이를 증가시킬 수 있는 기반이 마련 되었다. 이 인셉션 모듈을 통해 NIN(Network-in-Network) 구조를 갖는 deep CNN 구현이 가능하게 되었다.
인셉션 모듈을 개발하고, 2014년 ILSVRC에 참가하기 위한 구조 결정을 마치고, GoogLeNet이라는 이름을 붙인다. 인셉션은 모듈 개념이기 때문에, CNN을 적용하고자 하는 목적이나 분야에 따라 그 조합을 다르게 가져갈 수도 있다. GoogLeNet의 경우는 파라미터가 있는 layer를 기준으로 총 22개의 layer를 갖는 말 그대로 deep network이 만들어진다. 그리고 실제로 망에 들어 있는 unit의 수를 합하면 약 100개의 unit이 된다. 2014년 이전에는 볼 수 없었던 방대한(?) 네트웍이 만들어진 셈이다.
GoogLeNet에는 아래 그림과 같이, 총 9개의 인셉션 모듈이 적용이 되었다. 그림에서 빨간색 동그라미가 인셉션 모듈에 해당이 된다. 망이 워낙 방대하여 작은 그림으로 표현하기에 아쉬움이 있다.
위 그림에서 파란색 유닛은 convolutional layer를 의미하고, 빨간색은 max-pooling 유닛을 뜻하며, 노란색 유닛은 Softmax layer이고, 녹색은 기타 function을 가리킨다. 인셉션 모듈을 나타내는 동그라미 위에 있는 숫자는 각 단계에서 얻어지는 feature-map의 수를 나타낸다.
GoogLeNet의 각 layer의 구조는 위 표와 같다. 위 표는 얼핏 보기에는 복잡해 보이지만, 조금만 살펴보면 그 내용을 정확하게 이해할 수 있다.
l  Patch size/stride: 커널의 크기와 stride 간격을 말한다. 최초의 convolution에 있는 7x7/2의 의미는 receptive field의 크기가 7x7인 filter를 2픽셀 간격으로 적용한다는 뜻이다.
l  Output size: 얻어지는 feature-map의 크기 및 개수를 나타낸다. 112x112x64의 의미는 224x224 크기의 이미지에 2픽셀 간격으로 7x7 filter를 적용하여 총 64개의 feature-map이 얻어졌다는 뜻이다.
l  Depth: 연속적인 convolution layer의 개수를 의미한다. 첫번째 convolution layer는 depth가 1이고, 두번째와 인셉션이 적용되어 있는 부분은 모두 2로 되어 있는 이유는 2개의 convolution을 연속적으로 적용하기 때문이다.
l  #1x1: 1x1 convolution을 의미하며, 그 행에 있는 숫자는 1x1 convolution을 수행한 뒤 얻어지는 feature-map의 개수를 말한다. 첫번째 인셉션 3(a)의 #1x1 위치에 있는 숫자가 64인데 이것은 이전 layer의 192개 feature-map을 입력으로 받아 64개의 feature-map이 얻어졌다는 뜻이다. 즉, 192차원이 64차원으로 줄어들게 된다.
l  #3x3 reduce: 이것은 3x3 convolution 앞쪽에 있는 1x1 convolution 을 의미하여 마찬가지로 인셉션 3(a)의 수를 보면 96이 있는데, 이것은 3x3 convolution을 수행하기 전에 192차원을 96차원으로 줄인 것을 의미한다.
l  #3x3: 1x1 convolution에 의해 차원이 줄어든 feature map에 3x3 convolution을 적용한다. 인셉션 3(a)의 숫자 128은 최종적으로 1x1 convolution과 3x3 convolution을 연속으로 적용하여 128개의 feature-map이 얻어졌다는 뜻이다.
l  #5x5 reduce: 해석 방법은 “#3x3 reduce”와 동일하다.
l  #5x5: 해석 방법은 “#3x3”과 동일하다. #5x5는 좀 더 넓은 영역에 걸쳐 있는 feature를 추출하기 위한 용도로 인셉션 모듈에 적용이 되었다.
l  Pool/proj: 이 부분은 max-pooling과 max-pooling 뒤에 오는 1x1 convolution을 적용한 것을 의미한다. 인셉션 3(a) 열의 숫자 32 는 max-pooling과 1x1 convolution을 거쳐 총 32개의 feature-map이 얻어졌다는 뜻이다.
l  Params: 해당 layer에 있는 free parameter의 개수를 나타내며, 입출력 feature-map의 數에 비례한다. 인셉션 3(a) 열에 있는 숫자 159K는 총 256개의 feature-map을 만들기 위해 159K의 free-parameter가 적용되었다는 뜻이다.
l  Ops: 연산의 수를 나타낸다. 연산의 수는 feature-map의 수와 입출력 feature-map의 크기에 비례한다. 인셉션 3(a)의 단계에서는 총 128M의 연산을 수행한다.
위 설명에 따라 표에 있는 각각의 숫자들의 의미를 해석해 보면, GoogLeNet의 구조를 좀 더 친숙하게 이해할 수 있다.
인셉션 3(a)에는 256이라는 빨간색 숫자가 적혀 있는데, 이것은 인셉션 3(a)를 통해 총 256개의 feature-map이 만들어졌다는 뜻이며, 이것은 1x1 convolution을 통해 64개, 1x1과 3x3 연속 convolution을 통해 128개, 1x1과 5x5 연속 convolution을 통해 32개, max-pooling과 1x1 convolution을 통해 32개를 적용하여 도합 256개의 feature-map을 얻을 수 있게 되었다는 뜻이다.
3x3보다는 5x5 convolution을 통해 얻는 feature-map의 개수가 작은 이유는 5x5 convolution이 훨씬 연산량을 많이 필요로 하기 때문이며, 입력 이미지의 크기가 이미 28x28로 줄어든 상황에서는 3x3으로 얻을 수 있는 feature가 5x5로 얻을 수 있는 feature보다 많기 때문일 것이다.
만약에 3x3이나 5x5 convolution 앞에 1x1 convolution이 없다면 어떻게 되었을까? 3x3 convolution의 경우는 1x1 convolution을 통해 192개의 feature-map이 96개로 줄었기 때문에 50% 정도 연산량이 줄었으며, 5x5의 경우는 192개가 16개로 줄었기 때문에, 약 91.7% 정도 연산량이 줄어들게 되었다. 특히 5x5에서 연산량이 크게 줄었기 때문에, 1x1 convolution을 통한 차원 절감의 효과를 크게 누릴 수 있다.
이번 Class에서는 GoogLeNet의 구조에 대하여 설명을 하였다. GoogLeNet은 익숙한 기존 CNN 구조와 다른 점이 많아 상세한 설명을 하려면, 지면이 많이 소요된다. 다음 Class에서는 GoogLeNet에 적용된 Auxiliary classifier 및 Inception ?V2, Inception-V3 구조 등에 대하여 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

5. GoogLeNet [4]
   CNN(Convolutional Neural Network) ? “GoogLeNet (part4)”
   지난 3개의 Class를 통해 Inception 모듈과 GoogLeNet의 기본 개념에 대하여 살펴보았다.
   이번 Class에서는 GoogLeNet이 갖고 있는 독특한 블락인 Auxiliary classifier 및
   큰 필터 크기를 갖는 convolution kernel을 인수 분해(factorization)하여
   작은 크기의 여러 단의 convolution으로 구현하는 방법에 대하여 살펴볼 예정이다.
   Auxiliary classifier
   아래 그림의 GoogLeNet 블락도를 보면 Auxiliary classifier라고 불리는 독특한 유닛이 있다.
   이전의 CNN 구조에서는 볼 수 없었던 독특한 구조이다.
   망이 깊어지면서 생기는 큰 문제 중 하나는 vanishing gradient 문제이며,
   이로 인해  학습 속도가 아주 느려지거나 overfitting 문제가 발생한다.
   신경망에서는 최종 단의 error을 역전파(back-propagation)를 시키면서 파라미터 값을 갱신한다.
   그런데 gradient 값들이 0 근처로 가게 되면, 학습 속도가 아주 느려지거나
   파라미터의 변화가 별로 없어 학습 결과가 더 나빠지는 현상이 발생할 수 있다.
   ([Part Ⅲ. Neural Networks 최적화] 5. 학습 속도 저하 현상의 원인 참조)
   활성함수로 sigmoid 함수를 쓰는 경우
   [Part Ⅲ. Neural Networks 최적화] 5. 학습 속도 저하 현상의 원인에서 살펴보았던 것처럼,
   sigmoid 함수의 특성상 일부 구간의 제외하면 미분값이 거의 0으로 수렴하기 때문에
   출력 에러의 크기와 상관없이 학습 속도가 느려진다.
   또한 Class 15에서 살펴본 것처럼, cross-entropy 함수를 사용하면 좀 더 개선은 되지만
   본질적인 문제가 해결이 되는 것은 아니다.
   요즘 DNN(Deep Neural Network)에서는 활성함수로 ReLU를 사용한다.
   Sigmoid나 cross-entropy를 사용할 때보다 많은 이점이 있기 때문이며,
   비로소 DNN을 구현할 수 있는 기반이 마련되었지만,
   여러 layer를 거치면서 작은 값들이 계속 곱해지다 보면,
   0근처로 수렴되면서 역시 vanishing gradient 문제에 빠질 수 있고,
   망이 깊어질수록 이 가능성이 커진다.
   GoogLeNet에서는 이 문제를 극복하기 위해 Auxiliary classifier를 중간 2곳에 두었다.
   학습을 할 때는 이 Auxiliary classifier를 이용하여 vanishing gradient 문제를 피하고,
   수렴을 더 좋게 해주면서 학습 결과가 좋게 된다.
   이 Auxiliary classifier는 GoogLeNet 논문에 자세한 설명은 없지만,
   이후 DNN을 연구하는 사람들이 이 구조에 대한 개선이나 이론적인 설명을 위한 논문을 발표 했으며,
   대표적인 논문을 소개하면 다음과 같다.

> Deeply supervised nets (C.Y. Lee, S. Xie 등)
> Training Deeper Convolutional Networks with Deep SuperVision(Liwei Wang, Chen-Yu Lee 등)
> 이 중 Liwei Wang의 논문이 좀 더 체계적으로 Auxiliary classifier에 대한 설명을 하고 있으니,
> 이 논문을 참고하면 도움이 될 것 같다.
> 위 그림은 Liwei Wang의 논문에 나오는 실험용 DNN의 구조이며,
> X4의 위치에 그들이 SuperVision이라고 부르는 Auxiliary classifier를 배치하고,
> back-propagation 시에 X4 위치에서는 Auxiliary classifier와 최종 출력으로부터 정상적인
> back-propagation 결과를 결합시킨다.
> 이렇게 되면, Auxiliary classifier의 back propagation 결과가 더해지기 때문에 X4 위치에서
> gradient가 작아지는 문제를 피할 수 있다.
> GoogLeNet에서는 어느 위치에 Auxiliary classifier를 놓을 것인지,
> 어떤 결과를 얻었는지 명확하게 밝히지 않았지만,
> Liwei 연구팀은 초기 (10~50)번 정도의 iteration을 통해 gradient가 어떻게 움직이는지 확인을 하고,
> 그 위치에 Auxiliary classifier를 붙이는 것이 좋다고 논문에서 밝혔다.
> 위 그림에서 왼쪽 그림은 Axiliary classifier가 없는 경우에  X4를 비롯한 X4 뒷단의 gradient가 현저하게
> 작아지는 것을 보여주는 그림이다.
> 오른쪽 그림은 Auxiliary classifier가 없는 경우는 파란색 점선과 같이 0에 근접한 값을 갖는 반면,
> 빨간색 선은 Auxiliary classifier에 의해 gradient 값이 다시 증가하게 되는 것을 보여주며,
> 결과적으로 더 안정적인 학습 결과를 얻을 수 있게 된다.
> 이후 2015년 후반에 GoogLeNet의 첫번째 저자 Szegedy가 다시 발표한 논문
> “Rethinking the Inception Architecture for Computer Vision” 에
> 다시 Auxiliary classifier에 대한 이야기가 잠깐 나오는데,
> 여기에 따르면, Auxiliary classifier가 “Regularizer”와 같은 역할을 하며,
> 최종 단의 Main classifier가 중간의 side branch에 있는 Auxiliary classifier가 batch-normalize 되었거나
> drop-out layer를 갖고 있으면 결과가 더 좋아진다는 언급이 있다.
> “batch-normalization”에 대해서는 나중에 따로 다룰 예정이다.
> 학습이 끝나고, 학습된 DNN을 이용할 때는 Auxiliary classifier는 삭제한다.
> 즉, Auxiliary classifier는 학습을 도와주기 위한 도우미 역할만을 하고,
> 학습을 통해 결과를 얻게 되면, 본래의 역할을 다했기 때문에 제거한다.
> Factorizing Convolutions
> 큰 필터 크기를 갖는 convolution 커널을 인수 분해 하면,
> 작은 커널 여러 개로 구성된 deep network를 만들 수 있으며,
> 이렇게 되면 parameter의 수가 더 줄어들면서 망은 깊어지는 효과를 얻을 수 있다.
> 위 그림은5x5 convolution을 2 layer의 3x3 convolution으로 구현한 경우를 보여준다.
> 5x5 convolution은 3x3 convolution에 비해 더 넓은 영역에 걸쳐 있는 특징을 1번에 추출할 수 있지만,
> 25/9 = 2.78배 비싼 유닛이다.
> 이 5x5 convolution은 2단의 3x3 convolution을 사용해 구현이 가능하며,
> 이 경우는 free parameter의 수는 (9+9)로 5x5 convolution의25와 비교하면 28% 만큼 절감이 가능하다.
> 아래 그림은 이 방식을 적용하여 원래의 Inception을 변형시킨 결과이다.
> 7x7 convolution의 경우도 위 경우와 마찬가지로 3 단의 3x3 convolution으로 대응이 가능하며,
> 이 경우는 (9+9+9) = 27이 되어 49 와 비교하여 45% 만큼 절감할 수 있다.
> 5x5나 7x7 convolution을 여러 단의 3x3 convolution과 같이 symmetry를 유지하는 방식으로의
> 인수 분해가 가능하지만, symmetry를 유지 하지 않고 row 방향 혹은 column 방향으로 인수 분해 하는 것도 가능하다.
> 아래 그림은 3x3 convolution을 1x3 convolution과 3x1 convolution으로 분해한 것이다.
> 이렇게 되면, free parameter의 수는 (3+3) = 6이 되어, 9와 비교하면 33% 절감이 된다.
> 비슷하게 n x n convolution은 1xn 및 nx1로 분해가 가능하며, n이 클수록 파라미터 절감 효과가 커진다.
> 이것을 인셉션 구조에 표현을 하면 아래 그림과 같이 된다. 아래 그림에서 n = 3이면
> 앞서 살펴본 인셉션 모듈과 동일하다.
> 이번 Class에서는 GoogLeNet에 있는 Auxiliary classifier가 어떤 역할을 하는가에 대하여 살펴보았고,
> 큰 filter 크기를 갖는 convolution kernel을 인수 분해를 통해 여러 단의 작은 크기를 갖는 convolution으로
> 대체하게 되면 free-paramameter의 개수가 줄면서 연산량의 절감을 가져올 수 있다는 것을 살펴보았다.
> 큰 필터를 균일한 크기의 3x3으로 표현하는 것은 VGGNet의 핵심 아이디어이며,
> 여기서 설명한 인수 분해 기술과 다음 시간에 설명할 “효과적으로 grid 크기를 줄이는 기술”은
> 구글이 자신들의 인셉선 구조를 발전시킨 Inception-V2 및 Inception-V3의 핵심 아이디어가 된다.
> 다음 Class 에서는 효과적으로 grid 크기를 줄이는 기술 및 구글의 Inception?V2, Inception-V3 구조 등에
> 대하여 살펴볼 예정이다.

[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

5. GoogLeNet [5]
   CNN(Convolutional Neural Network) ? "GoogLeNet (part5)"
   통상적인 CNN의 구조를 보면, convolutional layer 뒤에 pooling layer를 두고,
   feature-map의 grid(해상도) 크기를 줄이는 것이 일반적이었다.
   하지만 inception module을 보면
   여러개의 convolutional layer와 pooling layer가 나란히 있는 좀 독특한 모양을 볼 수 있다.
   이번 Class에서는 왜 이런 구조가 만들어졌는지를 살펴볼 예정이며,
   이를 통해서 효과적으로 grid 크기를 줄이는 원리를 이해하고자 한다.
   또한 GoogLeNet part4에서 살펴본 것처럼,
   convolutional kernel에 대한 인수 분해 기술과 기타 최적화 기술을 사용하여,
   인셉션 모듈을 개선하는 방법에 대해서 살펴보고,
   이를 통해 성능이 어떻게 개선되었는지를 살펴볼 예정이다.
   2012년 이후로 CNN 구조는 춘추전국시대와 비슷한 느낌이 든다.
   누군가가 특정 방식으로 성능을 개선하면, 다른 그룹에서는 다른 방식으로 성능을 개선하였으며,
   ILSVRC의 결과도 거의 매년 2배씩 좋아지고 있는 셈이다.
   2016년은 누가 어떤 구조를 들고 나와 또 세상을 깜짝 놀라게 할지 기대가 된다.
   효과적으로 해상도(grid size)를 줄이는 방법
   앞에서도 이야기를 했듯이, grid 크기를 줄이는 대표적인 방식은 convolution을 수행할 때
   stride를 1 이상의 값으로 설정하거나, pooling을 사용하는 것이다.
   ([Part Ⅳ. CNN] 4. Convolution Layer [2] 참고)
   ?
   최대값을 취하는 max-pooling 방식과, 평균을 취하는 average pooling에 대해서는
   이미 [Part Ⅳ. CNN] 3. CNN 구조 에서 설명을 하였으니, 이 자료를 참고하면 된다.
   전형적인 CNN 구조에서는 convolutional layer를 이용하여,
   이전 feature-map으로부터 의미 있는 특징을 추출하며,
   이 때 convolutional kernel의 개수는 숨어 있는 특징을 잘 추출할 수 있도록 충분한 수가 있어야 함은 물론이다.
   그리고 다음 단에 오는 pooling layer를 이용해 feature-map의 크기를 줄이는 것이 일반적인 방식이다.
   위 그림에서는 convolution 대신에 Inception으로 표시가 되어 있지만,
   이것을 convolution이라고 생각을 해도 큰 차이는 없다.
   35x35 크기의 320개의 feature-map을 입력으로 하여 17x17 크기의 640개의 feature-map을 얻고자 한다면,
   위 방식 중 어느 쪽이 효과적으로 grid 크기를 줄이는 방식일까?
   먼저 왼쪽 방식은 35x35x320 feature-map에 먼저 pooling을 적용하여 크기를 절반으로 줄인다.
   뒤에 Inception을 적용하여 17x17 크기의 640개 feature-map을 얻었다.
   연산량 관점에서만 보면 이 방식이 효율적인 것처럼 보이지만,
   Pooling 단계를 거치면서, 原 feature-map에 있는 숨어 있는 정보(representational concept)가
   사라지게 될 가능성이 있으므로 효율성 관점에서 보면 최적이라고 보기는 어렵다.
   반면에 오른쪽은 Inception module을 먼저 적용하여 640개의 feature-map을 얻은 후에
   pooling을 적용하여 feature-map의 크기를 줄였다.
   이 경우에는 큰 크기의 feature-map에 Inception을 적용하였기 때문에
   연산량의 관점에서 보면, 결과적으로 4배가 많은 셈이 된다.
   당연히 feature-map의 크기를 줄이기 전에 Inception을 적용하였기 때문에
   숨은 특징을 더 잘 찾아낼 가능성은 높아진다.
   그렇다면 좀 더 효과적으로 grid 크기를 줄이는 방법은 무엇일까?
   Szegedy(GoogLeNet 설계자중 한 명)는
   자신의 논문(Rethinking the inception architecture for computer vision)에서 아래 그림과 같은 구조를 제시한다.
   먼저 왼쪽은 구조가 Inception module과 비슷하다는 것을 발견할 수 있을 것이다.
   (다른 점이 있다면, 최종단에 stride 값을 "1"이 아니라 "2"로 적용했다는 점이다)
   Pooling layer 및 convolutional layer를 나란하게 배치하고, 최종 단에 stride 값을 "2"를 적용하게 되면,
   결과적으로 5x5, 3x3 convolution을 통해 local feature를 추출하면서 stride 2를 통해 크기가 줄고,
   또한 pooling layer를 통해서도 크기를 줄이고 그 결과를 결합하는 방식이다.
   오른쪽은 왼쪽보다는 좀 더 단순한 방법으로 stride 2를 갖는 convolution을 통해
   320개의 feature-map을 추출하고 pooling layer를 통해 다시 320개의 feature-map을 추출함으로써
   효율성과 연산량의 절감을 동시에 달성할 수 있게 되었다.
   Inception-V2
   2014년 ILSVRC를 참가할 당시에는 구글은 전년도 ZFNet의 결과보다 거의 2배 정도의 성능을 얻었기 때문에,
   그리고 사람들이 식별할 수 있는 수준에 육박했기 때문에 Inception-V1 구조에 만족하였을 것 같다.
   하지만, 불과 1년 후 2015 ILSVRC에서 마이크로소프트의 ResNet이
   GoogLeNet 결과보다 거의 2배 좋은 성능으로 우승을 한다.
   추측이기는 하지만, 이 결과에 고무되어 다시 자신들의 구조를 재 검토를 했을 것 같다.
   그 결과 Inception-V2 및 Inception-V3가 나오게 되었으며,
   아마 지금은 더 성능을 올리는 방법에 대한 체계적인 연구를 하고 있을 것으로 추정이 된다.
   2014년에 발표한 VGGNet은 GoogLeNet과 거의 유사한 성능을 보이면서,
   구조도 3x3 convolution만을 사용하는 단순한 구조로 유명하다.
   여기에서 많은 힌트를 얻은 것으로 보이며,
   [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [4] 에서 살펴 본,
   convolution kernel에 대한 인수분해를 통해 망은 더 깊어지게 되고,
   효과적으로 연산량은 더 절감할 수 있게 된다.
   18-layer의 VGGNet이 22-layer의 GoogLeNet보다 연산량이 3배 가량 많았기 때문에,
   구글의 설계 철학과 맞지 않아 그들의 방식을 단순하게 따르지 않았고 단지 인수분해의 개념만을 따왔다.
   본래의 Inception module을 인수분해 방식을 사용하여 좀 더 개선하고,
   GoogLeNet 앞 단에 있던 7x7 convolution 등을 인수분해를 통해 작은 크기의 multi-layer 구조로 개선하였다.
   아래 표는 Inception-V2의 layer 구조를 보여주는 표이다.
   ?
   2014년 GoogLeNet에서는 입력 이미지로 224x224x3크기를 지원했지만,
   2015년 Inception-V2 사용한 구조에서는 입력 이미지를 299x299x3의 크기를 지원할 수 있도록 하였다.
   20014년 GoogLeNet의 앞 단 구조는 아래 그림과 같다.
   맨 앞 단은 stride 2값을 적용한7x7 convolution 뒤에 max-pooling을 적용하여
   이미지의 크기가 다시 1/4로 줄어들게 되어 56x56x64가 된다.
   다음 단에서 1x1 convolution 및 3x3 convolution, max-pooling을 통해
   인셉션 모듈의 입력으로는 28x28x192 크기가 적용이 된다.
   하지만, Inception-V2를 적용한 2015년 구조에서는 7x7 convolution은
   3개의 3x3 convolution으로 layer가 더 깊어지게 되었으며
   pooling을 통해 73x73x64 크기의 feature-map이 얻어지게 된다.
   [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [4] 에서 살펴본 인수분해 방식이 그대로 적용이 된다.
   다음은 3개의 convolution을 통해 최종적으로 35x35x288 이미지가 얻어지게 된다.
   2014년 구조에서는 1x1 convolution, 3x3 convolution, max-pooling을 거쳤지만,
   2015년 구조는 3개의 convolution으로 구현을 하였고,
   중간 과정에 stride 2를 적용하여 pooling의 효과를 얻을 수 있도록 하였다.
   다음 단에는 인셉션 모듈을 적용하는 것은 비슷하나,
   맨 앞 단의 인셉션 모듈의 개수가 2개에서 3개로 늘어났으며,
   적용하는 인셉션 모듈의 구조도
   [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [4] 에서 살펴본 구조로
   치환이 되어 망이 더 깊어지면서 연산량은 절감할 수 있게 되었다.
   최종단의 구조도 2014년 구조와 비슷하기는 하지만,
   feature-map의 개수가 좀 더 많아졌다는 점이 다르다.
   이런 구조 변화를 통해 22개의 layer를 갖던 2014년 구조에 비해,
   총 42개의 layer로 깊어지게 되었지만,
   연산량은 2.5배 늘어난 수준으로 여전히 효율성을 보인다.
   그 결과는 아래 표와 같다.
   아래 표에서 GoogLeNet의 결과가 원래 결과보다 높게 보이는 이유는
   data augmentation 기법을 적용하지 않고 single-crop을 했을 때의 결과이다.
   아래는 Inception-V2를 다양하게 적용했을 때의 결과이며,
   regularization 효과를 극대화 시키기 위해 batch-normalized auxiliary classifier를 적용하면,
   성능이 5.6%까지 좋아진다는 것을 확인할 수 있다.
   Muiti-crop을 144개까지 적용하고, Inception-V2의 성능을 더 극대화 시킨 Inception-V3 구조에서는
   top-1 error율이 4.1%까지 떨어져 아주 우수한 성능을 보이게 된다.
   이번 Class에서는 효과적으로 grid 크기를 줄이는 방법에 대하여 살펴보았다.
   단순하게 pooling layer를 적용하는 것보다는
   convolution layer와 같이 나란히 적용하는 것이 효과적이라는 것을 파악하였다.
   또한 convolution kernel 에 대한 인수분해 방식을 적용하고,
   앞 뒤 일부 구조 및 feature-map의 개수를 조정하는 것만으로도
   성능을 상당히 개선할 수 있다는 것도 확인을 하였다.
   GoogLeNet에서는 단순하게 분류(classification) 성능만 개선한 것이 아니라,
   R-CNN(Regions with Convolutional Neural Networks)를 이용해
   위치까지 포함한 object detection 성능을 개선하였다.
   다음 Class 에서는 이 부분에 대하여 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

5. GoogLeNet [6]
   CNN(Convolutional Neural Network) ? “GoogLeNet (part6)”
   object classification은 영상 내에 특정 대상(object)이 있는지 여부를 가리는 행위로 2012년 AlexNet이 발표되면서 거의 매년 성능이 2배 정도씩 좋아지는 경향을 보이고 있다. 하지만, object detection은 영상 내에 특정 대상이 존재하는지 여부를 판단하는 것 뿐만 아니라, 대상의 정확한 위치까지 파악을 하고 그것을 bounding box라고 부르는 사각형 영역으로 구분하는 것까지 수행을 하여야 하기 때문에 classification에 비해 훨씬 어렵다고 볼 수 있다.
   Object detection은 machine learning 관련 지식뿐만 아니라, computer vision 관련 지식을 같이 필요로 하기 때문에, classification에 비해서는 상대적으로 어렵다고 볼 수 있고 성능 향상도 더뎠다. 그러다가 2013년 버클리 대학교의 Ross Girshick 팀이 R-CNN(Regions with CNN features)이라는 방법을 발표하면서, 이전 detection 알고리즘에 비해 2배 이상의 성능 향상이 이루어지고, 뒤를 이어 여러 팀들이 속도와 성능을 개선하는 알고리즘을 속속 발표하게 된다.
   ILSVRC에서는 classification 성능뿐만 아니라, detection 성능도 같이 순위를 매긴다. GoogLeNet 설계자들은 이미 앞서 살펴본 것처럼 인셉션 모듈이라는 독특한 유닛을 이용하여 classification의 성능 향상을 도모하였으며, 전년에 발표된 Girshick의 R-CNN 개념을 적용하여 detection 분야에서도 1위를 차지하게 된다. 같은 해에 버클리 팀도 참여를 하였지만 mAP(mean Average Precision)에서 버클리 팀을 큰 폭으로 따돌리며 여유 있게 우승을 한다. 이는 버클리 팀이 AlexNet을 기반으로 사용하였지만,  성능면에서 GoogLeNet이 훨씬 뛰어났기 때문이라고 생각한다.
   GoogLeNet에서는 R-CNN 개념과 비슷한 개념을 사용했다고만 언급을 하고 있고 어떻게 적용을 하였는지 상세 설명을 하지 않았기 때문에, 이번 class에서는 CNN feature에 기반한 detection 알고리즘인 R-CNN, SPPNet 에 대하여 살펴 볼 예정이다.
   R-CNN(Regions with CNN features)
   R-CNN 알고리즘이 발표되기 이전에 대부분 object detection에 주로 사용되던 방법은 SIFT(Scale Invariant Feature Transform)이나 HOG(Histogram of Gradient)에 기반한 알고리즘이다. SIFT나 HOG는 대부분 영상 내에 존재하는 gradient 성분을 일정 블락으로 나누고 그 경향성을 이용하여 대상을 검출하는 방식을 사용한다.
   영상에 존재하는 low-level feature에 기반하기 때문에 성능상의 한계가 분명히 존재하며, 이를 보완하기 위해 여러 알고리즘을 섞어 사용하였다. (SIFT는 [Part Ⅴ. Best CNN Architecture] 4. ZFNet [2] 참고). 하지만 성능상의 한계로 인해 일정 정도 이상의 성능을 얻을 수 없었다.
   버클리의 연구팀은 그동안 DPM이나 HOG 등을 이용한 detection 연구를 꾸준히 해왔던 팀이며, 2012년 AlexNet의 연구 결과에 자극을 받아, CNN을 이용해 detection 연구를 하는 쪽으로 방향을 선회한 것으로 보인다. 특히 AlexNet 개발자들이 source code까지 open함에 따라 자신들의 아이디어를 실험해 볼 수 있는 기반을 얻게 된다. 그리하여 classification 분야에서 얻은 괄목할만한 성과를 detection 분야에 적용할 수 있는 R-CNN이라는 독특한 구조를 발표하게 된다.
   R-CNN은 위 그림과 같이 입력 영상으로부터 약 2000개의 후보 영역을 만든다. 이 때 사용하는 방법은 Selective search 방법(상세 내용은 Selective Search for Object Recognition 논문 참고)을 적용하여 후보 영역을 선정한다. Selective search는 Uijlings가 처음 발표를 하였으며, segmentation의 장점과 exhaustive search의 장점을 골고루 활용을 하였으며, 영상 속에 있는 color나 texture 등 단순한 정보뿐만 아니라 영상 속에 내재된 계층 구조도 같이 활용을 한다.
   Selective search를 통해 후보 영역을 선정하면, AlexNet이 224x224 크기의 이미지를 받아들이도록 되어 있기 때문에, 해당 영역을 warping이나 crop을 사용하여 224x224 크기로 만들고, 이것을 AlexNet을 약간 변형한 CNN에 인가하고 최종 출력에서 해당 영상을 대표할 수 있는 CNN feature vector를 얻어낸다.
   다음은 linear SVM을 이용해 해당 영역을 분류한다.
   결과적으로 보면, Computer Vision 관련 기술과 CNN 기술을 결합하여 뛰어난 성과를 얻게 된 것이다.
   R-CNN의 성능 향상과 학습 방법
   버클리 팀은 detection 성능 평가를 PASCAL VOC(Visual Object Class)를 사용하였는데, PASCAL VOC의 경우 label이 붙은 데이터의 양이 ILSVRC보다 상대적으로 적었기 때문에 ILSVRC 결과를 사용하여 CNN을 pre-training을 하였다. Pre-training에는 bounding box를 사용하지는 않았고, 단지 label 있는 ILSVRC data를 이용해 CNN에 있는 파라미터들이 적절한 값을 갖도록 하였다.
   다음은 warped VOC를 이용해 CNN을 fine tuning을 한다. 이 때는 ground truth 데이터와 적어도 0.5 IoU(Intersection over Union: 교집합/합집합) 이상 되는 region 들만 postive로 하고 나머지는 전부 negative로 하여 fine tuning을 시행한다. 이 때 모든 class에 대해 32개의 positive window와 96개의 background window를 적용하여 128개의 mini-batch로 구성을 한다.
   마지막으로 linear classifier의 성능을 개선을 위해 hard negative mining 방법을 적용하였다.
   위 과정은 아래 그림과 같이 표현이 가능하다.
   R-CNN을 적용하게 되면 기존 방법에 비해 아래 표와 같이 성능이 크게 개선되는 것을 확인할 수 있다.
   R-CNN의 문제점과 개선 알고리즘 SPPNet
   R-CNN은 region에 기반한 CNN feature를 사용하여 detection 성능을 크게 개선하였지만, 아무래도 처음 발표된 방식이라서 아래와 같은 문제점을 갖고 있다.
6. AlexNet의 구조를 그대로 사용하였기 때문에 입력 이미지 크기를 강제로 224x224 크기로 맞추기 위해 warping이나 crop을 사용했는데 이로 인한 이미지 변형이나 crop으로 인한 손실로 인해, 성능 저하가 일어날 수 있는 요인이 존재.
7. 2000여개에 이르는 region proposal에 대해 순차적으로 CNN을 수행해야 하기 때문에 학습이나 실제 run time이 긴 문제.
8. 사용하는 알고리즘이 특히, region proposal이나, SVM 튜닝 등이 GPU 사용에 적합하지 않다는 점.
   위 문제는 SPPNet(Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition) 개발자들에 의해 상당 부분이 극복이 된다.
   SPPNet은 마이크로소프트의 북경연구소에 근무하는 Kaiming He 등에 의해서 개발이 되었으며, R-CNN의 문제점을 파고 든 결과라고 보인다. 먼저 주목한 점은 AlexNet 중 convolutional layer 부분은 어차피 sliding window 방식을 사용하기 때문에 영상의 크기에 영향을 받지 않으며, 뒷단에 오는 fully connected layer 부분만 영상의 크기에 영향을 받는다는 점이었다.
   위 그림은 SPPNet 논문에 나오는 그림으로 crop이나 warping을 하면 위 그림처럼 왜곡이나 손실이 발생한다. 이 문제는 위 그림처럼 convolutional layer 다음에 spatial pyramid pooling layer를 두고 이 단계에서 pyramid 연산을 통해 입력 영상의 크기를 대응할 수 있게 되면, 굳이 crop/warp를 사용하지 않아도 된다.
   SPPNet은 BoW(Bag-of-Words) 개념을 사용한다. BoW란 특정 개체를 분류하는데 굵고 강한 특징에 의존하는 대신에서 작은 여러 개의 특징을 사용하면 개체를 잘 구별할 수 있다는 사실에 기반한다.
   AlexNet이나 ZFNet과 같은 기존 신경망의 입력 영상의 크기가 고정이 되는 이유는 Convolutional layer는 영상의 크기에 영향을 받지 않지 않지만 fully-connected layer가 입력 영상의 크기에 제한을 받기 때문이다.
   SPPNet 설계자들은 BoW 개념처럼 여러 단계의 피라미드 레벨에서 오는 자잘한 feature들을 fully-connected layer의 입력으로 사용하고, 피라미드의 출력을 영상의 크기에 관계없이 사전에 미리 정하면 더 이상 영상의 크기에 제한을 받지 않게 된다는 점에 주목을 하였다.
   그래서 기존 ZFNet과 같은 신경망의 최종 convolutional layer를 pyramid pooling layer로 변환을 시키고, 최종 피라미드 layer에서는 직전 convolutional layer의 결과를 여러 단계의 피라미드로 나눈다. 가령 1단계는 영상 전체를 커버할 수 있도록 1x1 pooling, 2단계는 영상을 4개의 영역으로 구분한 2x2 pooling, 3단계는 영상을 9개의 영역으로 구분은 3x3 pooling 등 영상을 spatial bin이라고 불리는 총 M개의 영역으로 나눈다. 이렇게 얻어진 여러 단계의 결과는 각각을 concatenation 시킨 후 fully-connected layer의 입력으로 사용한다.
   이 때입력 feature-map의 갯수가 k개 라면, 앞서 구한 M개의 벡터가 kM개의 차원으로 존재하는 셈이 된다.
   SPPNet이 R-CNN에 비해 갖는 또 다른 장점 중 하나는 R-CNN은 각각의 후보 window에 대해 crop/warp를 한 후 CNN 과정을 전부 거치지만, SPPNet에서는 영상 크기에 영향을 받지 않기 때문에 전체 영상에 대해 딱 1번 convolutional layer를 거친 후 해당 window에 대하여 SPP를 수행한 후에 이후 과정을 거치기 때문에, 가장 시간이 오래 걸리는 convolutional 과정을 건너 뛸 수가 있기 때문에 성능이 24 ~ 102 배 정도 빠르다.
   SPPNet은 2014년 ILSVRC에 참여를 하였으며, detection 부분에서는 GoogLeNet에 이어 2위를 차지했으며, classification 분야에서는 3위를 차지하였다.
   다음 그림은 SPPNet의 구조를 보여준다. AlexNet의 5번째 convolutional layer 다음에 SPP layer가 위치를 하며, 이후에 fully connected layer가 오는 구조를 취한다.
   GoogLeNet의 detection 소개 및 GoogLeNet 마무리
   GoogLeNet에서도 detection 구조에 대해서는 자세한 언급이 없다. 다만 R-CNN과 비슷한 구조를 사용하고 있으며, 인셉션 모델을 region classifier로 사용한다. 그리고 region proposal 단계에서는 selective search와 multi-box prediction을 혼합해서 사용한다. False positive의 수를 줄이기 위해 super-pixel의 크기를 2배 증가시켰으며, 이를 통해 region proposal의 수를 1/2로 줄였으며, 이렇게 효과적으로 region proposal의 개수를 줄임으로써 mAP가 1%를 개선하였으며, 학습 시간의 문제로 bounding-box regression 은 적용하지 않았지만 성능은 R-CNN에 비해 훨씬 좋은 결과를 얻을 수 있었다.
   이번 class에서는 object detection 부분을 주로 다뤘다. 아무래도 detection 부분은 원조격인 R-CNN과 SPPNet에 대한 부분을 다루는 편이 좋을 것 같아 이를 많이 소개했다.  Girshick은 나중에 마이크로소프트 사로 옮기고 Fast R-CNN이라는 더 개선된 알고리즘을 발표하기도 했지만, 이는 본 class의 범위를 벗어나는 것 같아 생략한다.
   다음 Class부터는 2014년 ILSVRC에서 주목을 받았고, 이후 CNN 알고리즘을 개발할 때 많이 사용되는 VGGNet에 대해서 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

6. VGGNet [1]
   CNN(Convolutional Neural Network) ? “VGGNet (part1)”
   2014년 ILSVRC에서 GoogLeNet과 함께 큰 주목을 받은 팀은 Oxford 대학교의 VGGNet 팀이다. 아마 GoogLeNet이 없었다면 모든 영광을 차지할 수도 있었겠지만, GoogLeNet에 근소한 차이로 뒤지며 2위를 차지한다.
   하지만 구조적인 측면에서 보면 GoogLeNet 보다 훨씬 간단한 구조로 되어 있어, 이해가 쉽고 변형을 시켜가면서 테스트 하기에 용이하기 때문에, 오히려 GoogLeNet보다 더 많이 사용이 되는 편이다.
   2015년 GoogLeNet의 첫번째 저자 Szegedy가 발표한 논문 “Rethinking the Inception Architecture for Computer Vision”의 introduction 부분을 보면, 자신들의 구조가 뛰어남에도 불구하고 VGGNet이 더 많이 쓰이는 것에 대한 아쉬움(?)을 표현하고 있으며, 더 개선된 구조임을 다시 밝힌다.
   하지만 Szegedy의 바램이나 주장에도 불구하고 GoogLeNet의 성능이 더 좋은 것이 사실이기도 하지만, 복잡하고 또한 어디를 어떻게 고쳐야 할지 한눈에 파악하기가 쉽지 않다는 것도 사실이다. 그리고 Inception-V2나 Inception-V3에서는 VGGNet의 단순한 구조를 일부 적용하였다.
   이번 Class에서는 VGGNet에 대하여 살펴볼 예정이다. VGGNet은 구조적으로 이해가 쉽기 때문에 2번의 Class로 마칠 예정이다.
   망의 깊이(depth)
   VGGNet이 발표되기 전에는 아래 그림과 같이, deep 이라고 하지만 8 layer 수준이었다. 하지만 공교롭게 2014년은 GoogLeNet과 VGGNet이 모두 이전 구조에 비해 훨씬 deep 해진다. 그리고 2015년에 발표된 마이크로소프트의 ResNet은 무려 152 layer로 더욱 깊어지게 된다.
   이렇게 망이 깊어지는 이유는 이미 GoogLeNet class에서 설명을 하였듯이, 훨씬 더 복잡한 문제를 풀 수 있기 때문이다.
   다음 그림은 VGGNet과 AlexNet의 구조를 비교 설명하는 것인데, 확실하게 깊이에서 차이가 남을 확인할 수 있다. 하지만 전반적으로 그 구조는 LeNet5 및 AlexNet과 크게 다르지 않음을 확인할 수 있다.
   VGGNet
   VGGNet 연구팀이 그들의 논문 “Very deep convolutional networks for large-scale image recognition”에서 밝혔듯이, 원래는 망의 깊이(depth)가 어떤 영향을 주는지 연구를 하기 위해 VGGNet을 개발한 것 같다.
   오직 깊이가 어떤 영향을 주는지 밝히기 위해, receptive field의 크기는 가장 간단한 3x3으로 정하고 깊이가 어떤 영향을 주는지 6개의 구조에 대하여 실험을 한다.
   여기서 주목할 점은 [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [5] 에서 이미 설명을 했던 것처럼, 3x3 kernel을 사용하면 5x5, 7x7 혹은 그 이상의 크기를 갖는 convolution을 인수분해하여 깊이는 깊어지지만 파라미터 수는 줄여서 표현할 수 있다는 것이다.
   처음부터 의도를 했던 것인지 아니면, 3x3을 기본으로 하면서 망이 깊은 구조를 만들다 보니 좋은 결과가 나왔는지는 명확하지 않지만, 어쨌든 그들의 구조 실험은 대 성공을 거두고 ILSVRC에서도 좋은 결과를 얻게 된다.
   아래 표는 그들이 실험에 사용한 6개의 구조를 보여준다.
   VGGNet은 AlexNet이나 ZFNet 처럼, 224x224 크기의 color 이미지를 입력으로 받아들이도록 하였으며, 1개 혹은 그 이상의 convolutional layer 뒤에 max-pooling layer가 오는 단순한 구조로 되어 있다. 또한 기존 CNN 구조처럼, 맨 마지막 단에는 fully-connected layer가 온다.
   VGGNet의 단점은 GoogLeNet 저자 Szegedy가 비판을 했던 것처럼, 파라미터의 개수가 너무 많다는 점이다. 아래 표를 보면 알 수 있는 것처럼, GoogLeNet의 파라미터의 개수가 5 million 수준이었던 것에 비해 VGGNet은 가장 단순한 A-구조에서도 파라미터의 개수가 133 million 으로 엄청나게 많다.
   그 결정적인 이유는 VGGNet의 경우는 AlexNet과 마찬가지로 최종단에 fully-connected layer 3개가 오는데 이 부분에서만 파라미터의 개수가 약 122 million개가 온다. 참고로 GoogLeNet은 Fully-connected layer가 없다.
   아래 표는 16 layer VGGNet(구조 D)에서 필요한 파라미터의 메모리 소요량을 정리한 것이다.
   VGGNet에서 특이한 점
   VGGNet은 몇가지 특이한 점이 있다.
7. VGGNet에서 A-LRN은 Local Response Normalization이 적용된 구조인데, 예상과 달리 VGGNet 구조에서는 LRN이 별 효과가 없어 사용하지 않는다.
8. 1x1이 적용되기는 하지만, 적용하는 목적이 GoogLeNet이나 NIN의 경우처럼 차원을 줄이기 위한 목적이라기 보다는 차원은 그대로 유지하면서 ReLU를 이용하여 추가적인 non-linearity를 확보하기 위함이다.
9. Deep net은 학습할 때 vanishing/exploding gradient 문제로 학습이 어려워질 수 있는데, 이것을 먼저 11-layer의 비교적 간단한 구조-A를 학습시킨 후 더 깊은 나머지 구조를 학습할 때는 처음 4 layer와 마지막 fully-connected layer의 경우는 구조-A의 학습 결과로 초기값을 설정한 후 학습을 시켜 이 문제를 해결하였다. (참고로 GoogLeNet은 auxiliary classifier를 사용하여 이 문제를 해결하였다. ? [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [4] 참고)
   VGGNet은 동일하게 3x3 convolution kernel을 사용하는 간단한 구조를 적용하였음에도 불구하고 놀랄만한 성능을 얻었다. 이번 Class에서는 VGGNet의 구조와 특징에 대하여 간단하게 살펴보았다. 다음 Class에서는 VGGNet이 무슨 이유로 좋은 성능을 보이는지 그리고 좀 더 세부적인 부분 및 실제 성능 등에 대하여 살펴 볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

6. VGGNet [2]
   지난 [Part Ⅴ. Best CNN Architecture] 6. VGGNet [1] 에서는 VGGNet의 기본 개념에 대해 살펴보았다. VGGNet은 Oxford 대학교의 Visual Geometry Group이 2014년 ILSVRC에 출전한 구조를 가리킨다. 이 연구팀들은 CNN, 특히 Deep CNN에 대한 연구를 꾸준히 해왔던 팀들로 CNN 구조에 관련된 좋은 논문을 많이 발표하였다. VGGNet은 이런 연구의 연장선에서 나온 개념으로 볼 수 있다. 많은 논문을 발표하였지만, 다음 2개의 논문은 매우 훌륭한 논문이라서 참고하면 좋을 것 같다.

* Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
* Understanding Deep Image Representations by Inverting Them
  이번 Class에서는 VGGNet이 무슨 이유로 좋은 성능을 보이는지 그리고 좀 더 세부적인 부분 및 실제 성능 등에 대하여 살펴 볼 예정이다.
  망을 깊게 만들기
  망이 깊어지면 왜 좋은지는 이미 GoogLeNet class에서 충분히 살펴보았다. GoogLeNet은 망을 깊게 만들면서 파라미터의 수를 줄이기 위해 inception이라는 독특한 구조를 만들었고, 깊어지는 망의 학습을 돕기 위해 auxiliary classifier 개념을 고안했다.
  VGGNet 팀은 깊은 망의 독특한 구조를 설계하는 것이 목적이 아니라, 망의 깊이가 어떤 영향을 끼치는지를 확인하기 위해 가장 단순한 구조인 3x3 convolution layer를 겹쳐서 사용하는 방법을 취했으며, 여러 개의 구조 실험을 통해, 어느 정도 이상이 되면 성능 개선의 효과가 미미해지는 지점이 있음을 확인했다. (ILSVRC-2012 데이터의 경우, 16-layer가 넘어서면 별 이득이 없는 것으로 논문을 통해서 확인이 되었다. 몇 개의 layer가 최적인지는 학습 데이터의 수나 해결하고자 하는 문제의 성격에 따라 달라질 수 있다)
  VGGNet의 기본 개념은 다음 그림과 같다. CNN의 기본 구조가 convolutional layer 다음에 해상도를 줄이기 위한 pooling layer가 오는 구조인데, 여기서 노란색 부분에  receptive field의 크기가 3x3인 filter를  여러 개를 stack 하는 구조를 택했다.
  이렇게 3x3 convolutional layer를 2개 쌓으면 5x5 convolution이 되고, 3개를 쌓으면 7x7 convolution이 된다. 그리고 덤으로 얻을 수 있는 효과는 파라미터의 수가 줄어들고, 그로 인해 학습의 속도가 빨라진다는 점이다. (자세한 내용은 [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [4] 참고) 또한 layer 수가 많아질수록 non-linearity가 더 증가하기 때문에 좀 더 유용한 feature를 추출하기에 좋아진다. 그래서 실제로 5x5 1개로 구현하는 것보다는 3x3 2개로 구현하는 편이 결과가 더 좋다. (뒤쪽 성능 비교 참고)
  11-layer의 구조에서 중간에, 아래 그림처럼 화살표의 위치에 convolutional layer를 추가하여 13-layer 구조로 만들고, 13-layer 구조에 다시 convolutional layer를 추가하여 16-layer 구조로 발전시켰으며, 구조 발전의 방향은 아래 그림과 같다.
  망이 깊어지면 vanishing/exploding gradient의 문제로 인해 학습에 어려움이 생기는데, 이 문제를 해결하기 위해  VGGNet은 11-layer 구조의 학습 결과를 더 깊은 layer의 파라미터 초기화 시에 이용함으로써 학습의 시간을 줄이고, vanishing gradient 문제도 해결하였다.
  VGGNet에서 성능을 끌어올리기 위해 적용한 방법
  VGGNet 팀은 3x3 convolution이라는 단순한 구조로부터 성능을 끌어내기 위해, training과 test에 많은 공을 들였으며, 다양한 경우에 성능이 어떻게 달라지는지 공개하여 CNN 구조 이해에 많은 기여를 하였다.
  Training 방법
  ILSVRC-2012 에는 1000개의 class가 있으며, 각각의 class에는 약 1000장 정도의 학습 이미지가 있다. Class 당 1000장의 학습 데이터라면 결코 많다고 볼 수 없으며, overfitting에 빠지지 않도록 학습 방법을 잘 설정해줘야 한다. 이 때 흔히 사용하는 방법이 지능적으로 학습 데이타의 양을 늘리기 위한 data augmentation 기법이다.
  AlexNet은 모든 학습 이미지를 256x256 크기로 만든 후, 거기서 무작위로 224x224 크기의 이미지를 취하는 방식으로 학습 데이터의 크기를 2048배로 늘렸으며, RGB 컬러를 주성분 분석(PCA)를 사용하여 RGB 데이터를 조작하는 방식도 사용하였다. 하지만 AlexNet에서는 모든 이미지를 256x256 크기의 single scale 만을 사용하였다.
  반면에 GoogLeNet은 영상의 가로/세로 比를 [3/4, 4/3]의 범위를 유지하면서 원영상의 8%부터 100%까지 포함할 수 있도록 하여 다양한 크기의 patch를 학습에 사용하였다. 또한 photometric distortion을 통해 학습 데이타를 늘렸다.
  VGGNet에서는 training scale을 ‘S’로 표시하며, single-scale training과 multi-scaling training을 지원한다. Single scale에서는 AlexNet과 마찬가지로 S = 256으로 고정시키는 경우와 S = 256과 S = 384, 두개의 scale을 지원한다.
  Multi-scale의 경우는 S를 Smin과 Smax 범위에서 무작위로 선택할 수 있게 하였으며, Smin은 256이고 Smax는 512이다. 즉, 256과 512 범위에서 무작위로 scale을 정할 수 있기 때문에 다양한 크기에 대한 대응이 가능하여 정확도가 올라간다. Multi-scale 학습은 S = 384로 미리 학습 시킨 후 S를 무작위로 선택해가며 fine tuning을 한다. S를 무작위로 바꿔 가면서 학습을 시킨다고 하여, 이것을 scale jittering이라고 하였다.
  이렇게 크기 조정을 통해 얻은 학습 영상으로부터 AlexNet과 마찬가지 방식으로 무작위로 224x224 크기를 선택하였으며, RGB 컬러 성분에 대한 변경 역시 비슷한 방식으로 수행했다.
  GoogLeNet과 VGGNet은 그 이름과 표현만 좀 다를 뿐이지, 결과적으로는 multi-scale을 고려하였고, RGB 컬러 성분 변경을 통해 deep network이 적은 학습 데이터로 인한 overfitting 문제에 빠지는 것을 최대한 막으려는 노력을 하였다.
  Testing 방법
  Test는 학습된 신경망에서 최적의 결과를 도출해내는 것이 목적이며, 대부분 학습 이미지로부터 여러 개의 patch 혹은 crop을 이용하여, 가능한 많은 test 영상을 만들어 낸 후 여러 영상으로부터의 결과를 투표(voting)을 통해 가장 기대되는 것을 최종 test 결과로 정한다. 역시 마찬가지로 AlexNet, GoogLeNet 과 비교를 통해 VGGNet 설계 팀이 어떤 노력을 했는지 확인해 보자.
  AlexNet은 Test 영상을256x256 크기로 scaling하고, 그 영상을 4 코너와 중앙에서 224x224 크기의 영상을 잘라 (CNN에서는 crop이라고 함)서 5개의 영상을 만들고, 이것을 좌우 반전시켜 총 10장의 Test 영상을 만들어 냈다. 10장의 테스트 영상을 망에 집어넣고 10개의 결과를 평균하는 방식으로 최종 출력을 정했다. Softmax의 결과가 숫자로 나오기 때문에 평균을 그것들에 대한 평균을 취해 class를 결정한다.
  GoogLeNet의 경우는 테스트 영상을 256x256 1개의 scale만 사용하는 것이 아니라, 256, 288, 320, 352 로 총 4개의 scale로 만들고 각각의 scale로부터 3장의 정사각형 이미지를 선택하고, 다시 선택된 이미지로부터 4개의 코너 및 센터 2개를 취해 총 6장의 224x224 크기 영상을 취하고, 각각을 좌우반전 시키면 4x3x6x2, 총 144개의 테스트 영상을 만들어 냈다. 결과에 대해서는 AlexNet과 마찬가지로 voting을 사용한다.
  VGGNet은 ‘Q’라고 부르는 test scale을 사용하며, 테스트 영상을 미리 정한 크기 Q로 크기 조절을 한다. Q는 training scale S와 같을 필요가 없으며, 당연한 이야기이겠지만, 각각의 S에 대해 여러 개의 Q를 사용하게 되면 학습의 결과는 좋아진다.
  VGGNet의 경우는 GoogLeNet과 같은 multi-crop( GoogLeNet은 144장, VGGNet은 150장) 방식의 테스트 영상 augmentation 방법을 적용하기도 했지만, 연산량을 줄이기 위해 OverFeat 구조(OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks)에서 사용한 dense evaluation 개념을 적용하였다. 최종적으로는 성능을 더 끌어올리기 위해, multi-crop 방식과 dense evaluation 개념을 섞어 사용하였다.
  Dense evaluation은 OverFeat 구조에서 매우 중요한 부분이고, 이해를 위해 설명이 좀 더 필요하기 때문에 다음 class에서 따로 상세하게 다룰 예정이다.
  Dense evaluation의 장점은 crop의 경우처럼 원 학습 영상으로부터 영상을 잘라낸 후 그것들을 각각 ConvNet에 적용시키는 방식이 아니라, 큰 영상에 대해 곧바로 ConvNet을 적용하고 일정한 픽셀 간격(grid)으로 마치 sliding window를 적용하듯이 결과를 끌어낼 수 있어, 연산량 관점에서는 매우 효율적이지만, grid 크기 문제로 인해서 학습 결과가 약간 떨어질 수 있다. 그러므로 crop과 dense evaluation을 상보적으로(complementary) 섞어 사용하면 더 성능이 좋아진다.
  VGGNet 결과
  VGGNet에 single-scale test 영상을 적용했을 때의 결과는 아래 표와 같다. 망이 깊어질수록 결과가 좋아지고, 학습에 scale jittering을 사용한 경우에 결과가 더 좋다는 것을 확인할 수 있다.
  3x3 convolution layer 2개를 겹치면 5x5 convolution의 효과를 얻을 수 있다는 것은 이미 앞선 class에서 살펴봤다. 둘 중 어느 결과가 좋을까?
  B 구조에 3x3 convolution layer를 2개 겹쳐서 사용하는 경우와 5x5 convolution 1개를 사용하는 모델을 만들어 실험을 했는데, 결과는 3x3 2개를 사용하는 것이 5x5 1개보다 top-1 error에서 7% 정도 결과가 좋았다고 한다. 3x3 convolution이 단순하게 망을 깊게 만들고, 파라미터의 크기를 줄이는 것뿐만 아니라, 뉴런에 있는 non-linearity 활성함수를 통해 feature 추출 특성이 좋아졌음을 반증한다.
  Multi-scale test 결과는 아래 표와 같다. Multi-scale test는 S가 고정된 경우는 {S-32, S, S+32}로 Q 값을 변화 시키면서 테스트를 했다. 여기서 학습의 scale과 테스트의 scale이 많이 차이가 나는 경우는 오히려 결과가 더 좋지 못해 32만큼 차이가 나게 하여 테스트를 하였다.
  학습에 scale jittering을 적용한 경우는 출력의 크기는 [256, 384, 512]로 테스트 영상의 크기를 정했으며, 예상처럼 scale jittering을 적용하지 않은 것보다 훨씬 결과가 좋고, single-scale 보다는 multi-scale이 결과가 좋다는 것을 확인할 수 있다.
  앞서 살펴본 것처럼, multi-crop과 dense evaluation은 각각 적용했을 때는 grid 크기 문제로 인해 multi-crop이 아주 약간 성능이 좋은 편이며, 상보적인 특성을 갖고 있기 때문에 같이 적용을 하게 되면 성능이 개선되는 것을 아래 표처럼 알 수가 있다.
  이상으로 VGGNet의 특성과 성능 개선을 위해 그들이 어떤 노력을 했는지 살펴보았다. 결론적으로 VGGNet은 그 구조가 간단하여 이해나 변형이 쉬운 분명한 장점을 갖고 있기는 하지만, 파라미터의 수가 엄청나게 많기 때문에 학습 시간이 오래 걸린다는 분명한 약점을 갖고 있다.
  또한 다양한 경우에 대한 simulation 결과를 발표하여 Deep CNN에 대한 이해를 도왔다는 점에서 많은 기여를 한 것 같다.
  다음 Class에서는 VGGNet이 test에 사용한 dense evaluation 개념을 처음으로 소개한 OverFeat에 대해서 살펴볼 예정이다. OverFeat는 2013년 ILSVRC에서 참가하여 localization 부분에서 1위를 차지하였고, 그들이 사용한 구조가 의미가 있기 때문에 1 Class로 짧게 살펴보고, ResNet으로 넘어갈 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

7. OverFeat
   2013년 ILSVRC는 뉴욕대(NYU)의 선전(善戰)이 돋보인다. Classification 분야에서 우승을 차지한 Clarifi-ZFNet이 뉴욕대 팀이고, Localization 분야에서 우승한 OverFeat 역시 뉴욕대 출신이다. 이는 CNN이라는 개념을 최초로 개발한 Yann LeCun이라는 거장이 뉴욕대에서 강의를 하고 있기 때문이 아닐까 생각한다.
   지난 [Part Ⅴ. Best CNN Architecture] 7. OverFeat 에서 VGGNet의 Test 에 multi-crop과 dense evaluation 방법을 상호보완적인 방법으로 사용하여 성능을 끌어올렸음을 확인하였지만, dense evaluation의 개념에 대하여 자세하게 설명을 하지는 않았다. 이는 설명이 길어질 수 있기 때문이기도 하지만, OverFeat 구조 역시 주목할 필요가 있어 따로 다루는 것이 좋을 것 같기 때문이었다.
   Dense evaluation은 Yann LeCun의 지도하에 Pierre Sermanet이 발표한 논문 “OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks”에 처음 소개가 되었다. CNN의 classifier로 흔히 사용되는 Fully-connected layer를 1x1 convolution 개념으로 사용하게 되면, 고정된 이미지뿐만 아니라 다양한 크기의 이미지를 sliding window 방식으로 처리할 수 있으며, feature extraction 블락의 맨 마지막 단에 오는 max-pooling layer의 전후 처리 방식을 조금만 바꾸면, 기존 CNN 방식보다 보다 조밀하게(dense) feature extraction, localization 및 detection을 수행할 수 있게 된다. 물론 multi-crop 방식보다 연산량 관점에서 매우 효율적이다.
   이번 Class에서는 OverFeat 구조에 대하여, 특히 dense evaluation에 대하여 상세하게 살펴 볼 예정이다.
   OverFeat과 다른 알고리즘 비교
   OverFeat은 classification, localization 및 detection에 대한 통합 프레임워크를 제공하는 것을 목표로 개발이 되었으며, 2013년 ILSVRC 대회에서 localization 부문에서 1위를 차지하였고, classification과 detection 분야에서도 우수한 성적(각각 4위와 3위)을 보였다. 2012년 ILSVRC에서 우승을 차지한 AlexNet은 classification 분야는 상세한 설명을 하고 있지만, localization에 대한 설명은 거의 없으며, 2013년부터 새롭게 대회에 추가된 detection 분야는 고려하고 있지 않다.
   OverFeat 설계팀은 classification, localization  및 detection 각 분야에 ConvNet을 효과적으로 사용할 수 있음을 보여주는 것을 논문의 목표로 삼았다. 실제로 2013년에 ILSVRC에서 3분야에 모두 출전한 팀은 OverFeat 팀이 유일하다. 그런데 Classification 분야에서는 뒤에 GoogLeNet, VGGNet 및 ResNet과 같은 구조가 속속 발표되면서 역시 관심에서 멀어지게 된다. 또한 시기적으로는 OverFeat의 발표시점보다 늦지만, 2013년에 발표된R-CNN([Part Ⅴ. Best CNN Architecture] 5. GoogLeNet  참고) 이 selective search 방법을 적용하여OverFeat보다 큰 성능차를 보이면서, OverFeat의 가치가 빛을 잃게 된다.
   뒤에 자세히 살펴보겠지만, OverFeat는 1-pass로 연산이 가능한 구조를 취하고 있기 때문에, 약 2000여개의 후보 영역에 ConvNet 연산을 해야 하는 R-CNN 보다 연산량 관점에서 효과적이었으며, 뒤에 R-CNN의 문제점을 해결한 SPPNet 설계자들에게 힌트가 되었을 것 같다. SPPNet 역시 1-pass 구조를 취하고 있는데, 이들은 OverFeat이 사용한 dense evaluation방법보다 더 진보된 Spatial Pyramid Pooling(SPP) 방식을 사용한다.
   ConvNet을 사용하여 detection을 개선시키는 방식은 애독자들의 요청에 따라, ResNet에 대한 구조 설명이 끝나고 나면 따로 상세하게 다룰 예정이다.
   Classification, Localization & Detection
   classification은 특정 대상이 영상 내에 존재하는지 여부를 판단하는 것을 말하며, 보통은 5개의 후보를 추정하고 그것을 ground-truth와 비교하여 판단한다. 5개의 후보 중 하나가 ground-truth와 맞는 것이 있으면 맞는 것으로 보며, 이것을 top-5 에러율로 표현하여 classification의 성능을 비교하는 지표로 삼는다. 아래 그림은 leopard 를 제대로 인식할 수 있는지 확인하는 것이다.
   localization은 bounding box를 통해 물체의 존재하는 영역까지 파악하는 것을 말하며, classification과 같은 학습 데이터를 사용하고 최대 5개까지 bounding box를 통해 후보를 내고 ground-truth와 비교하여 50% 이상 영역이 일치하면 맞는 것으로 본다. 성능 지표는 에러율(error rate)이다.
   detection은 classification/localization과 달리 200class 학습 데이터를 사용하며, 영상 내에 존재하는 object를 가능한 많이 추정을 하며, 경우에 따라서 없는 경우는 0이 되어야 하고, 추정이 틀린 false positive에 대해서는 감점을 준다. 전에 R-CNN 검토할 때 보았던 것처럼 보통은 mAP(mean Average Precision)으로 결과를 나타낸다.
   OverFeat
   OverFeat은 fast와 accurate 2개의 model이 있으며, fast model은 AlexNet과 마찬가지로 5 layer의 convolution layer/pooling layer 및 3개의 fully connected layer로 구성이 된다. Accurate model은 연산 시간은 좀 더 걸리더라도 정확도 향상을 위해 convolution/pooling layer가 1개 더 추가가 되었다. 아래 표는 fast model 구조에 대한 것이다.
   위 표에서 볼 수 있는 것처럼, 전형적인 구조를 취하고 있으며, VGGNet의 주장처럼, LRN(Local Response Normalization)은 별 효과가 없다고 하여 사용하지 않고 있으며, AlexNet과 같은 overlapped pooling 대신에 non-overlapped pooling 방식을 사용하고 있다.
   학습(Training) 방법
   Classification은 학습에는 모든 학습 영상을 256 크기로 scaling 하고 221x221 크기로 multi-crop을 한다. AlexNet과 마찬가지로 코너에서 4개 중앙에서 1개, 총 5개의 영상을 무작위로 고르고, 이것들과 좌우반전 영상 총 10개의 영상을 학습 영상으로 사용한다.
   Testing 방법 - Multi-scale dense evaluation
   OverFeat은 muti-crop voting 방식 대신에 dense evaluation 방식을 사용하였다. Multi-corp voting을 사용하는 경우는 서로 겹치는 부분이 있더라도 ConvNet 연산을 전부 새롭게 해야 되지만, dense evaluation 방식을 사용하여 이 문제를 해결하였다.
   Dense evaluation 방식의 개념은 다음 그림과 같다. 이 그림은 이해를 돕기 위해 1차원으로 표시한 것이다.
   만약에 최종 max-pooling layer에 총 20개의 데이터가 있고, 이것에 대해 3x1 non-overlapped pooling을 한다고 해보자. Pooling을 거치고 나면 데이터의 양이 1/3로 줄어드는 것뿐만 아니라, resolution 역시 1/3 로 줄어들기 때문에 위치의 정확도도 그만큼 떨어지게 된다.
   그런데 위 그림처럼, offset Δ를 1 데이터 단위로 하게 되면, 1 데이터 간격으로 각각 pooling을 진행하고, 그 결과를 Classifier에 적용을 하게 되면, pooling 이전의 해상도를 유지할 수 있어, pooling 이후 1번만 classifier 연산을 하는 것에 비해 훨씬 조밀한 검사(dense evaluation)을 할 수 있게 되는 것이다.
   아래는 accurate model의 구성을 보여주는 그림이다.
   Accurate model에서 stride 부분을 주목해서 살펴보면, Layer 1에서 conv와 pooling을 거치면 영상의 크기가 1/6로 줄어든다. Layer2를 거치면 다시 1/2로 줄어들고, Layer6의 pooling을 거치면, 다시 1/3이 줄어들어 전체적으로 Layer6를 거치고 나면, Layer7의 classifier로 입력의 1 픽셀은 실제 입력 영상의 36픽셀에 해당하게 된다.
   OverFeat의 입력 영상의 크기가 221x221인 것을 감안하면, Layer7의 1 픽셀의 해상도가 36픽셀에 해당하기 때문에 너무 간격이 듬성 듬성하게 된다.
   그래서 OverFeat 설계자가 고려한 부분은 classifier에 연결되는 Layer6의 pooling 부분에 주목을 하였다. Pooling의 stride가 3x3이기 때문에 pooling 단 앞단의 해상도는 36픽셀이 아니라 12 픽셀 수준이 되며, pooling 이전 단 기준으로 1픽셀씩 offset을 갖고 pooling을 수행하면 classifier로 들어가는 데이터의 양은 결과적으로 3x3, 즉 9배만큼 많아지지만, 해상력은 그만큼 좋아지게 된다.
   OverFeat에서 설계팀은 Fully-connected layer에 대한 해석을 다르게 하였다. LeCun이 FC layer에 대해 언급한 유명한 말은 다음과 같다.
   In Convolutional Nets, there is no such thing as "fully-connected layers". There are only convolution layers with 1x1 convolution kernels and a full connection table.
   LeCun은 FC-layer를 1x1 convolution으로 이해를 하고 있다. LeCun의 말처럼, FC-layer를 1x1 convolution으로 본다면, conv/pooling layer를 처리할 때와 과 마찬가지로 sliding window 개념을 적용할 수 있다.
   그간 ConvNet 설계자들이 어려워했던 부분은 convolution 부분은 영상의 크기에 상관없이 적용이 가능하지만, Fully-connected layer 부분은 fixed size를 갖고 있기 때문에 sliding window 개념이 아니라, ConvNet 앞단에서 fixed size를 고려한 crop을 수행하고 항상 FC layer 앞단에는 동일한 크기의 feature map이 확보되도록 하였다.
   그렇지만, FC layer가 개념적으로 보면 1x1 convolution으로 볼 수 있기 때문에 이제는 FC-layer 앞단의 feature-map 크기에 연연해 할 이유가 없어지는 것이다.
   이 개념을 2차원 구조를 갖는 영상에 적용하게 되면, 간단하게 예시를 하면 아래 그림과 같다.
   상단은 학습 과정을 보여준다. 입력 이미지의 크기가 14x14인 경우에 conv/pooling을 거처 최종적으로 5x5 크기를 얻은 후 classifier를 거쳐 학습을 한다. 하단은 입력 영상보다 크기가 큰 16x16 영상을 test 입력으로 사용하게 된다면, FC-layer 앞단에서는 6x6 크기의 feature-map이 얻어지고, 이것은 sliding window 개념으로 해석하면, 5x5 window 4개가 있는 것으로 볼 수 있고, 최종적으로 2x2 차원의 최종 출력을 얻게 된다.
   이 개념이 확보되었으므로 이제는 큰 이미지의 특정 위치를 무작위로 선택하는 것이 아니라 일정한 resolution 단위로 선택을 할 수 있게 된다. 또한 영상의 scale이 바뀌더라도, 바뀐 scale에 맞춰 sliding window를 움직이면서 결과를 얻으면 된다.
   아래 그림은 4개의 scale에 대해 dense evaluation을 보여주는 그림이다. Scale에 따라 특정 대상이 나타났다가 사라질 수 있으며, voting 개념을 활용하여 대상을 classification 및 localization 시킬 수 있다.
   Localization은 대상의 위치나 형태에 맞춰 bounding box까지 학습을 해줘야 한다. Classification과 동일한 학습이미지에 대해 bounding box까지 있는 ground-truth 데이터를 이용해 localization을 학습시킨다.
   Classification과 많은 부분을 공유하며, 최종단에 있는 classifier 부분을 bounding box regression network로 치환해주고, bounding box를 각각의 위치 및 scale에 맞게 학습을 시킨다.
   아래 그림은 동일 영상에 대해 voting에 의해 특정 대상이 검출되고 대상에 맞춰 bounding box를 찾아주는 것을 보여주는 그림이다.
   crop의 경우처럼 원 학습 영상으로부터 영상을 잘라낸 후 그것들을 각각 ConvNet에 적용시키는 방식이 아니라, 큰 영상에 대해 곧바로 ConvNet을 적용하고 일정한 픽셀 간격(grid)으로 마치 sliding window를 적용하듯이 결과를 끌어낼 수 있어, 연산량 관점에서는 매우 효율적이지만, grid 크기 문제로 인해서 학습 결과가 약간 떨어질 수 있다. 그러므로 crop과 dense evaluation을 상보적으로(complementary) 섞어 사용하면 더 성능이 좋아진다.
   OverFeat결과
   아래는 OverFeat의 실험 결과이다.
   이 결과를 보면, model voting과 4 scale, dense evaluation을 사용한 경우 top-5 에러율이 13.24로 상당히 좋은 결과를 낸다는 것을 알 수 있다. Dense evaluation을 사용하면, 겹치는 부분에 대한 연산을 공유할 수 있어 전반적으로 연산량이 크게 줄어든다. Grid 크기에 대한 우려가 있을 수 있지만, grid 크기를 연산량과 고려하여 적절한 수준으로 유지하면 결과가 좋다.
   Localization 결과는 아래 그림과 같다.
   Detection 분야는 대회를 마치고 다시 개선한 결과에서는 mAP가 24.3%로 1위를 차지할 수 있었지만, 결과적으로 좀 뒤에 나온 R-CNN이 동일 데이터 set에 대해 큰 폭으로 성능 차이를 보이게 되어 의미가 많이 퇴색해졌다.
   앞서 살펴본 것처럼, OverFeat는 ConvNet의 fully-connected layer를 1x1 convolution 개념으로 볼 수 있다는 점을 이용하여 연산량 절감 차원에서는 효과적이었지만, 근본적으로는 deeper network 이상의 성능을 얻을 수는 없었다. 하지만 ConvNet을 이용하여 localization/detection까지 통합 시도를 했다는 점에서 의미가 있고, 1년 뒤에 마이크로소프트 팀에서 발표할 SPPNet에 맥이 닿아 있다는 점에서 의미가 있는 것 같다. SPPNet은 dense evaluation 대신에 spatial pyramid pooling을 적용하여 detection 의 성능과 속도를 모두 개선시킬 수 있었다.
   다음 Class부터 ResNet으로 넘어갈 예정이다. ResNet의 성능은 압도적이고 매력적이다. 하지만, ResNet은 이전 ConvNet들과 개념과 구조가 좀 다르기 때문에 몇 번의 class에 걸쳐 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [1]
   ResNet(Residual Net)의 설계자 ‘Kaiming He’는 현재는 Facebook으로 자리를 옮겼지만, 금년 초까지 마이크로소프트 북경연구소에서 근무하면서 CNN의 구조에 관련된 좋은 논문을 많이 발표했다. 특히 2015년, R-CNN의 속도 문제를 해결한 SPPNet(Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)은 ConvNet을 이용하여 detection 분야의 성능을 크게 개선시켰으며, 이때부터 이미 주목을 받게 되었다. 아마도 ResNet이 classification뿐만 아니라 localization 및 detection 분야에서도 압도적인 성능을 보인 이유는 이처럼 선행 연구들이 많았기 때문인 것 같다.
   그 동안 캐나다의 토론토 대학교, 뉴욕대, 스탠포드/버클리 출신의 Google 연구팀 및 영국의 Oxford 대학교 등 주로 서구 쪽 연구팀이 ILSVRC에서 우승을 했다면, 2015년 ResNet 팀은 주로 마이크로소프트의 북경연구소에 있는 중국 연구진들이 성과를 냈다는 점에서 우리도 자극을 좀 받아야 할 것 같다.
   ResNet은 Residual learning이라는 생소한 이름이 등장하고, Shortcut Connection및 Identity mapping 등 다른 CNN 구조에서 볼 수 없었던 용어들이 사용이 되고, layer 수도 152 layer로 엄청나게 깊은(ultra-deep) 구조를 갖고 있는 등 이전에 살펴본 다른 CNN 구조와는 많이 달라 다소 어렵게 느껴질 수 있다. 하지만 하루 아침에 만들어진 구조는 아니기 때문에 하나씩 차근차근 살펴보면 충분히 이해할 수 있으리라 생각한다.
   이번 Class부터 몇 회에 걸쳐, ResNet에 대하여 살펴볼 예정이다.
   깊은 망의 문제점
   이미 앞선 Class를 통해 더 깊은(deeper) 망이 학습 데이터 속에 존재하는 대표적인 개념을 잘 추출할 수 있어 학습 결과가 좋아진다는 것은 확인을 하였다.
   하지만 망이 깊어지게 되면, 그 망이 좋은 결과를 낼 수 있도록 학습을 시키기가 점점 더 어려워지며, 그 원인은 크게 보면 다음 두 가지로 정리할 수 있다.
   l  Vanishing/Exploding Gradient 문제: CNN에서 파라미터 update를 할 때, gradient값이 너무 큰 값이나 작은 값으로 포화되어 더 이상 움직이지 않아 학습의 효과가 없어지거나 학습 속도가 아주 느려지는 문제가 있다. 망이 깊어질수록 이 문제는 점점 더 심각해지며, 이 문제를 피하기 위해 batch normalization, 파라미터의 초기값 설정 방법 개선 등의 기법들이 적용되고 있지만, layer 개수가 일정 수를 넘어가게 되면 여전히 골치거리가 된다.
   l  더 어려워지는 학습 방법: 망이 깊어지게 되면, 파라미터의 수가 비례적으로 늘어나게 되어 overfitting의 문제가 아닐지라도 오히려 에러가 커지는 상황이 발생한다.
   ResNet 팀의 실험 ? 망이 깊어졌을 경우의 학습 결과
   前에 VGGNet을 ImageNet 데이터를 이용해 학습을 시키는 경우 16-layer와 19-layer의 결과가 거의 차이가 없는 것을 확인하였다. 만약에 19-layer 이상으로 layer 수를 증가시킨다면 어떤 결과가 나타났을까?
   VGGNet 설계자들이 더 많은 layer수에 대한 실험을 했는지는 확인할 길이 없지만, layer수를 더 이상 증가를 시키게 되면 결과는 나빠지는 방향으로 나올 것은 분명하다.
   ResNet팀이 망이 깊어지는 경우 어떤 결과가 나오는지 실험을 하였다. 짧은 시간에 실험의 효과를 얻기 위해 ImageNet보다는 간단한 CIFAR-10 학습 데이터를 20-layer와 56-layer에 대하여 비교 실험을 하였으며, 결과는 아래 그림과 같다.
   실험 결과를 보면, 학습 오차와 테스트 오차 모두 56-layer의 결과가 20-layer보다 나쁘게 나오는 것을 확인할 수 있으며, 다른 학습 데이터에 대하여 layer 수를 달리하면서 실험을 해봐도 학습 결과가 오히려 더 나빠지는 것을 확인할 수 있었다.
   망이 깊어지면 결과가 더 좋아져야 하는데 왜 이런 결과가 나온 것일까? 이 부분이 본 논문의 저자의 핵심 이슈가 되고, 이것을 해결하기 위해 deep residual learning의 개념이 나오게 된다.
   Residual Learning
   ResNet 설계 팀은 망을 100 layer 이상으로 깊게 하면서, 깊이에 따른 학습 효과를 얻을 수 있는 방법을 고민하였으며, 그 방법으로 Residual Learning이라는 방법을 발표하였다. Residual이라는 단어를 사용한 이유는 아래 구조를 살펴보면 파악할 수가 있다.
   먼저 아래 구조와 같은 평범한 CNN 망을 살펴보자. 이 평범한 망은 입력 x를 받아 2개의 weighted layer를 거쳐 출력 H(x)를 내며, 학습을 통해 최적의 H(x)를 얻는 것이 목표이며, weight layer의 파라미터 값은 그렇게 결정이 되어야 한다.
   그런데 위 네트워크에서 H(x) 를 얻는 것이 목표가 아니라 H(x) - x를 얻는 것으로 목표를 수정한다면, 즉 출력과 입력의 차를 얻을 수 있도록 학습을 하게 된다면, 2개의 weighted layer는 H(x) ? x를 얻도록 학습이 되어야 한다. 여기서 F(x) = H(x) - x라면, 결과적으로 출력 H(x)는 H(x) = F(x) + x 가 된다.
   그러므로 위 블락은 아래 그림처럼 바뀌며, 이것이 바로 Residual Learning의 기본 블락이 된다. 달라진 점이 있다면, 입력에서 바로 출력으로 연결되는 shortcut 연결이 생기게 되었으며, 이 shortcut은 파라미터가 없이 바로 연결이 되는 구조이기 때문에 연산량 관점에서는 덧셈이 추가되는 것 외에는 차이가 없다.
   이렇게 관점을 조금 바꿨을 뿐이지만, 꽤 많은 효과를 얻을 수 있다. 예전에는 H(x)를 얻기 위한 학습을 했다면, 이제는 H(x) ? x를 얻기 위한 학습을 하게 되며, 최적의 경우라면 F(x)는 0이 되어야 하기 때문에 학습할 방향이 미리 결정이 되어, 이것이 pre-conditioning 구실을 하게 된다. F(x)가 거의 0이 되는 방향으로 학습을 하게 되면 입력의 작은 움직임(fluctuation)을 쉽게 검출할 수 있게 된다. 그런 의미에서 F(x)가 작은 움직임, 즉 나머지(residual)를 학습한다는 관점에서 residual learning이라고 불리게 된다.
   또한 입력과 같은 x가 그대로 출력에 연결이 되기 때문에 파라미터의 수에 영향이 없으며, 덧셈이 늘어나는 것을 제외하면 shortcut 연결을 통한 연산량 증가는 없다. 그리고 몇 개의 layer를 건너 뛰면서 입력과 출력이 연결이 되기 때문에 forward나 backward path가 단순해지는 효과를 얻을 수 있다.
   결과적으로 identity shortcut 연결을 통해 얻을 수 있는 효과는 다음과 같다
   .
   l  깊은 망도 쉽게 최적화가 가능하다.
   l  늘어난 깊이로 인해 정확도를 개선할 수 있다.
   이번 Class에서는 Residual Learning의 기본 개념과 이것이 도출된 개념에 대하여 살펴보았다. 다음 Class에서는 ResNet이 실제로 그 효과를 얻을 수 있는지 ResNet 팀의 실험 결과를 살펴보고, 좀 더 ResNet을 깊이 있게 이해할 수 있도록 할 예정이다.

[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [2]
   지난 [Part Ⅴ. Best CNN Architecture] 8. ResNet [1] 에서는 Residual Learning의 기본 개념에 대해 살펴보았다. Residual Learning 이란 개념이 고안된 이유는 VGGNet과 같은 기존 방식으로는 어느 일정 정도 이상의 layer 수를 넘어서게 되면, 결과가 나빠지는 문제를 해결하기 위함이었다.
   실제로 ResNet 팀은 2015년 ILSVRC에 출전하여 classification, localization 및 detection 분야에서 모두 우승을 차지했으며, classification에서는 152 layer를 사용하여 top-5 error율이 3.57%로, 훈련을 받은 사람이 이미지를 분류할 때의 오차율(약 5%정도)보다 더 좋은 결과를 얻었으며, 아래 그림과 같다.
   2014년 우승을 차지한 GoogLeNet보다 약 두배 성능이 좋았으며, 망의 깊이도 7배 이상으로 깊어졌다. GoogLeNet 팀은 이후 계속 자신들의 CNN 구조를 개선하여 2016년 2월Inception-V4/Inception-ResNet까지 발표를 하였으며, ResNet보다 더 좋은 결과를 얻었음을 발표하였다. (논문: Inception-V4, Inception-ResNet and the Impact of Residual Connections on Learning) a 뒤에 따로 다룰 예정.
   이번 Class에서는 ResNet 팀의 실험 결과를 통해, 정말로 Residual Learning방법을 적용하면 이런 문제가 해결이 되는지를 확인해 볼 예정이다.
   ResNet 팀의 실험
   ResNet 팀은 실험을 위한 망을 설계하면서 VGGNet의 설계 철학을 많이 이용하였다. 그래서 대부분의 convolutional layer는 3x3 kernel을 갖도록 하였으며, 다음 2가지 원칙을 지켰다. 또한 복잡도(연산량)를 줄이기 위해 max-pooling(1곳 제외), hidden fc, dropout 등을 사용하지 않았다.
9. 출력 feature-map 크기가 같은 경우, 해당 모든 layer는 모두 동일한 수의 filter를 갖는다.
10. Feature-map의 크기가 절반으로 작아지는 경우는 연산량의 균형을 맞추기 위해 필터의 수를 두 배로 늘린다. Feature-map의 크기를 줄일 때는 pooling을 사용하는 대신에 convolution을 수행할 때, stride의 크기를 “2”로 하는 방식을 취한다.
    이들은 비교를 위해 평범한 망(plain network)과 residual network으로 구별을 하였다. Plain network의 경우도 VGGNet 보다 filter의 수를 줄이고 복잡도를 낮춤으로써 34-layer의 plain network이 19-layer의 VGGNet에 비해 연산량을 20% 미만으로 줄였다. (VGGNet 19-layer: 19.6 billion FLOPs, ResNet 팀이 설계한 34-layer Plain setwork: 3.6 billion FLOPs)
    Residual network도 구조를 간단하게 하기 위해, 매 2개의 convolutional layer마다 shortcut connection이 연결되도록 하였다.
    이들이 실험에 사용한 망의 구조는 다음 그림과 같다. (그림은 대표적으로 34-layer의 구조만을 보여준다)
    또한 실험에는 18-layer, 34-layer, 50-layer, 101-layer 및 152-layer에 대하여 수행을 하였으며, 각각에 대한 layer 구성은 다음 표와 같다.
    위 표를 보면 18-layer와 34-layer는 동일한 구조를 사용하였고, 다만 각 layer에 있는 convolutional layer 수만 다르다는 것을 알 수가 있다.
    하지만 50-/101-/152- layer의 경우는 18-34-layer와 구조가 다르다는 것을 알 수가 있는데, 그 이유는 뒤에서 살펴보기로 한다.
    실험 결과
    먼저 실험 18-layer와 34-layer에 대한 Plain network과 Residual network의 실험 결과를 살펴보면 다음 그림과 같다.
    Plain network의 경우는 34-layer의 결과가 18-layer의 결과보다 약간 나쁘다는 것을 알 수가 있으며, Residual network은 34-layer가 18-layer 결과보다 좋은 것을 알 수가 있다. 아래 표는 각각의 경우에 대한 top-1 error 율을 비교한 결과이다.
    또 한가지 주목할 점은 학습의 초기 단계에서 residual net의 수렴 속도가 plain network보다 빠르다는 점이다.
    이 실험 결과를 살펴보면, residual network이 plain network에 비해 더 좋은 결과를 내면서도 빠르다는 것을 알 수 있다.
    Deeper Bottleneck Architecture
    학습에 걸리는 시간을 고려하여 50- /101-/152-layer에 대해서는 기본 구조를 조금 변경을 시켰으며, residual function은 1x1, 3x3, 1x1으로 아래 그림처럼 구성이 된다. Bottleneck 구조라고 이름을 붙인 이유는 차원을 줄였다가 뒤에서 차원을 늘리는 모습이 병목처럼 보이기 때문이라고 생각된다.
    이렇게 구성한 이유는 연산 시간을 줄이기 위함이다. 먼저 맨 처음 1x1 convolution은 NIN(Network-in-Network)이나 GoogLeNet의 Inception 구조에서 살펴본 것처럼 dimension을 줄이기 위한 목적이며, 이렇게 dimension을 줄인 뒤 3x3 convolution을 수행 한 후, 마지막 1x1 convolution은 다시 dimension을 확대시키는 역할을 한다. 결과적으로 3x3 convolution 2개를 곧바로 연결시킨 구조에 비해 연산량을 절감시킬 수 있게 된다.
    Deeper layer (50-/101-/152-layer) 실험 결과
    Single model에 대한 실험 결과는 아래 표와 같다. 152-layer에 대한 top-5 error율은 4.49% 수준까지 떨어졌다. Single model 만으로도 과거에 발표된 어떤 구조보다도 좋은 결과를 얻을 수 있었다.
    최종적으로 ILSVRC에 제출한 결과는 아래와 같다. 결과를 제출할 당시는2개의 152-layer 결과를 조합해서 얻은 결과이며, (이는 다른 팀들도 결과를 좋게 하기 위해 ensemble을 사용하는 것처럼..)  최종적으로는 depth를 달리하는 6개 모델에 대한 ensemble로 사용하였다.
    이번 class에서는 ResNet 팀이 결과를 얻기 위해 실험한 모델에 대하여 상세하게 살펴보았다. 기본적으로 VGGNet과 유사한 구조를 사용했지만, 연산량의 균형을 맞추기 위해 VGGNet과는 다른 모델로 발전을 하게 되었으며, Residual network 구조를 사용하여 152-layer라는 아주 깊은 망을 이용해 아주 뛰어난 결과를 얻었고, 이렇게 깊은 망에서도 충분히 학습이 가능하다는 것을 확인 할 수 있었다.
    다음 class에서는 ImageNet 모델뿐만 아니라, 다른 모델에서도 일반화가 가능한지를 확인하고, classification 성능뿐만 아니라, localization 및 detection 성능도 확인을 해 볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [3]
   지난 [Part Ⅴ. Best CNN Architecture] 8. ResNet [2] 에서는 ResNet 팀의 실험 결과를 통하여 Residual Learning 방법이 과연 효과가 있는지 확인하였다. VGGNet 의 기본 구조를 약간 변형시킨 Plain Network과 Residual Network을 비교 실험하였으며, Residual Network를 34/50/101/152-layer에 적용하였을 경우 망이 깊어지더라도 Plain Network과 달리 학습 결과가 열화되는 현상이 나타나지 않음을 확인하였다. 이들은 2015년 ILSVRC 대회에 출전하여 모든 분야에서 압도적인 성능차를 보이면서 우승을 하였다.
   이번 Class에서는 ResNet 구조가 과연 ImageNet 데이타뿐만 아니라 다른 학습 데이터에서도 좋은 결과를 내는지 확인을 해보고, Residual Learning에 대한 일반화가 가능한지를 확인해 볼 예정이다.
   CIFAR-10 데이터에 대한 실험
   CIFAR-10 영상 데이터 집합은 32x32 크기의 작은 영상 데이터들의 집합이며, 학습 이미지 5만장과 test 이미지 1만장으로 구성되어 있다. 총 10 개의 class가 있으며, 각각의 class에는 학습과 검사 영상을 합해 총 6000장이 있다.
   CIFAR-10 영상데이타는 유명한 토론토 대학교의 Alex Krizhevsky, Vinod Niar 및 Geoffrey Hinton이 수집을 하였으며, 영상의 크기가 작아 컴퓨터 비전이나 머신 러닝의 초기 알고리즘이나 구조 검증에 많이 사용이 된다. 10개의 class에 대한 예는 다음 그림과 같다.
   ResNet 팀은 CIFAR-10에 대한 검증을 위해 망 구성을 약간 변형을 시켰다. 그 이유는 ImageNet(ILSVRC) 데이타의 경우는 224x224로 영상의 크기가 컸지만, CIFAR-10에 있는 영상이 32x32로 크기가 작기 때문이다. 검증 방식은 동일한 layer 수를 갖는 Plain과 Residual Network을 비교 실험하는 것이다.
   ImageNet 데이타를 이용한 실험에서는 맨 처음 convolutional layer에 7x7 kerenl 사이즈를 사용했지만 ([Part Ⅴ. Best CNN Architecture] 8. ResNet [2] 참고), CIFAR-10의 영상은 크기가 매우 작기 때문에, 맨 처음 convolutional layer에 3x3 kernel을 썼다.
   다음은 6n개의 3x3 convolutional layer 그룹이 온다. 6n 중 각각의 2n에 대하여 feature map의 크기가 {32, 16, 8}이 되도록 하였으며, filter의 개수는 연산량의 균형을 맞춰주기 위해 {16, 32, 64}가 되도록 하였다.
   맨 마지막에 global average pooling과 10-way softmax를 배치하였으며, 결과적으로 전체 layer의 수는 “6n + 2”가 되었다.
   전체적인 블락의 구성은 아래 그림과 같다.
   이 그림은 개념적 이해를 돕기 위해 간단하게 14-layer를 갖는 구조만을 보여주는 것이며, layer수가 많아지면 각각의 2n 그룹에 layer를 추가하면 된다. 실제로 ResNet 팀은 실험 시 n값을 바꿔가면서 더 깊은 layer에서 어떤 결과가 나타나는지 비교 실험을 하였다.
   n = {3, 5, 7, 9}로 설정을 하면 20/32/44/56-layer를 얻을 수 있으며, 이것들에 대한 실험 결과는 다음 그림과 같다. ImageNet 데이터를 이용한 실험과 거의 동일한 양상의 결과를 보여준다.
   Plain 망은 일정 layer수가 넘어서면 layer 수가 증가함에 따라 결과가 더 나빠지는 경향성을 보이지만, Residual 망은 layer 수가 증가하더라도 결과가 더 좋아짐을 알 수가 있다. 특히 Residual 망에 대해서는 n = 18로 설정하여  101 layer까지 실험을 하였는데 56-layer 때에 비해 성능이 더 좋아지는 것을 확인할 수 있다.
   Layer별 Response에 대한 분석
   Plain 망과 Residual 망에 대하여 각각의 layer에서의 response에 대한 표준 편차를 살펴보았다. 이 결과는 Batch Normalization은 실행을 하고 ReLU 및 addition은 실행하기 전을 비교한 것이며, 결과는 아래 그림과 같다. 이 그림 중 밑에 있는 그림은 표준편차를 크기 별로 다시 sorting한 결과이다.
   이 결과를 보면, Residual 망의 response가 Plain 망에 비해 작다는 것을 확인할 수 있다.
   Residual 망의 결과가 Plain 망의 결과에 비해서 표준편차가 작으며, 이 실험 결과를 통해 망의 response가 크게 흔들리지 않는 것을 확인할 수 있었으며, 결론적으로 이런 이유로 인해 망이 깊어졌을 때 Plain 망에서 발생하는 문제가 Residual 망에서는 적게 발생하게 된다.
   1000 layer 넘었을 때 결과
   추가 실험으로, n = 200으로 설정하여 1202-layer의 깊이를 갖는 Residual 망에 대한 실험을 해봤는데, 110 layer 경우보다 결과가 약간 나쁘게 나왔다. 하지만 1000-layer가 넘는 경우에도 최적화에 별 어려움이 없었음을 언급하였다.
   1000-layer가 넘었을 때, 결과가 더 나쁘게 나온 이유는 망의 깊이에 비해 학습 데이터의 양이 부족했고, 결과적으로 학습 데이터에만 특화된, 즉 overfitting된 결과가 나왔을 것으로 추정한다.
   이 실험에서는 maxout이나 dropout과 같은 별도의 regularization 기술은 적용을 하지 않았으며, 단지 망을 깊게 했을 때 어떤 결과가 나오는지를 확인하기 위한 목적이었이며, 추가적인 regularization 기법을 적용하면 이것보다는 결과를 개선시킬 수 있을 것이라 생각된다.
   그리고 이것은 다음 해 Kaiming He 가 발표한 논문(Identity Mappings in Deep Residual Networks)을 보면. Pre-activation을 통해 개선이 되는 것을 확인할 수 있다. 이 논문에서는 2015년 발표한 Residual Network의 구조를 개선한 새로운 구조를 소개하였다. (이것은 추후 다룰 예정)
   다른 실험 결과 (2016년)
   Kaiming He가 2016 년 논문을 통해 발표한 실험 데이터를 보면, CIFAR-10뿐만 아니라 CIFAR-100에 대해서도 비슷한 실험을 하였으며, 그 결과는 아래 표와 같다. 이 실험은 단지 경향성을 확인하기 위한 실험이며, 그렇기 때문에 적당한 수준의 data augmentation만 적용 했음을 밝히고 있다.
   1001-layer에서는 pre-activation을 통해 기존 Residual 망의 이슈가 개선되었다는 사실을 확인할 수 있다.
   이번 class에서는 Residual 망을 ImageNet 데이터뿐만 아니라, CIFAR-10 및 CIFAR-100 데이터에 대한 실험을 통해 Residual 망이 효과적임을 파악하였다. 뿐만 아니라, 1000-layer가 넘어가는 경우에도 효과가 있음을 확인하였다.
   다음 class에서는 classification 성능뿐만 아니라, localization 및 detection 성능도 확인을 해 볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [4], Fast-RCNN
   지난 [Part Ⅴ. Best CNN Architecture] 8. ResNet [1] ~ 8. ResNet [3] 를 통해 Residual Learning은 망을 1000-layer 이상으로 깊게 만들 수 있게 해주고, image classification의 성능도 기존에 발표된 그 어떤 CNN 방법보다 뛰어나다는 사실을 확인하였다.
   그렇다면, image localization이나 detection 에서도 ResNet을 사용하는 것이 과연 효과가 있을까?
   결론부터 이야기 하면, 아래의 표처럼 2015년 ILSVRC에서 localization과 detection 모두 2위와 상당한 격차를 보이며 우승한 것으로 볼 때, ResNet이 효과가 있음을 알 수 있다.
   ResNet에서는 image detection/localization의 성능을 위해, Faster R-CNN 방법을 적용하였다. Faster R-CNN을 이해하려면, R-CNN, SPPNet, Fast R-CNN 및 Faster R-CNN으로 이어지는 detection 분야의 계보를 이해해야 한다.
   Image detection/localization에 관련된 내용은 GoogLeNet을 설명하는[Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [6]에서 R-CNN과 SPPNet을 다룬 바 있다.
   R-CNN은 2013년 당시 버클리에 다니고 있던 Ross Girshick등이 발표했으며, CNN을 image detection 분야에 적용하여, SIFT나 HOG와 같은 gradient 기반의 computer vision 알고리즘을 적용한 것보다 훨씬 좋은 결과를 거두었다. 또한 이후 연구자들이 image detection 분야에 CNN을 적용한 연구에 불을 붙이게 되는 도화선 역할을 한다.
   R-CNN은 성능은 뛰어나지만 속도가 느린 문제가 있다. R-CNN의 느린 속도 문제를 개선한 SPPNet은 2014 Kaiming He(ResNet 발표자) 등이 발표를 했으며, Spatial Pyramid Pooling이라는 독특한 개념을 적용하여 속도를 개선시킨다.
   이후 2015년 Ross Girshick이 마이크로소프트로 자리를 옮겨 R-CNN과 SPPNet의 문제를 개선한 Fast R-CNN을 발표한다. 이후 Shaoqing Ren, Kaiming He 및 Ross Girshick이 공동 저자로 더 개선시킨 Faster R-CNN을 발표하였으며, 논문의 부제도 “Towards Real-time Object Detection with Region Proposal Networks”로 붙이고, 실시간 객체 검출을 목표로 하였다. 이 Faster R-CNN의 개념이 다시 ResNet으로 이어져 localization/detection 분야에서 성과를 보이게 된다.
   이번 Class에서는 ResNet 의 image detection/localization의 설명에 앞서 Fast R-CNN에 대하여 살펴볼 예정이다.
   R-CNN과 SPPNet의 문제점
   R-CNN(논문: Rich feature hierarchies for accurate object detection and semantic segmentation)과 SPPNet(논문: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)에 대한 기본 개념은 [Part Ⅴ. Best CNN Architecture] 5. GoogLeNet [6] 에서 설명을 하였다.
   R-CNN이 그 이전 알고리즘보다 뛰어난 성능을 보이기는 했지만, 다음과 같은 3가지 문제점을 갖고 있다.
9. Training이 여러 단계로 이루어짐.
   R-CNN은 크게 3단계 과정으로 이루어진다. 우선 약 2000여개의 후보 영역에 대하여 log loss 방식을 사용하여 fine tuning을 한다. 이후 ConvNet 특징을 이용하여 SVM에 대한 fitting 작업을 진행한다. 끝으로 bounding box regressor(검출된 객체의 영역을 알맞은 크기의 사각형 영역으로 표시하고 위치까지 파악)에 대한 학습을 한다.
10. Training 시간이 길고 대용량 저장 공간이 필요.
    SVM과 bounding box regressor의 학습을 위해, 영상의 후보 영역으로부터 feature를 추출하고 그것을 디스크에 저장한다. PASCAL VOC07 학습 데이터 5천장에 대하여 2.5일 정도가 걸리며, 저장 공간도 수백 GigaByte를 필요로 한다.
11. 객체 검출(object detection) 속도가 느림.
    학습이 오래 걸리는 문제도 있지만, 실제 검출할 때도 875MHz로 오버클럭킹된 K40 GPU에서 영상 1장을 처리하는데 47초가 걸린다.
    R-CNN의 개념을 잘 표현한 Girshick의 slide 그림은 아래와 같다. 앞서 살펴본 것처럼 후보 영역에 대하여 모든 연산을 순차적으로 수행하는 구조이기 때문에 느릴 수밖에 없다.
    R-CNN의 가장 큰 문제는 모든 객체 후보 영역에 대하여 개별적으로 연산을 한다는 점이며 이것 때문에 속도가 매우 느리다. SPPNet 설계자들은 Spatial Pyramid Pooling을 사용하여 convolution연산을 공유할 수 있는 방법을 제안하였다. 이를 통해 R-CNN보다 학습에서는 3배 정도 빠르고, 실제 적용 시 약 10~100배 정도 빠른 결과가 나왔다.
    하지만 SPPNet은 구조적인 관점에서는 R-CNN과 마찬가지로 학습에 3 단계 파이프라인을 적용하는 것은 동일하다. 그리고 이렇게 얻어진 feature들은 디스크에 쓰여져야 한다. 또한 R-CNN과 달리 fine-tuning 알고리즘을 적용할 때 Spatial Pyramid Pooling 앞단에 있는 convolutional layer에 대해서는 fine tuning을 하지 않는데, 이로 인해 deep network를 적용할 때는 정확도가 문제가 될 수도 있다.
    역시 Girshitck의 slide에서 SPPNet을 한눈에 설명한 구조가 있는데 아래 그림과 같다. SPPNet은 ConvNet 단계는 전체 영상에 대하여 한꺼번에 연산을 하고 그 결과를 공유하고, SPP layer를 거치면서 region 단위 연산을 수행한다.
    Fast R-CNN
    Fast R-CNN은 기본적으로 검출 정확도(mAP)가 R-CNN이나 SPPNet보다는 좋으며, 학습 시 multi-stage가 아니라 single-stage로 가능하고, 학습의 결과를 망에 있는 모든 layer에 update할 수 있어야 하며, feature caching을 위해 별도의 디스크 공간이 필요 없는 방법을 개발하는 것을 목표로 삼았다.
    Fast R-CNN의 기본 구조는 다음 그림과 같다. Fast R-CNN 망은 전체 이미지 및 객체 후보 영역을 한꺼번에 받아들인다. Convolution과 max-pooling을 통해 이미지 전체를 한번에 처리를 하고 feature map을 생성한다.
    그 후 각 객체 후보 영역에 대하여 ‘RoI Pooling layer’를 통해, feature-map으로부터 fixed-length feature 벡터를 추출한다. 이 부분은 SPPNet의 Spatial Pyramid Pooling과 하는 일이 유사하다고 볼 수 있다.
    이렇게 추출된 fixed-length feature vector는 Fully-Connected Layer에 인가를 하며, 뒷단에는 “object class + background”를 추정하기 위한 softmax부분과 각각의 object class의 위치를 출력하는 bbox(bounding box) regressor가 온다.
    Fast R-CNN의 구조를 앞서 그린 것과 비슷한 방식으로 표현을 하면 아래 그림과 같다. 먼저 test time 시에 구조적으로 SPPNet과 비슷하게 전체 영상에 대해 ConvNet 연산을 1번만 수행을 하고 그 결과를 공유하며, RoI Pooling layer에서 다양한 후보 영역들에 대하여 FC layer로 들어갈 수 있도록 크기를 조정해주는 과정을 거친다. 이후 단계는 Softmax classifier와 BBox regressor를 동시에 수행하기 때문에 multi-stage pipeline 과정을 순차적으로 수행하는 SPPNet에 비해 빠르다.
    Fast R-CNN의 training 과정을 보여주는 다음 그림을 보면, SPPNet이나 기존 R-CNN과 달리 일반 CNN처럼 모든 layer에 대하여 학습이 가능하며 전체를 1 stage로 학습이 가능하다. SPPNet에서 ConvNet 부분은 1단계에서 학습을 마치면 파라미터가 고정이 되고, 단지 마지막 3 layer(FCs + SVM/BBox Regressor)만 학습이 가능했었다. 하지만 Fast R-CNN은 1-stage 연산을 하기 때문에 ConvNet 부분까지 error를 역전파 시키면서 학습이 가능해졌고, 이를 통해 정확도에 대한 개선이 가능해졌다.
    Fast R-CNN에서 학습 시간을 줄인 비법
    R-CNN과 SPPNet은 학습 시에 mini-batch 크기를 128개로 정했고, 128개의 서로 다른 이미지로부터 128개의 RoI를 취하는 region-wise sampling 방법을 사용하였으며, 아래 그림과 같다.
    이 방법을 Fast R-CNN에 동일하게 적용하면 속도가 느려질 가능성이 있다. R-CNN은 모든 후보 영역 RoI를 224x224 크기로 scale 하여 사용하지만, Fast R-CNN에서는 원 영상을 그대로 사용하기 때문에 경우에 따라서RoI 영역에 해당하는 receptive field의 크기가 매우 클 수 있으며, 최악의 경우는 전체 이미지가 될 수도 있다. 그렇게 되면 영상 크기에 비례하여 연산량이 증가하여 느려질 수 있게 된다.
    이 문제를 피하기 위해 mini-batch를 구성할 때 hierarchical sampling 방식을 사용하였다. R-CNN, SPPNet처럼 서로 다른 128장의 학습 영상에서 RoI를 무작위로 선택하는 것이 아니라, 작은 수의 학습 영상 (논문에서는 2장으로 정함)으로부터 128개의 RoI를 정하도록 하였으며, 아래 그림과 같다.
    이렇게 하면, test time과 마찬가지로 training time에도 학습 영상의 결과를 공유할 수 있게 되어 연산 속도가 빨라지게 된다.
    Fast R-CNN 성능
    다음 그래프는 Girshick의 slide에서 따온 것이며, 2013년 CNN 기법을 적용한 알고리즘이 적용되면서 object detection 성능이 비약적으로 발전하는 것을 볼 수 있으며, 저자는 자신의 slide에서 “object detection renaissance”라는 표현을 하였다.
    아래 그림에서 볼 수 있는 것처럼, R-CNN 기법을 적용하면서 mAP가 PASCAL VOC 영상데이타에서 50%를 넘어가지 시작했으며, Fast R-CNN을 적용하였을 때 70% 수준에 이르게 된다.
    R-CNN, SPPNet, Fast R-CNN의 비교 실험 결과는 다음과 같다. (여기서 R-CNN과 SPPNet은 동일한 환경에서의 실험을 위해, VGGNet-16에 구조에 올려 실험한 결과이다)
    이번 class는 독자들의 요청도 있고, 또한 ResNet의 detection/localization을 이해하는데 Fast R-CNN을 이해하는 것이 필요하여 Fast R-CNN에 대한 설명을 하였다. 설명을 하다보니 다소 길어진 감이 있어, 실제 ResNet에 적용한 Faster R-CNN은 다음 class에서 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [5], Faster-RCNN
   지난 [Part Ⅴ. Best CNN Architecture] 8. ResNet [4], Fast-RCNN 에서는 Fast R-CNN에 대하여 살펴보았다. Fast R-CNN은 R-CNN과 SPPNet이 갖고 있던 object detection/localization알고리즘의 정확성과 속도 문제를 개선하였다.
   R-CNN과 Fast R-CNN을 속도 관점에서만 살펴보면 아래 표와 같다.
   이 표를 보면, Fast R-CNN은 Test time에 1장의 영상을 처리하는데 0.32s로 상당히 빨라진 것처럼 보인다. 그런데 이 시간에는 object가 있을 것이라고 생각되는 후보 영역을 계산하기 위한 region proposal 시간이 고려되지 않았다.
   그러므로 실제적으로 region proposal 시간까지 고려하면, 영상 1장을 처리하는데는 약 2초 정도가 걸리게 되며, 실시간 관점에서 보면 2초는 부담스러운 시간이 된다.
   위 표에서 살펴본 것처럼, Fast R-CNN이 Test time 처리시간을 R-CNN보다 많이 개선시키기는 하였지만, 여전히object detection을 위한 후보 영역을 계산하는 과정을 필요로 하고, 이 과정이 상당한 시간이 걸리는  점은 문제이다. 또한 region proposal이 별도의 과정으로 되어 있기 때문에 진정한 의미에서 1-pass로 구현했다고 볼 수 없다.
   그리하여 Faster R-CNN 설계자들은 region proposal을 위해 별도의 과정을 거치는 대신에 image classification을 위한 ConvNet에 region proposal을 위한 특수한 용도의 망(RPN)을 추가하여 Fast R-CNN의 문제를 해결하였다.
   이번 class에서는 Shaoqing Ren, Kaiming He 및 Ross Girshick이 공동 저자로 object detection/localization 성능을 대폭 개선한 논문 “Faster R-CNN: Towards Real-time Object Detection with Region Proposal Networks”를 중심으로 Faster R-CNN에 대하여 살펴볼 예정이다.
   R-CNN/Fast R-CNN과 Region Proposal
   먼저 지난 class에서 살펴본 Fast R-CNN을 동작을 다시 보면, 입력 이미지와 함께, 적절한 region proposal 방법을 통한 region proposal을 같이 받아들이며, 이것은 RoI Pooling layer를 거쳐 FC로 넘어가고 최종단에는 Softmax classifier 와 bounding box regressor가 온다는 것을 확인할 수 있다.
   R-CNN 때부터 Region proposal 방법으로 적용했던 방식은 “selective search” 방식이며, 이 과정은 아래 그림과 같다. 즉 bottom-up segmentation을 수행하고 이것들을 다양한 scale 상에서 병합을 하고 이 region을 사각형 box 영역으로 구분하는 것이다.
   2015년 Jan Hosang 등이 발표한 논문 “What makes for effective detection proposals?”에는 저자가 기존에 발표된 detection 과련 proposal에 대한 비교 분석한 자료가 있으며, 그 결과는 다음 표와 같다.
   이 결과를 보면 EdgeBox 방식이 가장 뛰어나다는 것을 확인할 수 있으며, selective search 방식도 전체적인 평가에서 좋은 편에 속하지만, 연산 시간 관점에서 보면 결과 좋은 방식이라고 볼 수는 없다.
   RPN(Region Proposal Network)
   Faster R-CNN의 블락도는 아래 그림과 같다.
   Fast R-CNN의 기본 구조와 비슷하지만, Region Proposal Network(RPN)이라고 불리는 특수한 망이 추가되었으며, 이 RPN을 이용하여 object가 있을만한 영역에 대한 proposal을 구하고 그 결과를 RoI pooling layer에 보낸다. RoI pooling 이후 과정은 Fast R-CNN과 동일하다.
   이 RPN은 Fast R-CNN에서 사용했던 동일한 ConvNet을 그대로 사용하기 때문에 입력의 크기에 제한이 없으며, 출력은 각 proposal에 대하여 “objectness score”가 붙은 사각형 object의 집합이 되며, model의 형태는 fully-convolutional network 형태다.
   이 RPN은 ConvNet 부분의 최종 feature-map을 입력으로 받아들인 후, n x n 크기의 sliding window convolution을 수행하여 256 차원 혹은 512차원의 벡터를 만들어내고 이 벡터를 다시 물체인지 물체가 아닌지를 나타내는 box classification (cls) layer와 후보 영역의 좌표를 만들어 내는 box regressor (reg) layer에 연결한다.
   k개의 object 후보에 대하여 cls layer 에서는 object인지 혹은 object가 아닌지를 나타내는 score를 구하기 때문에 2k score가 되고, reg layer에서는 각각의 object에 대하여 4개의 좌표(X, Y, W, H) 값을 출력하기 때문에 4k 좌표가 된다.
   전체적인 RPN에 대한 구성은 아래 그림과 같다.
   각각의 sliding window에서는 총 k개의 object 후보를 추천할 수 있으며, 이것들은 sliding window의 중심을 기준으로 scale과 aspect ratio를 달리하는 조합(논문에서는 anchor라고 부름)이 가능하다.
   논문에서는 scale 3가지와 aspect ratio 3가지를 지원하여, 총 9개의 조합이 가능하다.
   위 그림은 PASCAL VOC 2007에 있는 영상에 대하여 RPN을 이용하여 proposal을 낸 예제들이다.
   이렇게 sliding window 방식을 사용하게 되면, anchor와 anchor에 대하여 proposal을 계산하는 함수가 “translation-invariant” 하게 된다는 것이다. 이것은 CNN에서sliding window를 사용하여 convolution을 하였을 때 얻는 효과와 동일하다.
   이 translation-invariant한 성질로 인해 model의 수가 크게 줄어들게 된다. k= 9인 경우에 (4 + 2) x 9 차원으로 차원이 줄어들게 되어, 결과적으로 연산량을 크게 절감할 수 있게 된다.
   Faster R-CNN 결과
   Faster R-CNN에 대한 학습 결과는 아래 표와 같다.
   위 표에서 SS는 Selective Search 방식을 의미하고, EB는 EdgeBox 방식을 의미한다. 먼저 연산 시간을 줄이기 위해 ZFNet을 이용한 test를 진행하였으며, SS와 EB는 별도로 region proposal을 만든 후 detection을 진행한 것이고, “RPN + ZF, shared”는 RPN과 ZFNet을 사용한 Fast R-CNN 블락의 경우로 region proposal이 300개 정도였음에도 불구하고 정확도가 1% 이상 좋다는 것을 확인할 수 있다. 결과적으로 보면 region proposal을 별도로 수행하는 것보다 RPN을 이용하여 동시에 수행을 하더라도 결과가 더 나쁘거나 하지 않다는 뜻이 된다.
   다음은 연산 속도 관점에서의 결과를 살펴보면 아래 표와 같다.
   Selective search를 이용한 region proposal을 수행하고 Fast R-CNN을 하는 경우는 약 2초 정도가 소요가 되었지만, RPN을 이용해 region proposal을 수행하고 Fast R-CNN의 VGG-16 모델과 결합시키는 경우는 약 0.2 초가 걸린다. 결과적으로 초당 5fps 속도로 object detection이 가능하기 때문에 real-time 영역에 근접하게 되었다.
   또한 아래 표처럼, Fast R-CNN 블락으로 비교적 망이 단순한 ZFNet을 사용한다면, 약 17fps까지 속도가 올라가게 된다.
   이번 class에서는 지난 class에 이어 Faster R-CNN object localization/detection 방법에 대하여 살펴보았다. 이 Faster R-CNN은 ResNet에 적용되어 빛을 발하게 된다.
   다음 class에서는 실제로 ResNet의 object detection/localization에 Faster R-CNN이 어떻게 적용이 되었고, 결과는 어떻게 되었는지 알아볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [6]
   지난 [Part Ⅴ. Best CNN Architecture] 8. ResNet [4], Fast-RCNN ~ 8. ResNet [5], Faster-RCNN 을 통하여 Fast R-CNN 및 Faster R-CNN에 대하여 살펴보았다. 그 이유는 ResNet의 object detection/localization의 기본 알고리즘이 Faster R-CNN에 기반하고 있기 때문이다.
   Object detection은 크게 "feature extraction 부분"과 "object classifier(classifier + bounding box regressor)" 부분으로 나누어 생각할 수 있다. 그간 많은 연구자들이 object detection 성능을 끌어올리기 위해 HOG나 SIFT와 같은 gradient 기반의computer vision 알고리즘을 이용하여 feature 추출하고, SVM이나 DPM과 같은 classifier를 적용하여 object를 검출하였다.
   그리하여 R-CNN 방식이 발표되기까지 매년 조금씩 성능 개선이 있기는 했지만 지지부진했다. 2013년 R-CNN이 object detection에 CNN을 적용하면서, object detection 성능이 크게 개선이 되었으며, 이후 몇 년간 CNN 방식을 이용한 더 좋은 알고리즘이 연속적으로 발표가 된다.
   ResNet 개발팀은 object detection 분야에도 정통한 사람들이었으며, CNN/Deep CNN을 이용하여 object detection 성능을 끌어올리는 연구 분야에서 단연 두각을 나타내는 R-CNN, SPPNet, Fast R-CNN 및 Faster R-CNN과 직간접적으로 연관이 되어 있다. 그 결과 ResNet은 classification성능뿐만 아니라 detection/localization에서도 2015년 ILSVRC에서 우승을 차지한다.
   이번 class에서는 실제로 ResNet에서 object detection/localization을 어떻게 적용하였는지 살펴볼 예정이다.
   Faster R-CNN과 RPN(Region Proposal Network)
   Faster R-CNN은 기본적으로 Fast R-CNN과 동일하지만, 결정적인 차이는 object의 후보 영역을 추천하기 위한 region proposal 과정이 별도의 과정으로 되어 있는 것이 아니라, 아래 그림과 같이 Faster R-CNN 망에 후보 영역을 추천하기 위한 RPN(Region Proposal Network)이 포함이 되어 있다는 점이다. (자세한 내용은 [Part Ⅴ. Best CNN Architecture] 8. ResNet [5], Fast-RCNN  참고)
   .
   NoC(Network on Conv feature map)
   Faster R-CNN을 구현한 논문을 보면, ZFNet과 VGG-16을 이용하여 Faster R-CNN을 구현하였다. 마지막 ConvNet에서 얻어진 feature-map을 RPN(Region Proposal Network)이라고 불리는 특수한 망의 입력으로 보내고, 이 RPN을 이용하여 object가 있을만한 영역에 대한 proposal을 구한 후 RoI pooling layer에 보낸다.
   RestNet의 구조는 VGGNet과는 구조가 다른데 어떻게 Faster R-CNN을 구현할까?
   Faster R-CNN에 있는 ConvNet 부분은 ResNet으로 구현하는데 큰 무리가 없을 것 같은데, RPN과 RoI Pooling Layer 및 뒷단에 있는 classifier와 bounding box regressor 부분은 어떻게 구현을 해야 할까?
   ResNet 개발팀은 ResNet은 구조적으로 Faster R-CNN이나 Fast R-CNN의 블락도와 모양이 다른데, ResNet 구조를 크게 변화시키지 않고 Faster R-CNN을 구현하기 위한 고민을 하였다, 그 고민에 대한 해답은 그들이 발표한 논문 "Object Detection Networks on Convolutional Feature Maps"라는 논문에 답이 있다.
   이 논문에 따르면, object detection은 크게 보면 feature extraction 부분과 object classifier 부분으로 나눌 수 있다. CNN을 이용한 feature 추출 방법이 기존에 발표된 어떤 computer vision 알고리즘을 사용하는 것보다 훨씬 성능이 뛰어나다는 점은 이미 밝혀졌고, 이들은 object classifier의 성능에 영향을 주는 것이 무엇인지를 이 논문에서 밝히고 있다.
   위 그림은 이 논문에 나오는 블락도 이며, feature extractor 부분은 ConvNet으로 구현하기 때문에 ResNet으로 충분히 구현이 가능하다는 것을 알 수 있다. 이후 Region Proposal 및 RoI Pooling을 담당하는 부분도 구조적인 관점에서 보았을 때 ConvNet으로 구현이 가능하다.
   Object classifier 부분은 CNN을 통해서 얻어진 feature를 기반으로 별도의 망을 통해서 처리한다는 관점에서 "NoC(Network on Conv feature map)"이라고 명명하였으며, NoC 부분을 어떻게 구성을 하면 정확도가 개선이 되는지를 논문에서 밝히고 있다.
   Classifier 부분에 대한 전형적인 구현 방법은 MLP(Multi-Layer Perceptron)를 fc(fully connected layer) 2~3개로
   구현을 하는 방법이다. 하지만 이 논문에서는 기존처럼 단순하게 fc로 구현을 하는 것이 아니라 이부분을
   하나의 망으로 간주를 하고 어떨 때 최적의 성능이 나오는지를 실험으로 확인을 하였다.
   정확도를 개선하려면, 기존처럼 fully connected layer만을 사용하는 것보다는 fully connected layer 앞에 convolutional layer를 추가하면 결과가 좋아지며, multi-scale을 대응하기 위해 인접한 scale에서의 feature map
   결과를 선택하는 "Maxout" 방식을 지원하면 결과가 더 개선되는 것을 확인할 수 있다.
   아래 그림은 Maxout을 적용한 경우를 보여주는 그림이다.
   ResNet에서 Faster R-CNN을 구현하기
   앞서 설명한 것처럼, Faster R-CNN을 ResNet에서 구현하기 위해 그들은 101-layer 구조를 이용하였으며,
   NoC 개념처럼 RoI Pooling 및 그 이후 부분을 ConvNet으로 구현하도록 하였다.
   기억을 되살려, 다양한 layer 수에 따른 ResNet 구성의 경우를 살펴보면 아래 표와 같다.
   Faster R-CNN을 VGG-16에 구현할 때와 마찬가지로 ResNet을 다시 구성을 해서 살펴보면, 위 ResNet 구조를 "feature extractor + object classifier"로 구분할 수 있다.
   Feature extractor 부분은 conv1, conv2_x, conv3_x, conv4_x(101 layer ResNet에서 91 layer 부분에 해당)를 이용하여 구현할 수 있으며, 이렇게 되면, VGG16 때와 마찬가지로 feature map에서의 1 픽셀은 stride 16에 해당이
   된다. 또한 이 layer들은 RPN(Region Proposal Network)과 Fast R-CNN Network 공유하여 사용하게 된다.
   RPN 이후 과정은 Fast R-CNN 구조(아래 그림에서 회색 영역에 해당)를 구현을 하여야 한다. 이미 Con1 ~ Conv4는 feature를 추출하기 위한 용도로 사용을 했으니, 당연히 Conv5 부분과 Classifier 부분을 잘 활용하거나 변화를 시켜야 한다는 것은 감을 잡을 수 있다.
   ConvNet을 거쳐 얻어진 feature map을 conv5_1에 보내기 전에 RoI pooling을 수행하며, Conv5_x 부분은 위 그림에서 FCs에 해당하는 부분을 수행을 하며, 최종단의 Classifier는 위 그림의 최종단으로 대체를 해주면, ResNet을 약간만 변형을 하면, object detection에 활용이 가능하다.
   ResNet을 이용한 object detection 결과
   ResNet을 사용하여 위와 같은 방식으로 object detection을 수행한 결과, VGG-16을 사용하여 object detection을 수행한 것보다 정확도가 아래 표처럼 개선되는 것을 확인할 수 있다.
   모두 VGG-16을 사용할 때보다 ResNet을 사용하는 것이 결과가 좋아지는 것을 확인할 수 있다.
   이는 ResNet을 이용한 더 깊은 망의 효과를 톡톡히 누린 결과라고 볼 수 있다.
   이번 class에서는 ResNet을 이용하여 object detection을 구현하기 위해 ResNet을 어떻게 사용을 하였는지 살펴보았다. 기본적으로 CNN을 활용하여 feature extractor와 object classifier를 수행하기 위해 ResNet을 나누어 사용하였음을 확인하였다
   다음 class에서는 이후 ResNet 연구팀이 더욱 성능을 개선시킨 Upgraded ResNet을 발표하는데 이것에 대하여 살펴볼 예정이다.


[Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [7]
   CNN(Convolutional Neural Network) ? “ResNet (part7)”
   지난 [Part Ⅴ. Best CNN Architecture] 8. ResNet [1] ~ 8. ResNet [6] 을 통하여 ResNet의 기본 개념, ResNet의 특징과 장점, ResNet을 영상 classification/ localization/ detection 등 영상 인식 전반에 적용했을 때의 성능 및 Fast/Faster R-CNN까지 같이 살펴보았다.
   ResNet은 기존 CNN과 달리 short-cut connection이라는 독특한 개념이 적용이 되어 다소 이해에 어려움은 있을 수 있지만, 이전 Class를 통하여 어느 정도 극복을 했을 것이라 생각이 된다.
   ResNet이 2015년 처음 발표되었을 당시, ResNet 팀은 residual network이라는 독특한 개념을 적용한다는 관점에서는 혁신적이었고 원래 구조만으로도 충분한 성능을 보였지만, ‘ILSVRC-2015에 참가’라는 시간상의 제한으로 인해서 구조적인 고려가 충분하지 못한 점도 있었다. 이것은 그들의 원래 구조를 설명하는 논문 “Deep Residual Learning for Image Recognition”을 자세히 살펴보면 몇 곳에서 그런 부분을 찾을 수 있다.
   ResNet 팀은 2015년 대회를 마치고 자신들의 architecture에 대하여 심층적인 고민과 실험을 한 것으로 보이며, 그 결과 2016년에 “Identity Mappings in Deep Residual Networks” 라는 논문을 통해 “pre-activation” 이라는 개념을 적용한 개선된 Residual Network 구조를 발표하게 된다.
   이번 class에서는 2016년에 발표된 논문을 바탕으로 기존 ResNet의 문제점 및 이것을 개선하여 pre-activation이 적용된 ResNet에 대하여 살펴볼 예정이다.
   Residual Network에 대한 고찰
   기존 Residual Network의 기본 구조는 아래 그림과 같으며 입력과 출력의 관계는 그림에 있는 수식과 같다. 여기서 xl+1은 identity 함수 h(x)와 Residual 함수 F(x)의 합에 대하여 activation 함수인 ReLU를 한 결과이다. 그래서 아래 식처럼 “after-add mapping” 이라는 이름이 붙은 것이다.
   위 그림처럼 기존 ResNet에서는 함수 h(x)를 identity mapping을 위한 short-cut connection이라고 불렀지만, 실제로는 identity 함수와 residual 함수의 합을 더한 후 ReLU를 수행하기 때문에 진정한 의미에서의 identity라고 부르기는 힘들다. Residual Network 1개만을 고려했을 때는 큰 차이가 없지만, 이것들을 여러개를 연결시킨다고 생각을 해보면, addition 뒤에 오는 ReLU로 인해 blocking이 일어나기 때문에 xl+1이 xl과  F(x)의 합으로 표현되는 것이 아니라, 위 그림의 식처럼 함수 f 처리를 해줘야 한다. (합이 아니라 곱의 형태가 됨)
   만약에 f를 identity 함수라고 한다면 위 그림은 아래와 같이 바뀔 수 있다. 위 그림과 비슷해 보이지만 f가 identity 함수 이기 때문에 수식이 아래 그림처럼 바뀌게 된다.
   그럼 위 그림에서 볼 수 있는 수식의 변화는 무엇을 의미하는 것일까?
   위 수식의 진정한 의미는 임의의 위치에서의 xL을 x1와 residual 함수의 합으로 표시할 수 있다는 것이며, 이것은 forward나 backward path를 아주 간단하게 표현할 수 있게 된다. 즉, 특정 위치의 출력은 특정 위치에서의 입력과 residual 함수의 합으로 표현이 가능해진다는 점이다. 전형적인 ConvNet의 경우 forward나 backward path가 성격상 곱(multiplication)의 형태로 표현되는 것과 비교하면 연산이 엄청나게 단순해지는 효과를 얻을 수 있게 된다.
   그러므로 이렇게 original Residual Network를 개선해주면, 원래의 ResNet의 철학을 거의 유지하면서도 수학적인 관점에서 좀 더 유용하게 된다. ResNet 연구팀은 이점에 주목을 하여 original ResNet을 개선한다.
   아래 그림은 forward/backward path를 한눈에 보여주는 그림으로, 그림에 같이 표시된 수식을 보면 forward/backward path 연산이 단순 덧셈의 형태로 간단하게 구현이 가능해진다는 것을 알 수 있다.
   Identity skip connection의 중요도
   Identity skip connection이 무슨 의미를 갖는지 살펴보자.
   Short-cut connection이 어떤 경우에 최적의 결과를 도출할 수 있는지 확인을 하기 위해, ResNet팀은 short-cut connection을 다양한 경우로 바꿔가면서 실험을 수행하였다.
   위 그림의 (a)는 original ResNet에서 사용한 구조로 short-cut connection에 다른 어떤 것을 추가하지 않고 identity connection으로 연결시킨 경우이고, (b)부터 (f)까지는 short-cut connection을 다양하게 변화를 시켜가며 실험을 한 것이다. 110-layer의 ResNet에 대하여 CIFAR-10데이터를 이용하여 실험을 수행한 결과 원래의 identity shortcut의 결과가 가장 좋다는 것을 확인할 수 있었다. 다양한 경우에 따른 에러율은 위 그림에 표시가 되어 있다.
   (b)부터 (f)까지는 short-cut connection이 multiplication에서 blocking된 형태이기 때문에 앞서 확인한 것처럼 정보가 곧바로 전달이 되는 것이 아니라, 곱셈에 의해 변형된 형태로 전달이 되며, 이로 인해 최적화 문제가 생기는 것으로 추정된다. 결과적으로 identity skip connection이 속도나 성능 면에서 최적이라는 것을 확인할 수 있다.
   Activation 함수 위치에 따른 성능 체크
   ResNet 팀은 activation 함수의 위치가 어떤 영향을 주는지 확인을 하기 위해 다음 그림과 같은 다양한 조합에 대하여 실험을 하였다.
   이 그림에서 (a)는 ResNet의 원래 버전이며, addition 뒤에 ReLU가 오는 형태 (after-add)이다. (b)부터 (e)까지는 다양한 경우에 어떤 결과가 나오는지 확인하기 위한 조합이다.
   (b)는 기본 ResNet 구조에서 BN(Batch Normalization)의 위치를 addition 뒤로 옮긴 경우이며 그 결과는 아래 그림과 같다. 결과적으로 BN이 propagation path를 방해할 수 있으며, 아래 그림에서 볼 수 있는 것처럼, training과 test에서 모두 original 보다 결과가 좋지 못한 것을 확인할 수 있다.
   (c)는 addition 뒤쪽에 있는 ReLU를 residual net쪽으로 옮긴 경우이다. 원래는 residual 함수가 (-∞  , +∞  ) 범위로 나오는데 ReLU를 residual net 쪽으로 옮기면 residual net의 결과가 항상 non-negative 결과가 나오게 되며, 이로 인해서 망의 representation 능력이 떨어지고 결과적으로 성능이 나빠지게 된다.
   (d)와 (c)는 pre-activation 성능을 확인하기 위한 구조이다. Pre-activation 을 고려하는 이유는 원래의 residual net에 있는 ReLU가 진정한 identity mapping 개념을 방해하기 때문에 이것을 움직일 수 있는지 확인하기 위함이다.
   아래 그림을 살펴보자. 그림 (a)는 원래 ResNet 구조인데 여기서 act. layer의 위치를 조정하면 (b)와 같은 형태를 만들어 낼 수 있다. 원래 ResNet의 identity mapping 함수를 asymmetric 하게 조정을 하면, activation의 위치를 residual 쪽으로 옮길 수 있으며, 이것을 다시 정리하면 (c)와 같은 형태가 된다.
   Activation의 위치가 weight layer 앞쪽으로 오면서 pre-activation 구조를 갖게 된다. 이 pre-activation 구조에서 BN layer가 ReLU 앞으로 간 형태를 “full pre-activation”이라고 부르며, 앞에 오는 BN에 의해서 regularization 효과를 얻게 되어 결과가 좋아지게 되며, 아래 그림은 실험 결과이다.
   이 그림을 보면 재미 있는 결과를 확인할 수 있다. Pre-activation을 사용하면 Test  error가 original ResNet보다 좋아졌다는 것을 확인할 수 있지만, Training에서는 iteration의 횟수가 커지게 되면 original보다 약간 결과가 나쁜 것을 확인할 수 있다. 이것은 regularization을 통해 일반화가 더 잘되어서 이런 결과가 나온 것으로 추정이 된다. 즉 original은 학습 데이터에 더 특화된 결과를 보이지만 일반화(generalization) 특성은 개선된 ResNet보다 좋지 못하기 때문에 test 결과가 나빠지는 것으로 볼 수 있다.
   성능 비교
   순수하게 activation 위치에 따른 성능만 비교하기 위해, single-crop (320x320) ImageNet 데이터에 대한 실험 결과는 아래 표와 같다. 망이 깊어질수록 pre-activation 을 사용한 ResNet이 original ResNet보다 결과가 좋아진다는 것을 확인할 수 있다.
   아래 표는 다양한 망 구조에 따른 실험 결과이다 역시 pre-activation을 사용한 ResNet이 가장 좋은 결과를 내는 것을 확인할 수 있다.
   이번 class에서는 개선된ResNet의 구조에 대하여 확인을 해봤다. 실험 결과를 종합해보면, short-cut connection의 구조는 간단할수록 결과가 좋으며 identity mapping을 사용하는 것이 가장 좋았다. Original ResNet에서는 ReLU의 위치가 addition 뒤에 위치되며, 결과를 forward/backward propagation 시킬 때 약간의 blocking 구실을 했었는데, 수학적인 조정을 통해 activation의 위치를 Residual net쪽에 위치를 시키고 그것도 weight layer 앞쪽에 위치를 시키면, 수학적으로 완결된 형태가 되고 이로 인해서 결과가 더욱 개선이 된다. 또한 개선된 구조에서는 forward/backward propagation이 쉬워지기 때문에 1000-layer가 넘는 deep network를 어렵지 않게 구현할 수 있다.
   다음 class에서는 GoogLeNet 팀이 ResNet 구조를 반영하여 Inception-V4 구조를 발표하였는데, 이 구조에 대하여 살펴 볼 예정이다.


Machine Learning Academy_Part Ⅴ. Best CNN Architecture]

8. ResNet [8]
   GoogLeNet 설계팀들은 자신들이 설계한 Inception 구조에 대한 소신이 깊지만, 다른 팀들의 아이디어도 적극적으로 반영을 한다. 그리하여 2014년 Inception 구조(Inception-V1)를 발표한 이래 계속 개선된 Inception 구조를 꾸준히 발표한다.
   2014년 ILSVRC에서 영국의 옥스퍼드 대학교팀(VGGNet)이 GoogLeNet과 근소한 차이로 2위를 차지하지만, VGGNet은 그 구조의 간결함으로 인해 이후 많은 딥러닝 팀들이 VGGNet 구조를 사용한다.
   GoogLeNet 팀 역시 같은 해 발표된 VGGNet의 개념을 발전된 방식으로 반영하여 convolution kernel에 대한 factorization(인수분해) 개념을 집어 넣어 망을 더욱 깊게 만들었으며, 이것은 Inception-V2 및 V3에 반영이 된다. Inception-V2는 기본적으로 Inception-V1에 BN(batch normalization)을 반영한 기본 모델을 말하며, 여기에 구조 실험을 위해 convolution factorization, label smoothing 및 Auxiliary classifier에 BN까지 모든 개념을 결합시켜 가면서 성능을 꾸준히 발전을 시켰으며, 최종적으로 모든 개념이 적용된 모델을 Inception-V3라고 명명하였다. ([Part V. Best CNN Architecture] 5. GoogLeNet [4] 참고)
   비록 Inception-V3구조로 무장을 하고 2015년 ILSVRC 대회에 참가하지는 않았지만, 그 해 우승을 한 ResNet 성능에 근접하는 성능을 내게 된다. 논문에 따르면multi-model & multi-crop ensemble에서 top-5 에러율이 3.58% 수준에 도달하게 된다. (논문: Rethinking the Inception Architecture for Computer Vision). 논문 발표를 통해 자신들이 발전시킨 구조 역시 성능이 뛰어남을 입증하는 ‘뒤끝 정신’을 보인다.
   한편, 2015년에 마이크로소프트팀이 발표한 ResNet 구조는 여러 면에서 혁신적인 구조라서 GoogLeNet 팀도 많은 검토를 한 것 같다. 그들이 2016년에 발표한 논문 “Inception-V4, Inception-ResNet and the Impact of Residual Connections on Learning”에 따르면, 일부는 ResNet 팀의 아이디어에 긍정을 하지만, 일부는 꼭 그렇지 않다고 언급하고 있다. 자신들이 개선시킨 Inception-V4 구조를 통해서도 ResNet 구조를 사용하지 않더라도 ultra-scale 딥러닝이 가능함을 밝힌다. 또한 자신들의 Inception 구조에 ResNet 개념을 접목시킨 Inception-ResNet 도 같이 발표를 하고, Inception 구조에 ResNet을 개념을 접목시킨 새로운 Inception 구조가 ResNet 팀이 원래 발표한 성능보다 좋다는 것도 보여준다.
   이번 class에서는 2016년에 발표된 논문을 바탕으로 Inception-V4 및 Inception-ResNet에 대하여 살펴볼 예정이다.

Inception-V4

Inception ?V4의 기본 구조는 stem이라고 불리는 9개의 layer를 갖는 부분을 통해 크기가 299x299x3 영상을 받아 35x35x384 크기의 feature-map을 만들어 낸다. 이후 여러 단의 Inception-A, Inception-B 및 Inception-C가 오며, 인셉션 모듈 중간에는 feature-map의 크기를 줄이기 위한 Reduction-A와 Reduction-B가 오고 최종단에는 average pooling, dropout 및 softmax layer가 온다. 전체적인 구조는 아래 그림과 같다.
기본 구조는 이전 Inception구조와 크게 다르지 않다는 것을 확인할 수 있지만, stem 부분은 이전 인셉션 구조가 비교적 단순한 convolution과 pooling의 조합이었던 반면에 Inception-V4는 좀더 복잡한 구조가 되었다는 것을 확인할 수 있다. 특히 feature-map의 크기를 줄이는 부분이 이전 인셉션 구조와 달리 형태가 다양한 형태를 취하고 있다.
Inception-V3와 마찬가지로 35x35 크기의 feature-map으로부터 8x8 크기의 feature-map을 만드는 과정에서는 feature-map에 크기에 따라 3가지 다른 형태의 Inception 모듈을 사용했으며, 다음 그림과 같다.
Inception-V4에는 이전과 달리 이미지 크기를 줄이기 위한 reduction module이 2개 추가가 되었으며, 이것들의 구조는 아래 그림과 같다.
Inception-ResNet
Inception-ResNet은 V1과 V2가 있다. 기본적인 구조는 모두 동일하고, stem 부분에 적용된 layer 구조가 차이가 있다. V1의 stem은 비교적 단순한 구조를 취하고 있지만, V2의 stem은 Inception-V4와 동일한 구조를 취하는 점이 다르다. 또한 ResNet-V2가 ResNet-V1 보다 필터의 개수가 더 많다.
원래의 ResNet과 달리 GoogLeNet 팀은 인셉션 구조의 특성을 고려하여 ResNet 구조를 아래 그림처럼 조금 변형을 시켰다.
위 그림처럼 변형을 시킨 이유는 인셉션은 맨 앞단에 오는 1x1 convolution을 통해 차원을 감소시켜 연산량을 줄이는 구조를 취하지만, ResNet 구조에 맞게 feature-map의 차원을 맞춰주면서 연산량을 적정선으로 유지하기 위해 맨 앞단에 오히려 차원을 맞춰주기 위한 1x1 convolution이 온다.
Inception-ResNet의 메인 구조는 아래 그림과 같다. 앞서 이야기한 것처럼, stem 부분은 Inception-V4와 동일하다. 전체적으로 Inception-V4나 Inception-Resnet-V2의 메인 부분은 75 layer로 layer 깊이는 동일하다.
Inception-ResNet-A, B, C 부분의 구조는 아래 그림과 같다. 원래의 ResNet은 맨 앞에 오는 1x1 convolution 부분이 차원을 줄여주는 역할을 했지만, Inception-ResNet은 차원을 유지시켜주기 위해 1x1 convolution을 통해 만들어내는 feature-map의 개수가 큰 것을 확인할 수 있다.
Reduction-A와 Reduction-B의 구조는 다음 그림과 같다.
성능
ILSVRC-2012 데이터를 이용하여, single-crop single-model로 실험한 결과 성능은 아래 표와 같다. BN-Inception은 Inception-V2에 해당한다고 볼 수 있으며, 인셉션 구조가 발전함에 따라 에러율이 개선되는 것을 확인할 수 있다. Inception-V4와 Inception-ResNet-V2는 거의 비슷한 성능을 보이고 있다.
또한 144-crop single-model에 대한 실험도 하였는데, 그 결과는 아래 표와 같다. 이 표에서 ResNet-151 결과는 ResNet 팀이 논문에 발표한 성능이다. 이것을 보면 Inception-V3 이상의 구조에서 모두 ResNet-151 보다 성능이 좋다는 것을 알 수 있다.
또한 더욱 성능을 끌어올리기 위해 여러 모델에 대한 ensemble을 하여 실험한 결과는 아래 표와 같다. 최고의 성능은 Inception-V4와 3개의 Inception-ResNet-V2 모델을 조합했을 때이며, 이 때 top-5 에러율은 3.08%까지 떨어져 현재까지 발표된 모든 딥러닝 논문 중 최고의 성능을 보였다.
ResNet 구조를 채용했을 때의 가장 큰 장점 중 하나는 학습속도가 빠르다는 점이며 실험결과는 아래의 그래프를 통해 확인이 가능하다. 훨씬 적은 epoch에서 학습 결과가 나오는 것을 확인할 수 있으며, 이는 초기에 최적의 hyper-parameter를 결정할 때 유용하게 활용할 수 있다.
기존 인셉션 구조를 개선한 Inception-V4의 성능과 인셉션 구조에 residual net 개념을 도입한 Inception-ResNet-V2는 모두 뛰어난 성능을 보이는 것이 확인이 되었다. GoogLeNet 팀들이 발표한 인셉션 구조도 뛰어나고, ResNet 팀이 발표한 ResNet 역시 뛰어난 구조임이 확인되었지만, 아직도 sub-network의 구조를 어떻게 가져가는 것이 정말 최고인지는 명쾌하게 설명할 수 없는 것 같다. 어쩌면 ILSVRC-2012라는 데이터 집합에만 특화된 결과를 내고 있는 것인지도 모른다.
그런 관점에서 보면 2016년 하반기와 2017년에는 또 누가 얼마나 더 뛰어난 구조를 발표할 것인지 기대가 된다.

- 이번 Class를 끝으로 처음에 기획하였던, 영상 관련 딥러닝 구조에 대한 설명을 마친다. LeNet5로부터 시작된 Best CNN Architecture 설명이 좀 부족한 부분이 있을 수도 있지만, 큰 흐름을 살펴봤다는 관점에서는 의미를 갖는 것 같다. 다음 Class에서는 그 동안 구조 설명을 하면서 충분히 자세하게 다루지 못한 내용에 대하여 하나씩 다뤄갈 예정이다.
