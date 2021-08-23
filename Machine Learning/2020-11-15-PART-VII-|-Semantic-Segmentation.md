---
title: "[기계학습] 7. Semantic Segmentation"
excerpt: "Process of classifying each pixel belongings to particular level"
slug: "7-ml-semseg"
layout: post
category: "ml"
lang: en
tags: ["ml", "coursework"]
date: 2020-11-15T20:28:09-05:00
use_math: true
draft: false
---
PART VII / Semantic Segmentation
ⓒ [라온피플](https://blog.naver.com/laonple), Stanford cs231n

**fly to the moon.**
<!--more-->


PART VII / Semantic Segmentation​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [1]
이전 Class에서 살펴본 것과 같이, Image classification 성능은 2012년 AlexNet이 발표되면서 CNN 기술을 이용하여 기존 컴퓨터 비전 기술 대비 압도적인 성능차를 보여줬기 때문에 classification에 CNN을 적용하는 것은 너무 당연하게 인식 되었다. 또한 매년 거의 두 배씩 성능을 향상을 보이면서, classification 분야에서는 이미 사람의 수준을 넘어섰다.
영상의 중요한 응용분야 중 하나가 영상에서 의미 있는 부분으로 구별을 해내는 segmentation 분야이다. threshold나 edge에 기반한 단순 구별이 아니라, 영상에서 의미 있는 부분으로 구별해내는 기술을 semantic segmentation 이라고 하며, detection 분야가 CNN을 이용하여 성능이 많이 개선되었듯이 semantic segmentation 분야 역시 CNN 기술을 활용하여 최근 몇 년 동안 상당한 성과를 거두었다.
Classification은 전체 영상에서 특정 객체의 존재 여부만 알아내면 되지만, detection은 위치까지 파악을 해야 하고 해당 개체의 주위로 bounding box로 구별을 해줘야 하기 때문에 classification에 비해서 더 어려운 것은 당연하고, CNN으로 구현을 할 때도 이점을 고려해줘야 하기 때문에 구조가 좀 더 복잡하다.
반면에 segmentation은, 특히 semantic segmentation은 bounding box로 대강의 영역을 표시하는 것이 아니라, 자연 영상에서 정확하게 개체의 경계선까지 추출하여 영상을 의미 있는 영역으로 나눠주는 작업을 해줘야 하기 때문에 더 어렵다고 볼 수 있다.
우리는 이미 이전 class를 통해 classification과 detection에 대하여 최신 구조까지 충분히 살펴보았다. 앞으로는 semantic segmentation에 대하여 살펴볼 예정이나, 선행 지식을 많이 필요로 하기 때문에, 이번 class 부터는 segmentation에 관한 기초 이론부터 차근차근 살펴볼 예정이다.
Gestalt 시지각(示知覺) 이론
Gestalt는 ‘형태, 모양’을 뜻하는 독일어로, Gestalt 시지각 이론에서는 일반적으로 시각을 통해 대뇌로 전달되는 외계의 형태에 대한 정보들은 기억하기 쉬운 상태로 혹은 특징 지워질 수 있는 상태로 정리되는지 설명하고 있으며, 이 이론에 따르면 영상 속에 있는 개체를 집단화할 수 있는 5개 법칙이 있다. 이것을 집단화의 법칙(Law of Grouping)이라고도 하고, 1923년 베르트하이머가 체계화 시켰다.
유사성(Similarity): 모양이나 크기, 색상 등 유사한 시각 요소들끼리 그룹을 지어 하나의 패턴으로 보려는 경향이 있으며, 다른 요인이 동일하다면 유사성에 따라 형태는 집단화.
근접성(Proximity): 시공간적으로 서로 가까이 있는 것들을 함께 집단화해서 보는 경향.
공통성(Commonality): 같은 방향으로 움직이거나 같은 영역에 있는 것들을 하나의 단위로 인식하며, 배열이나 성질이 같은 것들끼리 집단화 되어 보이는 경향.
연속성(Continuity): 요소들이 부드럽게 연결될 수 있도록 직선 혹은 곡선으로 서로 묶여 지각되는 경향.
통폐합(Closure): 기존 지식을 바탕으로 완성되지 않은 형태를 완성시켜 지각하는 경향.
Gestalt 이론은 이런 인지심리학적인 면에 대한 고찰의 결과이며, 이런 특성들을 활용하여 광고나 예술 등에도 활용이 되며, 영상에 대한 segmentation 연구나 이론 개발 시에도 고려가 되어야 한다.
Image Segmentation
Segmentation의 목표는 영상을 의미적인 면이나 인지적인 관점에서 서로 비슷한 영역으로 영상을 분할하는 것을 것이다. 예를 들어 아래 그림에서 기차 부분만을 완벽하게 분리해내는 것이 최고의 segmentation이 되겠지만, 실제로는 그렇게 쉽지만은 않다
Segmentation은 영상을 분석하고 이해하는데 필요한 기본적인 단계이며, 영상을 구성하고 있는 주요 성분으로 분해를 하고, 관심인 객체의 위치와 외곽선을 추출해 낸다.
Segmentation의 접근 방법에 따라 크게 3가지 방식으로 분류가 가능하다.
픽셀 기반 방법: 이 방법은 흔히 thresholding에 기반한 방식으로 histogram을 이용해 픽셀들의 분포를 확인한 후 적절한 threshold를 설정하고, 픽셀 단위 연산을 통해 픽셀 별로 나누는 방식이며, 이진화에 많이 사용이 된다. thresholding으로는 전역(global) 혹은 지역(local)로 적용하는 영역에 따른 구분도 가능하고, 적응적(adaptive) 혹은 고정(fixed) 방식으로 경계값을 설정하는 방식에 따른 구별도 가능하다.
Edge 기반 방법: Edge를 추출하는 필터 등을 사용하여 영상으로부터 경계를 추출하고, 흔히 non-maximum suppression과 같은 방식을 사용하여 의미 있는 edge와 없는 edge를 구별하는 방식을 사용한다.
영역 기반 방법: Thresholding이나 Edge에 기반한 방식으로는 의미 있는 영역으로 구별하는 것이 쉽지 않으며, 특히 잡음이 있는 환경에서 결과가 좋지 못하다. 하지만 영역 기반의 방법은 기본적으로 영역의 동질성(homogeneity)에 기반하고 있기 때문에 다른 방법보다 의미 있는 영역으로 나누는데 적합하지만 동질성을 규정하는 rule을 어떻게 정할 것인가가 관건이 된다. 흔히 seed라고 부르는 몇 개의 픽셀에서 시작하여 영역을 넓혀가는 region growing 방식이 여기에 해당이 된다. 이외에도 region merging, region splitting, split and merge, watershed 방식 등도 있다.
Image Segmentation의 과정
Segmentation 과정은 bottom-up 방식과 top-down 방식이 있다. Bottom-up 방식은 비슷한 특징을 갖는 것들끼리 집단화 하는 것을 말하며, top-down 방식은 같은 객체에 해당하는 것들끼리 집단화 하는 것을 말하며, 아래 그림과 같다.
이번 Class에서는 segmentation의 기초에 대하여 알아보았다. 궁극적으로 머신 러닝에 기반한 방법까지 살펴볼 예정이지만 기본 이론이 필요하기 때문이며, 다음 Class부터 segmentation 방법을 하나씩 구체적으로 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [2]
Image Segmentation ? “기본 이론 (part2)”
이전 Class에서는 영상 segmentation의 방법에 대하여 살펴보았으며, 크게 보면 픽셀 기반, Edge 기반 및 영역 기반의 방법이 있음을 알게 되었다. 이번 Class에서는 그 중 가장 직관적으로 알 수 있는 픽셀 기반의 방법부터 먼저 살펴볼 예정이다.
픽셀 기반의 segmentation에서 흔히 사용되는 방식이 바로 thresholding이다. 말 그대로 특정 임계값을 정하고 그보다 작으면 검은색으로 그보다 같거나 크면 흰색으로 표시하는 단순한 방법으로 아래 그림과 같다.
위 그림은 임계값 T를 1개만 적용한 경우이며, 새로운 픽셀값은 아래 수식에 의해서 결정이 된다.
이 방식은 간단하기 때문에 누구나 쉽게 이해를 할 수가 있다. 경우에 따라서는 2개 혹은 그 이상의 threshold가 필요한 경우도 있으며, 아래의 예는 2개의 threshold를 적용한 경우로 임계값 T1과 T2를 두고 그 사이에 있는 경우는 흰색으로 밖에 있는 경우는 검은색으로 표시를 하였다. 생각보다 근사하게 경계를 추출할 수 있음을 확인할 수 있다.
이 그림에 대한 수식은 아래와 같다.
Threshold를 설정하는 방법?
아래 그림을 보면 임계값을 너무 낮게 혹은 너무 높게 설정하여 thresholding을 한 경우를 보여준다.
임계값이 제대로 설정되지 못하면 엉뚱한 결과가 나오므로, 임계값을 적절하게 설정하기 위한 방법이 필요하다. 이 때 가장 많이 사용되는 방식이 픽셀값의 누적 분포를 알 수 있는 히스토그램을 사용하는 것이다.
아래 그림을 보면 픽셀값 분포를 보여주는 히스토그램이 나오는데 다행스럽게도 peak가 2개이고, 그 간격도 상당히 떨어져 있는 것을 확인할 수 있다. 이럴 경우는 대략적으로 2개의 peak 가운데 값으로 잘라주는 방식으로 쉽게 thresholding이 가능하다.
Otsu Algorithm
위 그림은 peak 사이의 간격도 멀고, peak가 2개이며 peak 근처로 확실하게 데이터가 몰려 있기 때문에 쉽게 임계값 T를 정할 수가 있었다. 하지만, 그렇지 못한 경우도 많다. 이 때 최적의 T를 결정할 수 있는 방법은 무엇일까?
그것은 Otsu 알고리즘을 적용하면 된다. 임계값 설정에 대한 여러 알고리즘이 알려져 있지만, 그 중에서 통계적 방법을 이용한 Otsu 알고리즘이 가장 자연스러운 임계값을 설정할 수 있게 해준다. 이 알고리즘은 1979년에 발표가 되었으며, 논문 제목은 “A threshold selection method from gray-level histograms”이니 참고를 하면 좋을 것 같다. 알고리즘이 그리 복잡하지 않으면서도 결과가 상당히 좋으며, 무엇보다도 직관이 아니라 자동화가 가능하다는 점이 매우 유용하다.
예를 들어 위 그림의 왼쪽과 같은 쌀을 찍은 영상에서 쌀의 경계만을 추출하여, 쌀의 품질을 검사하는 경우를 생각해보자. 쌀의 경계를 잘 추출하기 위해 히스토그램을 구했더니 오른쪽 그림과 같은 분포를 보였다. Peak의 경계가 명확하지 않기 때문에 어느 값을 기준으로 삼아야 할 것인지 애매하다.
이런 경우에 Otsu 알고리즘을 사용하면 알맞은 T값을 구할 수 있다. 영상에서 임의의 임계값 T를 기준으로 하여 T보다 작은 픽셀(즉, 어두운 픽셀)의 비율을 q1라고 하고, 같거나 큰 픽셀의 비율을 q2라고 하자. 또한T를 기준으로 만들어진 2개의 그룹에 대하여 평균과 표준편차를 구하고 각각 μ1, μ2 및 σ1, σ2라고 하자.
Otsu 알고리즘의 핵심 아이디어는 간단하다. 2개의 그룹에 대하여 ‘그룹 내 분산(within class variance)’을 최소화하거나 혹은 ‘그룹 간 분산(inter class variance)을 최대로 하는 방향으로 T를 정하면, 가장 적절한 임계값을 구할 수 있다.
예를 들어 6개의 T 값에 대하여 다음과 같은 결과를 얻었다고 하면, T = 3 인 경우가 Otsu 알고리즘 상으로는 최적의 임계값이 된다.
‘그룹 내 분산’을 최소화하는 것과 ‘그룹 간 분산’을 최대화 하는 것은 같은 의미를 갖는다. 하지만 식은 ‘그룹 간 분산’을 구하는 식이 좀 더 쉽기 때문에 일반적으로는 ‘그룹 간 분산’ 방식을 더 많이 사용한다.
Global or Local
그림 전체에 동일한 임계값을 적용하는 경우에 조명이나 특정 잡음으로 인해 문제가 되는 경우가 있다. 아래 그림은 lens shading의 문제로 인해 중심은 밝고 주변은 어두운 이미지(왼쪽)에 대하여, 그림 전체에 대하여 동일한 임계값을 적용하는 경우는 중간에 있는 그림과 같이 문제가 있지만, 전체 영역을 여러 개로 나누고 각각의 영역에 대하여 적절한 thresholding을 하는 경우는 왼쪽의 그림과 같이 좋은 결과를 얻을 수 있다.
이처럼 그림 영역을 여러 개의 영역으로 나누고 각각에 대하여 다른 임계값을 적용하는 것을 local thresholding이라고 부른다.
또 다른 예로 아래와 같이 banding noise가 있는 경우에도 local thresholding을 적용하면 유용한 결과를 얻을 수 있다.
이번 Class에서는 픽셀 기반의 segmentation 알고리즘에 주로 사용되는 thresholding 방법에 대하여 살펴보았다. Thresholding은 직관적이기 때문에 쉽게 이해가 가능한 편이며, 실제로 비젼 알고리즘의 응용분야에서 영상의 경계를 추출하거나 이진화(Binarization)를 위해 흔히 사용이 된다. 다음 Class에서는 Edge 기반의 segmentation 방법에 대하여 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [3]
Image Segmentation ? “기본 이론 (part3)”
이전 Class에서는 thresholding방법을 사용하는 픽셀 기반의 segmentation의 방법에 대하여 살펴보았다. 픽셀 기반의 segmentation은 직관적이라서 이해에 큰 무리가 없을 것이라 생각된다. 이번 Class에서는 엣지(edge)기반의 segmentation 방법에 대하여 살펴볼 예정이다.
엣지 기반의 segmentation의 장점은 복잡하지 않다는 점과 영역을 분리할 때 엣지가 중요한 특징이 된다는 점이다. 말 그대로 그리 복잡하지 않으면서도 효과는 매우 좋기 때문에 영상의 segmentation에서 많이 사용이 된다. 하지만 엣지 기반의 방식은 영상에 잡음이 있거나 부분적으로 가려지는 현상이 발생하면 성능이 떨어지는 문제가 있다.
엣지 기반의 방법은 크게 2개의 과정으로 이루어진다.
엣지 검출 ? 엣지에 있는 픽셀(edgel이라고도 부름)을 찾아내는 과정
Gradient, Laplacian, LoG, Canny filter 등등
엣지 연결 ? 엣지에 있는 픽셀들을 연결하는 과정
지역적인 방식(local processing) - Gradient의 방향과 크기가 미리 정한 기준을 만족하는 경우 인접한 픽셀을 연결
전역적인 방식(global processing) - 허프 변환(Hough transform)
Gradient
엣지란 영상에 존재한 불연속적인 부분을 말하여, 대상으로는 밝기, 컬러, 표면의 무늬(texture) 같은 것들이 있다. 엣지 검출 방법은 매우 많으며, 흔히 공간 필터(spatial filter)를 많이 사용한다.
공간 필터들의 역할은 영상으로부터 gradient를 추출하며, 아래 그림과 같이gradient는 가장 변화의 속도가 빠른 방향을 나타내게 된다.
또한 gradient는 변화의 세기를 나타내는 “크기”와 가장 변화의 속도가 빠른 쪽을 나타내는 “방향”을 가지며 아래 식과 같다.
에지 검출 필터 (1차 미분/2차미분)
엣지 검출이란 불연속한 부분을 찾아내는 것이기 때문에 흔히 1차 미분과 2차 미분을 이용한다. 1차 미분을 사용하는 대표적인 필터로는 Prewitt, Robert, Sobel 등이 있으며, 많이 사용하는 Sobel 필터만 살펴보면 아래의 수식과 같으며, x와 y 방향의 엣지를 각각 검출한 후 합치는 과정을 거친다.
2차 미분을 사용하는 대표적인 공간필터는 Laplacian 필터가 있으며 수식은 아래와 같다.
에지 검출과 잡음의 영향
잡음이 많은 경우에는 아래 그림처럼, 미분을 하더라도 엣지 검출이 어려울 수도 있다.
그러므로 효과적인 엣지 검출을 하려면, 사전에 smoothing 필터 등을 사용하여 잡음의 영향을 줄이는 것이 필요하다. 아래 그림은 동일한 데이터에 대하여 가우시안 필터를 적용한 후에 엣지 검출이 되는 것을 보여준다.
Laplacian of Gaussian (LoG), Difference of Gaussian (DoG)
또한 앞서 고려한 것처럼, 사전에 가우시안 필터를 적용하여 잡음을 제거하고 여기에 2차 미분 필터인 라플라시안 필터를 적용하면, 아래 그림처럼 큰 엣지를 비교적 쉽게 찾아낼 수 있다.
가우시안과 LoG 필터의 커널은 아래 그림과 같다.
LoG는 잡음이 있는 환경에서도 엣지를 잘 찾을 수 있기 때문에 여러 곳에서 많이 쓰이고 있다. 가우시안 필터는 σ 값에 따라 smoothing을 적용하는 범위가 달라지게 되며, SIFT(Scale Invariant Feature Transform)에서는 연산량이 많은 LoG 대신에 DoG(Difference of Gaussian)을 사용하기도 한다.
아래 그림의 왼쪽은 원 영상이며, 오른쪽은 5x5 가우시안 필터를 적용한 영상이다. 5x5 가우시안 필터를 적용하면 영상의 detail이 많이 무너지는데, 같은 공간에 원영상과 가우시안 핕터가 적용된 영상을 같이 나타내기 위해 영상의 크기를 줄였더니, 가우시안 필터 효과가 작아보인다는 점은 감안이 필요하다. 어쨌든 이 두개의 그림의 차(difference)를 이용하면, 즉 DoG 된, 영상을 구하면 엣지가 그런대로 잘 검출되는 것을 확인할 수 있다.
위 2개의 영상의 차를 구하게 되면, 아래 그림과 같으며 어느 정도 엣지 검출 능력이 있음을 알 수 있다. 아래 영상은 그 4를 곱하고 128을 offset으로 더해서 얻어진 영상이다. (검출되는 엣지를 확인할 수 있도록 좀 더 큰 그림으로 나타낸다)
검출된 엣지에 후처리 (post-processing)
공간필터를 통해 검출된 엣지는 곧바로 사용하기에는 부적절하기 때문에 후처리(post-processing)을 필요로 한다. 왜냐하면, 아래 그림에서 볼 수 있는 것처럼, 개체의 경계 부분만 검출이 되면 좋겠지만 경계가 아닌 부분도 같이 검출이 되고 때로는 경계에 해당하는 부분에서 검출된 값이 작아서 후처리 과정에서 사라지는 경우도 존재한다. 앞에서 살펴본 LoG의 장점에도 불구하고, LoG를 사용하게 되면 경계 부분이 smoothing 되면서 사라질 수 있어 정확한 외곽선 검출이 어려울 수도 있다.
위 그림은 그래도 상태가 좋지만, 아래와 같은 경우를 고려해보자. 원영상에 작은 detail이 너무 많기 때문에 gradient 를 구하더라도 상당히 곤혹스럽다. 왼쪽은 원 영상이고 오른쪽은 공간필터를 이용하여 구한 ‘gradient 크기’ 영상이다.
이 ‘gradient 크기’ 영상으로부터 좀 더 의미 있는 엣지를 추출하려면 추출된 엣지들에 대하여 thresholding을 적용해야 하며, 아래 그림은 서로 다른 thresholding을 적용한 경우를 보여준다. Thresholding 값을 달리하였을 때 좀 왼쪽이 좀 나은 것처럼 보이지만 여전히 여기서 뭔가를 끄집어내기는 쉽지 않다.
후처리 과정으로 대표적인 것들은 다음과 같다.
Thresholding: 일정 크기 이하의 엣지들을 제거하여 큰 엣지 위주로 정리
Thinning: non-maximum suppression을 적용하여, 굵은 엣지를 얇게 처리
Hysteresis thresholding: 1 개의 threshold를 이용하는 대신에, high와 low 2개의 threshold를 사용하며, high threshold 엣지에 연결된 low threshold 엣지는 제거하지 않고 살려두는 것을 말한다. 아래 그림에서 왼쪽 위에 있는 그림은 작은 threshold를 적용했기 때문에 많은 작은 엣지가 살아 있으며, 오른쪽은 높은 threshold를 적용했기 때문에 엣지가 많이 사라졌으며, 밑에 있는 그림은 hysteresis thresholding을 적용하였을 때를 보여주며, low와 high threshold의 두가지 threshold의 장점을 취할 수 있게 되었다.
앞서 살펴본 대부분의 후처리 과정 및 장점들은 “Canny Edge Detector”에 거의 대부분 적용이 된다. 이 부분은 다음 Class에서 상세하게 살펴 볼 예정이다.
이번 Class에서는 엣지 기반의 segmentation 방법에 대하여 주로 살펴 보았다. 엣지를 추출하기 위해 공간 필터를 사용하지만 원하는 부분뿐만 아니라 원하지 않는 부분들도 같이 검출이 되기 때문에 원하는 부분만을 떼어내려면 여러 후처리 과정이 필요하다는 것을 알게 되었다.
다음 Class에서는 Canny Edge Detector 및 검출된 에지를 연결하는 방법에 대하여 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [4]
Image Segmentation ? “기본 이론 (part4)”
이전 Class에서는 엣지(edge) 기반의 segmentation 방법의 일반적인 부분 및 공간 필터를 사용하는 방식에 대하여 알아보았다. 이번 Class에서는 이전 Class에 이어 엣지 기반의 segmentation 방법에 대하여 살펴볼 예정이다.
Edge thinning
Edge thinning 이란 말 그대로 검출한 엣지를 1 픽셀 크기로 얇게 만드는 것을 말한다. 보통 엣지 검출을 하기 전에 잡음을 제거하기 위해 미디안(median)이나 가우시안 필터와 같은 smoothing 필터를 적용한다. 그 후 엣지 검출을 위해 공간 필터를 적용하면, 보통은 선폭이 1 픽셀 이상이기 때문에 이것을 적절한 방법으로 얇게 만들어줘야 하는데, 이것을 thinning 이라고 부른다.
Thinning을 하는 이유는 얇고 선명한(sharp) 엣지가 객체 등의 응용에서 훨씬 유용하기 때문이며, 허프 변환(Hough transform)을 이용하여 직선이나 타원 등을 검출할 때도 얇은 경우에 결과가 더 좋게 나오기 때문이며, 때로 엣지가 객체의 경계에 위치하고 있는 경우는 객체의 perimeter를 구할 때 복잡한 공식을 사용하지 않고도 쉽게 할 수 있기 때문이다.
Thinning은 아래 그림처럼 보통 4방향 혹은 8 방향 방법을 사용한다.
4방향 또는 8방향 방법을 정했다면, 중심 픽셀의 엣지값과 주변 픽셀의 엣지값을 비교하여 중심픽셀이 가장 큰 값인 경우는 남겨두고, 그렇지 않은 경우는 제거하는 방식이다. 그래서 이것을 non-maximum suppression이라고도 부른다.
예를 들면, 왼쪽과 같은 영상에 대하여 thinning을 하면 오른쪽과 같은 결과가 나온다. 먼저 맨 왼쪽/위 방향부터 3x3 sliding window를 적용시키면서 3x3 윈도의 중심값보다 큰 주변값이 있으면 중심값을 계속 0으로 바꿔가면, 최종적으로는 오른쪽과 같은 영상을 얻을 수 있다.
Canny Edge Detector
Canny 엣지 검출기는 1986년에 “A Computational Approach to Edge Detection” 라는 제목으로 J. Canny가 발표를 하였으며, 다른 엣지 검출기에 비해 성능이 뛰어나기 때문에 가장 많이 사용되는 엣지 검출 알고리즘 중 하나이다.
최적의 엣지(optimal edge)를 검출하려면 다음과 같은 3가지 항목의 특성이 좋아야 한다.
잡음 제거 또는 평활화(Noise smoothing)
엣지 개선(Edge enhancement)
엣지 위치 파악(Edge localization)
3가지 항목의 특성을 살펴보기 위해 아래 그림과 같은 이상적인 edge에 백색 가우시안 잡음이 추가된 경우를 고려해보자.
잡음이 있더라도 우수한 엣지 검출력을 보이려면, 실제로 엣지가 위치하고 있는 x = 0에서 잡음보다 큰 응답 특성이 나와야 하고, 실제로 가장 큰 엣지 값이 나와야 한다. (Noise smoothing, Edge localization)
또한 x = 0 일정 근처에서 단 1개의 최고값만을 갖도록 해줘야 한다. 아래 그림에서 볼 수 있는 것처럼, 다음 peak와는 어느 정도 거리 확보가 필요하다. (Edge enhancement)
Canny 엣지 검출기는 위 3가지 특성을 극대화하기 위한 목적으로 개발된 개발된 엣지 검출기라고 볼 수 있다. Canny 엣지 검출기는 다음과 같은 3개의 과정으로 이루어져 있다.
Gaussian smoothing한 후에 Gradient의 크기와 방향을 구한다.
Non-maximum suppression을 통해 검출된 엣지를 1 픽셀의 크기로 얇게 만든다.
Edge Linking 및 Hysteresis thresholding을 통해 의미 있는 엣지만을 남긴다
Canny 엣지 검출의 상세 과정 ? Smoothing, Gradient 구하기
첫 번째 과정은 잡음의 영향을 최소화하기 위해 가우시안 필터를 적용하여 영상을 부드럽게 만들고, gradient를 구하기 위해 x 및 y 방향으로 미분을 하는 것이다.
원래는 가우시안 필터링을 먼저 하고 미분을 하는 것이 맞지만, 가우시안 함수의 convolution 특성 상 가우시안 함수의 미분을 먼저 한 뒤에 convolution을 해도 상관없다. 이 수식은 아래와 같다. 여기서 I는 엣지를 구할 원 영상이다.
가우시안 함수를 미분한 수식은 아래와 같으며, 이 식에서 σ는 연산을 적용할 커널의 크기를 결정한다.
이 식에서 scale(σ)는 smoothing이 되는 정도를 결정하는 변수이며, 이 값이 커지면 커질수록 아래와 같은 성질을 갖는다.
잡음 엣지를 더 많이 제거할 수 있게 됨.
엣지를 더 부드럽게 하면서 두껍게 만드는 경향이 있음.
영상에 있는 섬세한 detail들을 제거.
이런 점들을 고려하여, smoothing 된 영상에 대하여 x, y 방향의 미분을 구하면 아래 그림과 같이 된다.
이 결과를 이용해 gradient의 크기와 방향을 구하고, 이렇게 구해진 엣지를 thresholding을 통해 임계값 이상만을 보여주면 아래와 같아진다.
Canny 엣지 검출의 상세 과정 ? Thinning (Non-maximum Suppression)
Canny 엣지 검출에서의 thinning은 앞서 살핀 것과 약간 다르다. 첫번째 과정을 통해 gradient의 방향을 구했기 때문에 gradient의 변화의 방향을 이미 알고 있으므로, 변화하는 방향으로 스캔을 하면서 최대값이 되는 부분을 찾는다.
아래 그림은 thinning을 설명할 때 흔히 사용되는 그림으로 왼쪽과 같이 gradient magnitude 영상이 얻어졌고, gradient의 방향이 오른쪽의 직선과 같은 방향이라면, 직선 방향으로 스캔을 하면서 가장 큰 값만 남기고 픽셀 값을 모두 0으로 만든다. 그러면 1 픽셀단위의 얇은 선이 만들어지게 된다.
앞서 구한 동일한 영상에 대하여 thinning을 적용하면, 아래 그림과 같이 얇은 선을 얻을 수 있으며, 이전과 동일하게 특정 임계값에 대한 thresholding을 한 결과 역시 아래와 같음을 확인할 수 있다.
Canny 엣지 검출의 상세 과정 ? Hysteresis
Hysteresis에 관련된 부분은 [Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [3] 에서 살펴봤으니 그 부분을 참고하면 된다. 아래 그림은 high와 low 및 hysteresis 엣지에 대한 예이며, 그림을 보면 hysteresis thresholding을 사용한 경우가 적절하게 엣지가 살아 있는 것을 확인할 수 있다.
Canny 엣지 검출에서 σ의 영향
아래 그림은 scale의 변화에 따라 검출되는 엣지의 모양이 달라지는 것을 보여주는 예이다. σ가 클수록 강한 엣지가 검출이 되고, σ가 작아지면 섬세한 특징을 보기에 좋다.
또한 σ가 커지면 커지면 엣지 검출은 잘 되지만 엣지의 위치가 smoothing에 의해 달라질 수 있기 때문에, 엣지 검출과 정확도에 trade-off 관계가 있다는 점에도 주의를 해야 한다.
이번 Class에서는 Thinning과 Canny 엣지 검출기에 대하여 알아보았다. 다음 Class에서는 엣지 기반 segmentation 방법에 대하여 좀 더 살펴보고, 엣지 기반 방법은 다음 Class에서 마칠 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [5]
Image Segmentation ? “기본 이론 (part5)”
이전 Class에서는 엣지(edge) 검출 알고리즘의 대표격인 Canny 엣지 검출기에 대하여 살펴보았다. 이번 Class에서는 엣지 기반의 segmentation 방법의 마지막 편으로 SUSAN 엣지 검출기에 대하여 살펴볼 예정이다.
SUSAN 엣지 검출기
1986년에 발표된 Canny 엣지 검출기는 뛰어난 성능으로 인해서 지금까지도 널리 사용이 되고 있다. 1995년에 발표된 SUSAN 엣지 검출기 역시 많이 쓰이는 엣지 검출기 이다.
흔히 SUSAN이라는 여성이 개발한 엣지 검출기라고 생각할 수도 있겠지만, SUSAN은 “Smallest Univalue Segment Assimilating Nucleus”의 약어이고, 영국 옥스포드 대학교의 S.M. Smith와 J.M. Brady가 개발한 엣지 검출기이며, “SUSAN ? A New Approach to Low Level Image Processing”이라는 제목의 논문으로 발표가 되었다.
제목이 의미하듯, low-level 영상 처리에 대한 새로운 접근법을 제시하였으며, 엣지 검출기뿐만 아니라, 잡음 제거에도 꽤 괜찮은 성능을 보인다. 그렇다면 여기서 말하는 “새로운 접근법”이란 무엇을 의미할까?
대부분의 엣지 검출 알고리즘은 미분(derivative, gradient)를 사용하지만, SUSAN에서는 전혀 그렇지 않다. 또한 대부분의 영상 처리에 관련된 알고리즘들이 3x3 혹은 5x5와 같은 정방형 윈도우 혹은 마스크를 사용하지만, SUAN에서는 원형 또는 근접한 원형 마스크를 사용하는 점도 다르다.
SUAN 엣지 검출기를 이해하려면, 먼저 USAN(Univalue Segment Assimilating Nucleus)”를 이해해야 하며, 여기서 Nucleus는 마스크의 중심에 있는 픽셀값을 의미한다. 아래 그림은 논문에 나온 그림으로, 그림에서 원은 마스트를 영역을 나타내며, “+” 부호의 위치에 있는 픽셀이 마스크의 중심(Nucleus)가 된다.
밝은 색 영역과 어두운 색으로 구성된 간단한 위 그림에 5가지 종류의 마스크 형태를 보여준다. “e”는 Nucleus와 주변 픽셀이 모두 같은 경우로 edge나 corner가 아니지만, 나머지 “a ~ d”는 edge나 corner가 마스크에 포함이 된 경우이다.
USAN은 전체 마스크에서 Nucleus와 같은 (혹은 거의 비슷한) 값을 갖는 면적을 뜻한다. 이 면적을 살피면, 평탄한 지역에 있는지, 엣지나 코너에 영역에 있는지 파악이 가능하다.
다음 그림은 위 그림에서 USAN 영역에 해당하는 부분을 나타낸 그림으로 Nucleus와 같은 부분은 흰색으로, 다른 부분은 검은색으로 표시를 한 것이다.
그렇다면 이것을 통해서 어떻게 엣지나 코너를 검출할 수 있을까? 엣지나 코너 검출에는 다음과 같은 기본 식이 적용이 된다.
위 식에서    는 Nucleus이고,   은 마스크 영역에 있는 다른 픽셀 위치를 말하며, I는 필셀의 밝기(intensity) 값이고, t는 유사도를 나타내는 threshold 값이다. 즉 위 식의 의미는 중심에 있는 픽셀의 밝기와 임계값(t) 이내로 비슷한 경우에는 1이 되고, 그 이상의 차이를 보이는 경우는 0이 되는 비선형 필터이다. c는 결과적으로 전체 마스크 내에서 중심과 임계값 범위에 있는 픽셀의 개수를 의미한다.
흔히 SUSAN엣지 검출기에서 사용되는 마스크는 반지름이 3.4를 많이 사용하며, 이 경우 마스크에 있는 픽셀의 개수는 37개 이다. 최소 마스크의 크기는 3x3 이다.
USAN의 의미
그렇다면, 정말 이것을 이용하여 엣지나 코너를 검출할 수 있을까? 그리고 Canny 엣지 검출기 수준의 성능이 얻어질까?
이것을 알아보려면, 먼저 아래 그림을 자세히 보면 가능하다는 것을 확인할 수 있다.
위 그림에서 “part of original image” 부분을 보면, 엣지나 코너가 어떻게 분포되어 있는지를 확인할 수 있다. 이 원영상에 마스크를 적용하여 USAN 값을 구하고, 그것을 3차원 그림에서 표시하며, 위 그림과 같은 형태가 된다.
앞서 말했듯이, 균일한 영역은 USAN 값이 크고, 엣지나 코너로 갈수록 USAN 값이 작아지는 특성이 있으며, 시인성(visibility)를 높이기 위해 USAN 값을 작은 것을 위로 표시를 하면, 코너의 USAN이 가장 작고(즉, 역으로 그린 그림이기 때문에 peak를 보이고), 엣지 부분 역시 값이 낮기 때문에 위 그림에서는 산등성이(ridge) 처럼 보인다.
엣지나 코너 위치 파악
위 그림을 보니, USAN값을 보면 대략적으로 엣지나 코너를 잘 반영하고 있는 것 같은데, 엣지나 코너의 정확한 위치는 어떻게 파악할 수 있을까?
정답은 Canny 엣지 검출기에서처럼, thinning을 이용하는 것이다. Canny 엣지 검출기에서의 thinning 방법은 엣지의 방향을 따라가면서 가장 높은 값을 갖는 위치에 엣지가 있다고 보는 것이다.
SUSAN 엣지 검출기도 마찬가지이다. 전체 중에서 엣지의 방향을 따라 가장 낮은 값을 갖는 USAN을 엣지로 보며, 특히 더 낮은 값은 코너로 보면 된다. 그래서 smallest USAN이라는 뜻에서 SUSAN이 된 것이다.
그런데 한가지 문제가 있다. Thinning을 하려면 엣지의 방향을 따라가면서 체크를 해줘야 하는데, USAN 값을 구할 때는 방향을 구하는 부분이 없다 (Canny 의 경우는 미분을 취하기 때문에 엣지 방향 파악에 전혀 문제가 없다).
이것에 대한 해답은 아래 그림처럼 USAN의 무게 중심을 구하는 것이다. 이 그림을 보면, 가장 작은 마스크 3x3을 이용하여 USAN을 구하여 무게 중심(그림에서는 “o”표 위치)을 구한다. 무게 중심의 위치와 Nucleus의 위치를 비교하면 엣지의 방향을 파악할 수 있게 된다.
즉, 그림에서 ‘a’는 Nucleus의 위치가 무게 중심보다 오른쪽에 있는데, 이런 경우에는 엣지가 오른쪽에 있으며, ‘b’는 그 반대의 경우이다. ‘c’는 무게 중심의 위치와 Nucleus의 위치가 같은데, 이런 경우는 얇은 엣지가 중심에 있는 경우이다.
이렇게 엣지의 방향을 파악한 뒤에 엣지의 방향 쪽으로 스캔을 하면서 가장 낮은 USAN 값을 찾으면 거기에 엣지가 있다.
SUSAN 엣지 검출기의 엣지 검출 성능
[Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [4] 에서 엣지 검출기의 성능을 측정할 때, 3대 핵심 포인트가 Noise smoothing, Edge enhancement, Edge localization”이라는 것을 확인하였다.
SUSAN 엣지 검출기의 성능을 파악하기 위해 다양한 엣지들에 대하여 정확하게 검출이 가능한지를 확인한 그림은 아래와 같다.
이 그림을 보면, 1픽셀과 2픽셀 폭을 갖는 ridge profile에 대하여 정확한 위치 및 두께까지 검출이 가능한 것을 확인할 수 있다. Step edge의 경우에는 정확하게 스텝이 생기는 위치를 파악할 수 있고, ramp edge의 경우 역시 스텝부분을 정확하게 파악할 수 있다. 다른 두가지 엣지 형태도 문제 없음을 알 수 있다.
끝으로 roof edge의 경우는 검출이 안된다. Roof edge는 Canny 엣지 검출기로도 검출이 불가능하며, 사람이 볼 때도 서서히 값이 바뀌기 때문에 엣지로 인식을 하지 못한다.
결론적으로 말하면, SUAN 엣지 검출기는 엣지 검출의 성능이 탁월함을 알 수 있다.
Canny 엣지 검출기와 비교
Canny 엣지 검출기와의 성능 비교를 위해 아래처럼 다양한 엣지가 있는 경우를 실험 영상으로 사용한다.
비교 결과 그림을 보자.
왼쪽은 t =10을 적용한 SUSAN 엣지 검출기 결과이고, 오른쪽은 가장 좋은 성능을 얻을 수 있도록 σ = 0.5를 적용한 Canny 엣지 검출기의 예이다.
Canny 엣지 검출기는 알고리즘의 특성상 코너 부분이 연결이 잘 안되고 끊어지는 부분이 좀 있는데 SUSAN 엣지 검출기는 그럼 부분은 없다. 그렇지만 SUSAN 필터의 경우는 코너 쪽에서 휘어지는 현상이 나타난다.
또 다른 예를 보면, 아래 그림의 왼쪽은 SUSAN이고 오른쪽은 Canny를 사용한 경우이다. SUSAN의 경우는 동그라미를 정확하게 검출을 하지만, Canny에서는 오른쪽이 방향성 문제로 원 모양이 좀 찌그러들었다.
SUAN 엣지 검출기 개발자들은 SUSAN 엣지 검출기의 성능이 더 뛰어나다고 주장을 하고 있지만, t 값을 어떻게 정하고, 마스크를 어떻게 정해야 하는지 고민이 필요하다. Canny edge도 σ 값을 어떻게 정해야 할지 고민이 된다.
단순한 영상뿐만 아니라 자연영상의 경우는 실제 엣지외에도 많은 다른 엣지들이 검출이 되기 때문에 이것을 어떻게 잘 정리해서 정말로 의미 있는 엣지만을 남길 것인지는 검출기 성능을 넘어서는 부분이다. 오히려 이 부분이 더 중요할 수도 있다.
하지만 SUSAN 엣지 검출기의 확실한 장점은 미분 연산을 하지 않기 때문에 속도가 빠르다는 점은 분명한 것 같다. Canny 엣지 검출기 역시 σ를 조절하면 잡음의 영향을 최소화 할 수 있지만 엣지의 정확한 위치가 약간 옮겨가는 문제가 있었다. 하지만 SUSAN 엣지 검출기는 잡음에 강인한 특성이 있다. (이 부분은 논문 참고)
이번 Class에서는 SUSAN 엣지 검출기에 대하여 알아보았다. 엣지 기반 방식은 앞서 설명한 것처럼, segmentation에 필요한 의미 있는 엣지뿐만 아니라, 영상 속에 있는 모든 엣지를 검출하기 때문에 거기서 의미 있는 엣지만을 다시 골라야 하는 어려움이 있다. 이 문제를 해결하기 위한 방법이 영역 기반의 segmentation 방법이며, 다음 Class에서는 영역 기반의 segmentation 방법에 대하여 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [6]
Image Segmentation ? “기본 이론 (part6)”
[Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [3] ~ 1. Image Segmentation 기본이론 [5] 를 통하여 엣지 기반의 segmentation 방법에 대하여 살펴보았다. 엄밀히 말하면, segmentation 방법보다는 엣지를 검출하는 부분에 집중을 한 듯 하다. 엣지 기반의 segmentation 방법은 엣지 검출까지는 직관적이라서 그리 어렵지 않은데, 검출된 많은 엣지 중 과연 어느 것이 segmentation에 필요한 진정한 엣지인지 파악하는 것은 그리 호락호락한 문제가 아니다.
?흔히 후처리에 사용하는 Thinning과 Linking ?만으로는 의미 있는 엣지만을 추출하기가 쉽지 않으며, Hough Transform이나 기타 다른 알고리즘을 사용하더라도, 자연 영상에 대하여 사람이 느끼는 것처럼 ?의미 있는 boundary만을 검출하기가 쉽지 않다. 그리고 아직 보편적인 방법이라고 할만한 방법도 없다.
특히? 복잡한 자연 영상에서 객체의 boundary를 ?정확하게 파악해야 하는 응용이나 의미 있는 객체를 파악해줘야 하는 semantic segmentation에 사용하기에 많이 부족한 감이 있다.
위 그림은 사람이 segmentation 하는 방법과 엣지 기반 방식의 차이를 보여주는 대표적인 그림 중 하나이다. 사람은 기본적으로 의미 있는 엣지만으로 구별해 보는 경향이 있는데, 엣지 기반 방식은 아무리 자잘한 엣지를 잘 제거하고 큰 엣지만을 남기더라도 한계가 있다.
그러므로 엣지 기반 방식의 한계를 극복하는 다른 segmentation 방법이 필요하다. 의미 있는 영역으로 구별하는 방식으로는 최근에는 딥러닝을 사용하여 좋은 결과가 나오고 있다. 그렇지만 딥러닝까지 한번에 넘어가기에는 무리가 있으니, 기본 이론을 좀 더 살펴보기로 한다.
이번 Class에서는 영역 기반의 segmentation 방법에 대하여 살펴볼 예정이다.
영역 기반의 segmentation 방법
엣지 기반 segmentation 방법은 영상에서 차이(difference)가 나는 부분에 집중을 하였다면, 영역(region) 기반의 segmentation 방법은 영상에서 비슷한 속성을 ?갖는 부분(similarity)에 집중하는 방식이며, 엣지 기반 방식이 outside-in 방식이라고 한다면 영역 기반의 방식은 inside-out 방식이라고 볼 수 있다.
여기서 비슷한 속성이란 흔히 사용하는 밝기(intensity, gray-level)뿐만 아니라, 컬러나 표면의 무늬(texture) 등도 해당이 되며, 영역 기반의 방법은 기본적으로 인지가 가능한 객체의 경계를 다른 것들과 구별짓는 그 무엇이라고 할 수 있다.
참고로 아래와 같은 극단적인 영상을 올바르게 segmentation 할 방법은 고민해보자. 원래의 영상은 왼쪽의 영상인데 배경과 비슷한 밝기로 나비 모양과 육각형 모양이 있다. 크기자 작아서 잘 보이지 않기 때문에 오른쪽은 확대한 결과를 보여주는 영상이다.
이 영상에 대하여, 앞서 살펴본 Canny 엣지 검출기로 엣지를 검출하는 경우와, 영역 기반 방식 중 평균과 표준편차를 이용해 구별하는 경우에 대한 결과는 아래 그림과 같다. 아래 그림을 보면 표준 편차를 객체를 구별할 수 있는 속성으로 이용하면 구별이 되는 것을 확인할 수 있다.
이 경우처럼, 영역 기반의 방법에서는 객체를 구별할 수 있는 중요한 속성을 잘 선정하면, 엣지 기반 방식에서는 거의 불가능하다고 볼 수 있었던 것을 구별할 수도 있다.
Region을 정하는 방법은?
영역 기반의 에서 region을 결정하는 방식은 매우 다양하며 아래와 같은 방식들이 있다.
Region Growing
Region Merging
Region Splitting
Split and Merge
Watershed
…..
Region Growing 알고리즘
영역 기반의 방식에서 가장 많이 사용되는 방식이 region-growing 알고리즘이다. 이 방식은 기준 픽셀을 정하고 기준 픽셀과 비슷한 속성을 갖는 픽셀로 영역을 확장하여 더 이상 같은 속성을 갖는 것들이 없으면 확장을 마치는 방식이다.
아래 그림은 꽃잎 부분을 같은 영역이라고 찾아내는 예를 보여준다. 먼저 임의의 픽셀을 seed로 정한 후 같은 속성(아래의 예에서는 컬러)을 갖는 부분으로 확장을 해나가면 오른쪽처럼 최종 영역을 구별해 낼 수 있게 된다. 그림에 있는 다른 영역들도 비슷한 방식을 적용하면 구별이 가능해지게 된다.
Region growing 방식에서 중요한 seed(시작 픽셀)를 정하는 방식은 보통 다음과 같은 3가지 방식을 사용한다.
사전에 사용자가 seed 위치를 지정
모든 픽셀을 seed라고 가정
무작위로 seed 위치를 지정
또한 seed 픽셀로부터 영역을 확장하는 방식도 다음과 같은 여러 방식이 사용이 되고 있다.
원래의 seed 픽셀과 비교: 영역 확장 시 원래의 seed 픽셀과 비교하여 일정 범위 이내가 되면 영역을 확장하는 방법. 잡음에 민감하고, seed를 어느 것으로 잡느냐에 따라 결과가 달라지는 경향이 있음.
확장된 위치의 픽셀과 비교: 원래의 seed 위치가 아니라 영역이 커지면 비교할 픽셀의 위치가 커지는 방향에 따라 바뀌는 방식. 장점은 조금씩 값이 변하는 위치에 있더라도 같은 영역으로 판단이 되나, 한쪽으로만 픽셀값의 변화가 생기게 되면 seed와 멀리 있는 픽셀은 값 차이가 많이 나더라도 (심각한drift 현상) 같은 영역으로 처리될 수 있음. 아래 그림의 경우 컬러가 조금씩 차이가 나더라도 인접 픽셀만 비교하면서 영역을 확장하면 전체가 같은 영역이 될 수 있다.
영역의 통계적 특성과 비교: 새로운 픽셀이 추가될 때마다 새로운 픽셀까지 고려한 영역의 통계적 특성(예를 들면 평균)과 비교하여 새로운 픽셀을 영역에 추가할 것인지를 결정. 영역 내에 포함된 다른 픽셀들이 완충작용을 해주기 때문에 약간의 drift는 있을 수 있지만 안전. “centroid region growing” 이라고도 함.
Region Growing 방식의 예
Region Growing 방식은 많이 널리 사용이 되는 방식으로 “위키”에 나와 있는 예를 이용해 설명을 한다. 아래 그림은 번개가 치는 날을 찍은 사진인데, 여기서 번개의 가장 밝은 부분을 연결하여 그리고자 한다.
위 흑백 영상의 밝기는 0부터 255까지 분포하는데, 그 중에서 밝은 영역 만을 연결하고 싶다면, seed 값을 255로 설정을 하면 된다. 이렇게 seed를 설정한 영상은 아래 그림과 같다.
이렇게 설정된 seed를 바탕으로 임계값을 변경시키면서 region을 확장하면 아래와 같은 결과를 얻을 수 있다.
위와 같은 응용에서는 region growing 방식을 매우 효과적으로 사용할 수 있으며, 임계값을 어떻게 설정하느냐에 따라 다른 결과를 얻을 수 있음을 확인할 수 있다.
Region Growing 방식의 장단점
이 방식의 장점은 다음과 같다.
처리 속도가 빠르다.
개념적으로 단순하다.
Seed 위치와 영역 확장을 위한 기준 설정을 선택할 수 있다.
동시에 여러 개의 기준을 설정할 수도 있다.
하지만 다음과 같은 단점도 있다.
영상의 전체를 보는 것이 아니라 일부만 보는 “지역적 방식(local method)”이다.
알고리즘이 잡음에 민감하다.
Seed 픽셀과 비교하는 방식이 아니면 drift 현상이 발생할 수 있다.
?
?
이번 Class에서는 영역 기반의 segmentation 방법의 기본적인 방법인 region growing 방식에 대하여 살펴보았다. 다음 Class에서는 다른 영역 기반의 segmentation 방법에 대하여 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [7]
Image Segmentation ? “기본 이론 (part7)”
[Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [6] 에서는 영역 기반의 segmentation 방법에서 많이 사용이 되는 region growing 방법에 대하여 살펴보았다. Region growing 방법은 1개 또는 여러 개의 기준 픽셀(seed)을 기반으로 주변 픽셀과의 비교를 통해 비슷한 특성을 갖는 픽셀들을 같은 영역으로 간주하는 방법임을 확인하였다.
이번 Class에서는 영역 기반의 방법 중 region merging (영역 결합) 및 region splitting (영역 분리) 방법에 대하여 살펴볼 예정이다.
Region Merging의 기본 개념과 방법
영역 기반의 방법에서 merging과 splitting은 대표적인 방법으로, merging과 splitting은 서로 반대 방향으로 움직일 뿐이지, 어찌 보면 그 기본 개념은 동일하다고 볼 수 있다.
Region merging은 그 이름이 의미하듯이 비슷한 속성을 갖는 영역들을 결합시켜 동일한 꼬리표(label)를 달아주는 방식이다. region merging은 어떤 경우는 매 픽셀 단위가 될 수 있으며, 일반적으로 심하게 나뉜 영역(over-segmented region)을 시작점으로 하며, 일반적으로 아래와 같은 과정을 거친다.
인접 영역을 정한다.
비슷한 속성인지를 판단할 수 있는 통계적인 방법을 적용하여 비교한다.
같은 객체라고 판단이 되면 합치고, 다시 통계를 갱신한다.
더 이상 합칠 것이 없을 때까지 위 과정을 반복한다.
위 방식을 보면, region growing과 거의 유사하다는 생각이 들 것이다. Region growing은 region merging 방법 중 하나이며, region growing 방법은 1개 혹은 적은 수의 seed를 사용하는 방식으로 픽셀 단위로 판단을 하는 점만 차이가 있다. 반면에 merging은 영역을 기본 단위로 하며, 물론 가장 작은 영역은 픽셀이기 때문에 픽셀을 기본 영역으로 볼 수 있음, 그림 전체에 여러 개의 seed를 사용한다고 볼 수 있다.
Region Merging 좀 더 상세하게
아래와 사진에서 “사과를 배경과 분리”하고 싶은데, 여기에 region merging을 적용한다고 해보자. 사람이 보면 너무나 간단해 보이지만, 아래 그림은 그리 쉽지만은 않다.
일단 매 픽셀에 각각의 꼬리표(label)를 할당하고, 이것을 기반으로 하여 밝기 값의 차가 10 이하이면 동일한 영역으로 본다고 가정을 하고, merging 알고리즘을 충실하게 적용을 하여 분리를 하면 아래 그림과 같다.
흑백 영상의 밝기 값만을 고려하여 merging을 적용하면, 단계적으로 변하는 위 영상의 특성으로 사과와 배경이 2개 혹은 3개의 영역으로 깔끔하게 분리가 되지는 못하지만 어느 정도 구별이 가능하다는 것을 확인할 수 있다.
Region merging 방법에서 한가지 신경 써야 할 점은 처리되는 순서에 따라서 결과값이 달라진다는 점이다. 아래 그림은 위 실험보다 밝기 단계를 좀 더 크게 하고 분리를 하는 경우로, 오른쪽과 왼쪽의 결과가 다르다. 오른쪽은 그림은 상하 반전 시킨 상태에서 merging을 적용하고 다시 상하 반전을 시킨 경우이다.
Region Splitting의 기본 개념과 방법
영역 분리(region splitting) 방식은 merging과는 정반대 개념이라고 보면 된다. 그림 전체와 같은 큰 영역을 속성이 일정 기준을 벗어나면 쪼개면서 세분화된 영역으로 나누는 방식을 사용한다.
보통은 아래 그림과 같이 4개의 동일한 크기를 갖는 영역으로 나누기 때문에 quad-tree splitting 방식을 많이 사용한다. 아래 그림은 큰 영역을 먼저 4개의 영역으로 나누고, 다시 각 영역을 검토하여 추가로 나눠야 할 것인지를 결정한다. 그림에서는 2번 영역이 추가로 나눠야 하기 때문에, 21부터 24까지 다시 4개의 영역으로 나눈다. 다시 검토를 해보니 23영역을 4개의 영역으로 나눈 것이다.
이렇게 region splitting 방식에서는 해당 영역에서 “분산(variance)”이나 “최대값과 최소값의 차”와 같은 통계 방식의 일정 기준을 설정하고, 그 값이 미리 정한 임계 범위를 초과하게 되면 영역을 분할한다.
Region Splitting 좀 더 상세하게
앞서 살펴본 동일한 사과 사진에 대하여 Region splitting 방식을 적용하면 아래와 같다.
이 그림만으로는 region splitting을 적용했을 경우 segmentation이 잘 되었다라고 볼 수가 없을 것 같다.
위 그림이 이렇게 보이는 이유는 동일하지 않은 속성에 대하여 splitting을 하면 사실은 인접한 부분끼리 비슷한 부분이 나올 수 있음에도 splitting 방식의 원리가 그렇기 때문에 같은 속성을 갖는 부분도 다른 꼬리표가 붙게 된다.
Split & Merge 방법
앞서 살펴본 것과 같이, splitting 방식만으로는 원하는 결과를 얻기가 어렵기 때문에 동일한 영역을 다시 합쳐주는 과정을 거쳐야 한다. 이것이 바로 split & merge 이다. 앞서 splitting 에서 살펴본 동일한 그림을 다시 살펴보자.
Quad-tree splitting 방식을 사용하면 실제로는 모양이 위 그림과 같더라도 여러 개의 영역으로 분리될 수 밖에 없기 때문에 over-segmentation이 일어난다. 이것을 같은 특성을 갖는 부분끼리 다시 묶어주면 {1, 3, 233}과 {4, 21, 22, 24, 231, 232, 234} 두 개의 영역으로 된다.
이것을 split & merge라고 한다.
앞서 살펴본 사과 사진에 대하여 split을 적용하고 merge를 적용한 결과는 아래 그림과 같다. 이 그림에서 볼 수 있는 것처럼 의미 있는 영역으로 구별됨을 확인할 수 있다.
그렇다면 region merging과 split & merge의 차이점은 무엇일까?
차이점은 split & merge가 좀 더 속도가 빠르다는 점이다. 단순 merge가 막고 품는 것과 같다면, quad-tree splitting을 적용하면, 비슷한 영역은 통으로 처리가 되기 때문에 좀 더 속도가 빠르다. 이것은 단순 검색보다는 검색 방법은 개선한 binary search가 효율적인 것과 비슷하다고 볼 수 있다.
이번 Class에서는 영역 기반의 segmentation 방법의 기본적인 방법인 region merging, region splitting, split & merge방식에 대하여 살펴보았다. 다음 Class에서는 다른 영역 기반의 Watershed segmentation 방법에 대하여 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
1. Image Segmentation 기본이론 [8]
Image Segmentation ? “기본 이론 (part8)”
지난 [Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [6] ~ 1. Image Segmentation 기본이론 [7] 를 통하여 영역 기반의 segmentation 방법에 대하여 살펴보았다. 영역 기반의 방식은 기본적으로 밝기 값, 컬러, 표면의 특성 등 비슷한 속성을 갖는 부분을 같은 영역으로 지정을 하여 객체를 분류하는 방식을 주로 사용한다.
Watershed segmentation 방식은 region growing, merging 및 splitting 과는 조금 다른 접근 방식을 취한다. 비슷한 속성에 주목을 하는 것이 아니라, 지형학에서 사용하는 개념들과 비슷한 방식으로 영역을 나누는 방식이다.
이번 Class에서는 watershed segmentation 방법에 대하여 살펴볼 예정이다.
Watershed segmentation 방식의 기본 개념
Watershed segmentation 방식은 1979년에 Serge Beucher와 Christian Lantuej의 “Use of Watershed in Contour Detection” 이라는 논문을 통해 처음 소개되었다. 이후 널리 사용이 되고 있으며, 특히 의료분야에서 많이 사용이 되고 있다.
Watershed 방식을 설명할 때 나오는 중요한 2가지 개념은 Watershed 와 Catchment Basin이며, 아래 그림을 보면 쉽게 이해가 갈 수 있을 것이라 생각된다.
Watershed는 산등성이나 능선처럼 비가 내리면 양쪽으로 흘러 내리는 경계에 해당이 되며, Catchment Basin은 물이 흘러 모이는 집수구역에 해당이 된다. Watershed는 기본적으로 영역을 구분해주는 역할을 하기 때문에 Watershed만 구하면 영상을 segmentation 할 수 있게 된다.
최초의 S. Beucher의 논문에 발표된 것처럼, Watershed는 gradient를 이용하여 구한다. 먼저 영상에서 gradient를 구하고, 엄밀히 말하면 gradient의 크기로 구성된 gradient magnitude 영상을 구하고, 그 영상으로부터 Watershed를 구한다.
아래 그림에서 맨 왼쪽은 gradient의 크기를 눈에 잘 띌 수 있게 부조(relief) 형태로 보여주는 것이고, 두번째 영상은 gradient 크기 영상으로 밝을수록 gradient의 크기가 크다. 세번째 영상은 gradient 크기를 이용해 Watershed를 구한 것이며, 의미 없는 부분은 제거가 되었다. 맨 마지막 영상은 첫번째 영상에 비하여 한결 간결해진 결과를 부조 형태로 다시 보여준 것이다. (위키피디아에 그림 참고)
?
이것을 보면, 다른 segmentation 방식과 마찬가지로 의미 있는 객체의 외곽선을 구하려면 다른 많은 segmentation 방법처럼, 뭔가 간결화를 위한 수단이 필요함을 알 수 있다.
Watershed를 구하는 방법
Watershed를 구하는 알고리즘은 다양하다. 그 이유는 Watershed의 개념이 소개된 이후 많은 사람들이 좀 더 효율적으로 Watershed를 구할 수 있는 방식을 발표하였기 때문이다.
여기서는 가장 많이 쓰이는 Flooding 알고리즘에 대하여 살펴볼 예정이다. 이 알고리즘은 기본적으로 각각의 Catchment Basin의 최소값(local minima)에 구멍이 있고, 그 구멍에서 물이 차오르기 시작한다고 가정에서 출발한다. 물이 일정 수위에 오르게 되면, 서로 다른 2개의 Catchment Basin이 합쳐지는 상황이 발생할 수 있는데, 이 때는 물이 합쳐지는 막기 위해 댐(dam)을 설치한다. 이 댐이 바로 영역을 구별하는 영역의 경계(boundary) 역할을 하게 되며, 이것이 Flooding 알고리즘의 기본 개념이다.
아래 그림 A는 3개의 basin에 각각 마커를 할당한 경우로, 각 basin에서 물이 점점 차오르면, ?어느 순간에 2개의 basin이 합쳐지는 상황이 발생하게 되는데, 그러면 그 자리에 댐을 건설한다. B는 애초에 2 개의 마커만을 할당하였기 때문에, 최종적으로는 영역이 2개로 나눠지게 된다.
Watershed 알고리즘 적용 예
아래 그림과 같은 사진을 Watershed 알고리즘을 통해 구별한다고 해보자. 아래 그림은 칼라 영상을 받아, 흑백 영상으로 만든 결과이다. (편의상 사진은 MathWokrs 자료를 사용)
위 영상에 대하여, 소벨(Sobel) 필터를 사용하여 gradient magnitude 영상을 구하고, 그것으로부터 순수하게 Watershed 알고리즘을 사용하여 segmentation을 사용하면 아래와 같이 over-segmentation 된 영상을 얻게 된다.
위 그림처럼, 심하게 segmentation 영상을 얻게 되는 이유는 오른쪽 그림에서 볼 수 있듯이 gradient magnitude 영상이 아주 작은 크기로 localize 되는 형태를 보이기 때문이다.
Over-segmentation된 영상에 대한 후처리는 이전 [Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [7] 에서 살펴본 것처럼, region merging 방법을 사용할 수도 있고, 다른 방법을 사용할 수도 있다.
위 결과처럼, over-segmentation이 발생하는 이유는 영상에 존재하는 detail 이나 잡음이 gradient를 구하였을 때 local minimum을 만들어내기 때문이다. 그러므로 사전에 영상을 부드럽게 만들어 주는 smoothing 필터를 적용하거나, 잡은 제거에 효과가 탁월한 모포로지(morphology) 연산을 적용 적용하여 gradient magnitude 영상을 단조롭게 만드는 것도 해결책이 된다.
Marker-Controlled Watershed Segmentation
앞서 살펴본 것처럼, 자연 영상에 watershed 알고리즘을 적용하면, 대부분 over-segmentation 문제가 발생을 하며, 이 over-segmentation 문제를 피하기 위한 방법으로 마커(marker)를 사용하여 segmentation 될 영역을 지정하는 방식이다. 마커의 지정은 수동이나 자동으로 가능하며, 보통은 최종적으로 segmentation 되는 영역의 수와 marker의 수가 동일하다. 마커를 지정할 때는 보통 blob 연산을 많이 사용하며, 모포로지 연산을 같이 사용하는 것이 일반적인 추세이다.
“Watershed를 구하는 방법”의 그림 B를 보면 “seg2” 자리에는 마커가 없기 때문에 “seg3”와 통합이 되는 것을 확인할 수 있다.
아래 그림에서 왼쪽은 원래 영상이며, 오른쪽은 watershed 알고리즘을 적용하여 얻은 over-segmented image 이다.
이 문제 해결을 위해 blob 이미지에 marker를 할당하면 아래 그림의 왼쪽이 되며, marker가 할당된 영역에 대해서만 watershed 알고리즘을 적용한 결과는 오른쪽 그림과 같다.
이번 Class에서는 watershed segmentation 방법에 간단하게 살펴보았다. Watershed 방법은 1979년 처음 발표 된 이후 최근까지도 논문이 꾸준히 발표가 되고 있으며, 기본 watershed 알고리즘에 컴퓨터 비전 알고리즘을 결합하여 over-segmentation에 관련된 문제를 줄이면서 효율적으로 객체의 경계를 검출 능력을 강화하기 위한 방법들을 제시하고 있다. 또한 CT나 MRI와 같은 의료 영상 처리에서 watershed 알고리즘이 많이 사용이 되고 있다.
이것으로 segmentation에 대한 기본편을 마치고, 다음 Class부터는 machine learning 방법에 근거한 sematic segmentation에 다가가는 방법들에 대하여 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
2. Selective Search [1]
Semantic Segmentation ? “Selective Search (part1)”
?
지난 [Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [1] ~ 1. Image Segmentation 기본이론 [8] 를 통하여 image segmentation의 기본적인 방법들에 대하여 살펴보았다. Image segmentation은 결과적으로 object regcognition이나 semantic segmentation으로 이어지는 기반 기술이기 때문에 매우 중요하며, 終局에는 사람이 영상을 이해하는 것처럼, 눈앞에 펼쳐지는 장면들을 이해할 수 있는 “scene understanding” 기술이 될 것이다. 현재 딥러닝 기술을 활용하여 활발하게 연구가 진행되고 있기 때문에 향후 3~5년 이내에 어느 정도 사람이 인식하는 수준에 도달할 수 있을 것으로 본다.
Semantic segmentation의 첫번째 테마로 무엇을 다룰까 고민을 하였다. 아무래도 object detection을 하려면, 먼저 객체의 후보 영역을 선정을 해야 하는데, 여기서 탁월한 성능을 보인 “Selective Search” 방법을 살펴보는 것이 적당할 것 같아, 이번 Class에서는 이 방법에 대하여 살펴볼 예정이다.
Selective Search란?
SS(이후부터Selective Search를 간단하게 줄여서 SS로 표시)란 Uijlings와 Sande 등이 발표한 알고리즘으로 2013년 ILSVRC에 참여하여, “detection” 분야에서 뉴욕대의 “Overfeat” 방식을 누르고 1위를 차지하였다.
이후 object detection 분야의 CNN으로 유명한 R-CNN, SPPNet, Fast R-CNN 방식에 후보 영역을 추천에 사용이 되었고, R-CNN 및 R-CNN 개선 알고리즘들이 워낙 탁월한 성능을 보였기 때문에 같이 주목을 받게 되었다.
하지만, 후보 영역 추천 과정이 실제 객체 검출 CNN과 별도로 이루어지기 때문에, SS를 사용하면 진정한 의미에서 end-to-end 학습을 시키는 것도 불가능하고, 실시간 적용에도 어려움이 있다. 이후 Faster R-CNN, YOLO, FCN과 같은 방식들이 나오면서 후보 영역을 선정하는 부분이 CNN 망으로 들어오거나 구조 자체의 변화가 생기면서, 한동한 hot 하게 사용이 되다가 다시 주춤해진 상태이다.
Selective Search의 목표
SS란 객체 인식이나 검출을 위한 가능한 후보 영역을 알아낼 수 있는 방법을 제공하는 것을 목표로 한다. 물론 SS가 발표되기 이전에도 동일한 목표를 위한 논문들이 발표되었지만, SS가 기존에 발표되었던 다른 방식들보다 우수하다.
SS가 주목한 부분은 exhaustive search 방식과 segmentation 방식을 결합하여 보다 뛰어난 후보 영역을 선정하는 것이다. Exhaustive search 는 그 이름이 의미하듯 후보가 될만한 모든 영역을 샅샅이 조사하는 방식을 말한다. 후보가 될만한 대상의 크기(scale)가 일정하지도 않고, 또한 가로/세로 비율(aspect ratio)도 일정하지 않은 상황에서 모두 찾는 것은 연산시간 관점에서는 수용이 불가하다. 마치 한강에서 물고기를 잡기 위해 한강 양쪽을 막아놓고 양동이로 물을 퍼내는 것과 마찬가지이다.
segmentation 방법은 exhaustive search과 같이 영상의 특성을 고려하지 않고 찾는 것이 아니라, 이전 [Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [1] ~ 1. Image Segmentation 기본이론 [8] 과정에서 살펴본 것과 같이 영상 데이터의 특성에 기반하는 방식이다. 색상, 모양, 무늬 등 다양한 기준에 따라 segmentation이 가능하지만, 모든 경우에 동일하게 적용할 수 있는 segmentation 방식을 찾기란 불가능하다.
?
?
위 그림에서 (a)를 보면, 테이블 위에 샐러드 접시가 있고 샐러드 접시에는 샐러드 덜어먹을 때 쓰는 집게(tongs)가 있는 것과 같이 영상이 본질적으로 계층적임을 보여준다. 또한 테이블을 가리킬 때도 ‘아무것도 없는 것도 없는 나무 테이블만을 테이블이라고 할까, 아니면 테이블 위에 있는 모든 물체들까지 포함한 것을 테이블이라고 해야 할까’는 고민 거리다. (아래 그림 참고)
그림(b)는 무늬(texture)는 같지만, 색으로 고양이를 구별해야 하는 경우이고, (c)는 카멜레온과 나뭇잎을 색으로 구별하기는 어렵지만 무늬로 구별을 해야 하는 경우이며, (d)는 자동차의 바퀴가 자동차의 본체와 색깔과 무늬가 다르지만 자동차에 있는 일부분이기 때문에 자동차로 고려해줘야 한다.
사람은 구별하고자 하는 대상의 무늬, 크기, 색상 등이 달라도, 그리고 선험적인 지식(prior)가 없더라도 0.1초도 걸리지 않고 기가 막히게 유의미한 객체로 분리(semantic segmentation)하여 인지할 수 있다. 하지만 안타깝게도 아직까지 발표된 어떤 이론도 이러한 기준을 만족시키는 방법은 없다. 아직은 어떤 원리로 사람이 이렇게 인지하는지 정확하게 파악이 되지 않았기 때문일 수도 있다.
SS는 exhaustive search와 같은 무식한 방법으로 후보 영역을 선정하는 대신에, 비록 segmentation 방법이 한계는 있지만, segmentation에 동원이 가능한 다양한 모든 방법을 활용하여 seed를 설정하고, 그 seed에 대하여 exhaustive한 방식으로 찾는 것을 목표로 하고 있다. 논문에서는 이것을 segmentation 방법을 가이드로 사용한 data-driven SS라고 부른다.
아래 그림은 그 과정을 보여주는 것이다. 입력 영상에 대하여 segmentation을 실시하면, 윗줄의 왼쪽에 해당하며, 이것을 기반으로 후보 영역을 찾기 위한 seed를 설정한다. 그러면 아랫줄의 왼쪽에 해당하는 것처럼, 엄청나게 많은 후보가 만들어지게 되며, 이것을 적절한 기법을 통하여 통합을 해나가면, segmentation은 윗줄 오른쪽과 같은 형태로 바뀌게 되며, 결과적으로 그것을 바탕으로 후보 영역이 통합되면서 개수가 줄어들게 된다.
이 과정은 이전에 Sunny 엣지 검출기에서 최대가 아닌 자잘한 엣지들을 없앨 때 사용했던 non-maximum suppression이나 region growing과 같은 기법들을 떠올리면 도움이 될 것 같다.
종합하면, SS의 목표는 아래 3가지로 요약이 가능하다.
영상은 계층적 구조를 가지므로 적절한 알고리즘을 사용하여, 크기에 상관없이 대상을 찾아낸다.
컬러, 무늬, 명암 등 다양한 그룹화 기준을 고려한다.
빨라야 한다.
영상에 존재한 객체의 크기가 작은 것부터 큰 것까지 모두 포함이 되도록 작은 영역부터 큰 영역까지 계층 구조를 파악할 수 있는 “bottom-up” 그룹화 방법을 사용한다.
Selective Search의 3단계 과정
SS는 아래와 같은 3단계 과정을 거친다.
1. 일단 초기 sub-segmentation을 수행한다.
이 과정에서는 Felzenszwalb가 2004년에 발표한 “Efficient graph-based image segmentation” 논문의 방식처럼, 각각의 객체가 1개의 영역에 할당이 될 수 있도록 많은 초기 영역을 생성하며, 아래 그림과 같다.
2. 작은 영역을 반복적으로 큰 영역으로 통합한다.
이 때는 “탐욕(Greedy) 알고리즘을 사용하며, 그 방법은 다음과 같다. 우선 여러 영역으로부터 가장 비슷한 영역을 고르고, 이것들을 좀 더 큰 영역으로 통합을 하며, 이 과정을 1개의 영역이 남을 때까지 반복을 한다.
아래 그림은 그 예를 보여주며, 초기에 작고 복잡했던 영역들이 유사도에 따라 점점 통합이 되는 것을 확인할 수 있다.
3. 통합된 영역들을 바탕으로 후보 영역을 만들어 낸다.
이 과정을 통합적으로 보여주는 과정은 아래와 같다.
이번 Class에서는 Selective Search의 기본 개념에 대하여 알아보았다. 다음 Class에서는 영역 기반의 Selective Search의 좀 더 상세한 부분에 대하여 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
2. Selective Search [2]
Semantic Segmentation ? “Selective Search (part2)”
지난 [Part VII. Semantic Segmentation] 2. Selective Search [1] 에서 Selective Search(SS)는 segmentation을 통해 검출을 원하는 후보가 있을 것 같은 영역에 대한 guide를 설정한 뒤 그 영역을 중심으로 exhaustive search를 하여 최종적으로 후보 영역을 확정한다는 사실을 알게 되었다.
이번 Class는 SS에서 sub-segmentation에 사용하는 Felzenszwalb의 segmentation 방식에 대하여 좀 더 상세하게 살펴볼 예정이다.
Hierarchical Grouping or Bottom-up Grouping?
?
[Part VII. Semantic Segmentation] 2. Selective Search [1]에서 살펴본 것처럼, 영상 속의 물체가 겹쳐 있을 수 있기 때문에, 후보 영역을 정할 때는 이것을 고려한 계층적인 그룹화가 정말 중요하다. 하지만 대부분의 segmentation 알고리즘은 계층구조를 고려하는 것이 아니라, seed로부터 비슷한 속성을 갖는 인접 영역으로 확장하는 방식이기 때문에 bottom-up 성격이 강할 수 밖에 없다.
SS는 segmentation을 후보 영역을 검출하기 위한 가이드로 사용하기 때문에 후보 영역의 선정이 정말 중요하기 때문에  segmentation 방법을 정할 때 고민이 많았을 것 같다.
논문에서 선택한 방식은 Felzenszwalb와 Huttenlocher가 제안한 segmentation 방식을 사용하였는데, 이 방식은 비교적 간단하면서도 그 성능이 뛰어나다.
?
?
Felzenszwalb와 Huttenlocher의 Segmentation 알고리즘
?
코넬대의 Felzenszwalb는 2004년 인용횟수가 4400번이 넘는 뛰어난 논문 “Efficient Graph-based Image Segmentation”을 발표하였다. 이 논문의 저자는 “인지적인 관점에서 의미 있는 부분을 모아서 그룹화”와 “연산량 관점에서 효율성 증대”라는 2 개의 목표를 기반으로 새로운 segmentation 기법을 개발하였다. 물론 기존에도 “의미 있는 부분을 모아서 segmentation” 하는 기법들이 있기는 했지만, 연산량 관점에서는 매력적이지 않았다.
이 논문에서는 아래 그림 (a)와 같은 합성 이미지를 이용해 segmentation을 수행하는데, 만약에 사람이 인지하는 방식으로 제대로 segmentation을 수행한다면 그림 (b)와 같이 크게 3개의 영역으로 나뉘어져야 한다.?
그림 (a)를 보면 왼쪽 부분은 밝기가 조금씩 변하기 때문에 그림 (b)의 왼쪽과 같이 segmentation을 한다는 것이 쉽지 않다. 또한 (a)의 오른쪽을 보면 바코드와 같은 형태가 있는데 이것을 여러 개의 작은 영역으로 구분하지 않고 (b)의 오른쪽 그림처럼 1개의 영역으로 깔끔하게 구별하는 것 역시 기존 segmentation 알고리즘으로 쉽지 않다.  그림 (c)는 사람이 하는 것처럼 의미 있는 부분만을 모아서 그룹화를 제대로 시키지 못한 경우를 보여준다.
논문에서는 사람이 인지하는 방식으로의 segmentation을 위해 graph 방식을 사용하였다.  그래프 이론 G = (V, E)에서 V는 노드(virtex)를 나타내는데, 여기서는 픽셀이 바로 노드가 된다.
?
기본적으로 이 방식에서는 픽셀들 간의 위치에 기반하여 가중치(w)를 정하기 때문에 “grid graph 가중치” 방식이라고 부르며, 가중치는 아래와 같은 수식으로 결정이 되고 graph는 상하좌우 연결된 픽셀에 대하여 만든다.
E(edge)는 픽셀과 픽셀의 관계를 나타내며 가중치 w(vi, vj)로 표현이 되는데, 위 식에서 알 수 있듯이 가중치는 픽셀간의 유사도가 떨어질수록 큰 값을 갖게 되며, 결과적으로 w 값이 커지게 되면 영역의 분리가 일어나게 된다.
위 그림과 같이 C1과 C2가 있는 경우에, 영역을 분리할 것인지 혹은 통합할 것인지를 판단하는 아래와 같은 수식을 사용한다.
위 식에서 Dif(C1, C2)는 두개의 그룹을 연결하는 변의 최소 가중치를 나타내고, MInt(C1, C2)는 C1과 C2 그룹에서 최대 가중치 중 작은 것을 선택한 것이다. 즉, 그룹간의 차가 그룹 내의 차보다 큰 경우는 별개의 그룹으로 그대로 있고, 그렇지 않은 경우에는 병합을 하는 방식이다.
?
이렇게 비교적 간단한 알고리즘으로 segmentation을 수행했음에도 불구하고 아래 그림과 같이 양호한 결과를 얻을 수 있다. 운동장의 잔디 부분의 픽셀값들이 변화가 꽤 있음에도 불구하고 같은 영역으로 처리하는 것을 확인할 수 있다. 물론 담장 부분 역시 운동장으로 처리되는 오류가 있기도 하다.
논문에서는 인접한 픽셀끼리, 즉 공간적 위치 관계를 따지는 방법뿐만 아니라 feature space에서의 인접도를 고려한 방식도 제안을 하였다. 이 방식은 “nearest neighbor graph 가중치” 방식이라고 부르며, 적정한 연산 시간을 유지하기 위해 feature space에서 가장 가까운 10개의 픽셀에 한하여 graph를 형성한다.
이를 위해 모든 픽셀을 (x, y, r, g, b)로 정의된 feature space로 투영을 하여 사용하는데, (x, y)는 픽셀의 좌표이고 (r, g, b)는 픽셀의 컬러 값이다. 가중치에 대한 설정은 5개 성분에 대한 Euclidean distance를 사용하였다. 그룹화 방식은 동일하다.
?
아래 그림은 합성 이미지에 대하여 feature space 방식을 사용하여 segmentation을 실시한 경우이다. 배경에 잡음이 많이 있음에도 불구하고 같은 영역으로 통합이 되는 것을 확인할 수 있다.
아래 그림은 feature space 방식을 사용하여 segmentation을 실시했을 때 아주 좋은 결과를 얻을 수 있는 대표적인 예를 보여준다. 왼쪽의 경우, 구름이 있음에도 불구하고 하늘이 거의 같은 영역으로 묶이고, 잔디의 경우도 상당한 변화가 있음에도 불구하고 동일 영역으로 묶이는 것을 확인할 수 있다.
?
오른쪽의 에펠탑 역시 중간 밝은 노란색 불이 영역을 나누고 있기 때문에 기존 segmentation 방법으로는 에펠탑을 같은 대상으로 처리하기가 어렵지만, 본 논문의 방식을 사용하면 오른쪽과 같은 좋은 결과를 얻을 수 있다.
Felzenszwalb의 segmentation 알고리즘은 앞서 살펴본 것처럼 결과가 비교적 좋고 연산 속도가 매우 빨라 Selective Search의 3단계 과정 첫번째 단계에 적용이 되었다. 이 논문의 방식은 절대적이거나 완벽하지는 않지만, SS에서는 객체 검출을 위한 guide로 사용을 하기 때문에 그 관점에서 보면 적당한 알고리즘인 것 같다.
대부분의 경우, Felzennszwalb알고리즘을 적용하더라도 상당히 많은 영역이 만들어지기 때문에 이 영역들을 효율적으로 병합하는 방법이 필요하다. Selective Search에서는 이 척도로 유사도(similarity)를 사용하게 되는데 이 부분은 다음 Class에서 살펴볼 예정이다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
2. Selective Search [3]
Semantic Segmentation ? “Selective Search (part3)”
[Part VII. Semantic Segmentation] 2. Selective Search [2] 에서는 Selective Search(SS)의 sub-segmentation에 사용되는 Felzenszwalb의 segmentation 방법에 대하여 살펴보았다. SS는 segmentation을 가이드로 사용하여 exhaustive search를 수행하기 때문에, 성능이 꽤 우수하고 연산량이 적은 알고리즘을 선택하였다.
논문 제목 (객체 인식을 위한 선택적인 탐색)에서 드러나듯이 아주 정교한 segmentation이 아니더라도 상관없다. 왜냐하면, SS 를 사용하여 후보 영역을 찾은 다음에 사용하는 분류기에서(본 논문에서는 SVM을 분류기로 사용, R-CNN 에서는 AlexNet을 기본 신경망으로 사용) 정확한 판단을 하면 되기 때문이다.
Felzenszwalb의 그래프 기반의 segmentation을 수행하고 나면, 자연 영상에서는 많은 영역이 도출이 되는데, 영상이 갖고 있는 계층 구조적인 특성에 맞게 적절한 후보 영역을 찾아내려면, 1단계 sub-segmentation에서 만들어진 여러 영역들을 합치는 과정을 거쳐야 한다. SS에서는 여기에 유사도를 기반으로 한 greedy 알고리즘을 적용하였다.
이번 Class는 SS에서 얻어진 영상을 병합하는 방식과 그들이 사용한 유사도 척도(similarity metric)에 대하여 좀 더 상세하게 살펴볼 예정이다.
Hierarchical Grouping
앞서 설명한 것처럼, Felzenszwalb의 segmentation 방법을 사용하면, 아래 그림처럼 많은 작은 영역들이 만들어지게 된다. 다른 scale에 있는 중요한 객체 후보들을 찾아내려면, 작은 영역들을 병합시키는 작업을 해줘야 한다.
영역 병합에는 아래와 같이 단순한 greedy 알고리즘을 사용한다.
먼저 segmentation을 통하여 초기 영역을 설정해준다. 그 후 인접하는 모든 영역들 간의 유사도를 구한다. 전체 영상에 대하여 유사도를 구한 후, 가장 높은 유사도를 갖는 2개의 영역을 병합시키고, 병합된 영역에 대하여 다시 유사도를 구한다. 새롭게 구해진 영역은 영역 list에 추가를 한다. 이 과정을 전체 영역이 1개의 영역으로 통합될 때까지 반복적으로 수행을 하고, 최종적으로 R-리스트에 들어 있는 영역들에 대한 bounding box를 구하면 된다.
위 과정을 수행하게 되면, 아래 그림처럼, 영역이 통합이 되는 것을 확인할 수 있으며, 다양한 scale에 있는 후보들을 검출할 수 있게 된다.
?
다양화 전략 (Diversification Strategy)
후보 영역 추천의 성능 향상을 위해 SS에서는 다음과 같은 다양화 전략을 사용한다.
다양한 컬러 공간을 사용
color, texture, size, fill 등 4가지 척도를 적용하여 유사도를 구하기
다양한 컬러 공간
다른 컬러 공간은 서로 다른 항상성(invariance)을 보인다. 그러므로 단순하게 RGB나 흑백 컬러 공간을 사용하는 대신에 8개의 서로 다른 컬러 공간을 사용하게 되면 특정 컬러 공간에서는 검출하지 못했던 후보 영역까지 검출이 가능하게 된다.
가령 RGB 컬러 공간을 사용하는 경우는 그림자나 광원의 세기 변화 등으로 인한 조도 변화에 3개의 채널이 모두 영향을 받게 되지만, HSV(색상, 채도, 명도) 컬러 공간을 사용하면 색상이나 채도는 밝기의 변화에 거의 영향을 받지 않지만 명도는 영향을 받는다.
아래의 표는 각각의 컬러 공간이 영향을 받는 수준을 평가한 것이다. 이 표에서 “-“는 영향을 받는 경우이고, “+”는 영향을 받지 않는 경우이며, “+/-“는 부분적으로 영향을 받는 것을 나타낸다. 또한 1/3과 2/3는 3개의 컬러 채널에서 1개와 2개의 채널이 영향을 받지 않는다는 것을 나타낸다.
논문에서는 sub-segmentation과 ?영역을 병합하는 과정에 서로 다른 8개의 컬러 공간을 적용했다고 밝히고 있다.
다양한 유사도 검사의 척도
SS는 유사도 검사의 척도로 color, texture, size, fill을 사용을 하며, 유사도 결과는 모두 [0, 1] 사이의 값을 갖도록 정규화(normalization) 시킨다.
1) 컬러 유사도
컬러 유사도 검사에는 히스토그램을 사용한다. 각각의 컬러 채널에 대하여 bin을 25로 하며, 히스토그램은 정규화 시킨다. 각각의 영역들에 대한 모든 히스토그램을 구한 후 인접한 영역의 히스토그램의 교집합 구하는 방식으로 유사도를 구하며, 식은 아래와 같다.
2) texture 유사도
object matching에서 뛰어난 성능을 보이는 SIFT(Scale Invariant Feature Transform)와 비슷한 방식을 사용하여 히스토그램을 구한다. 8방향의 가우시안 미분값을 구하고 그것을 bin을 10으로 하여 히스토그램을 만든다. SIFT는 128차원의 디스크립터 벡터를 사용하지만, 여기서는 80차원의 디스크립터 벡터를 사용하며, 컬러의 경우는 3개의 채널이 있기 때문에 총 240차원의 벡터가 만들어진다.
컬러 유사도의 히스토그램과 마찬가지로 히스토그램에 대하여 정규화를 수행하며, 영역간의 유사도는 컬러 유사도와 마찬가지 방식을 사용한다.
3) 크기 유사도
작은 영역들을 합쳐서 큰 영역을 만들 때, 다른 유사도만 따지만 1개 영역이 다른 영역들을 차례로 병합을 하면서 영역들의 크기 차이가 나게 된다. 크기 유사도는 작은 영역부터 먼저 합병이 되도록 해주는 역할을 한다. 일종의 가이드 역할을 하게 되는 것이다.
아래 수식을 보면 명확해진다. size(im)은 전체 영역에 있는 픽셀의 수이고, size(ri)와 size(rj)는 유사도를 따지는 영역의 크기(픽셀 수)이다. 영역의 크기가 작을수록 유사도가 높게 나오기 때문에, 다른 유사도가 모두 비슷한 수준이라면 크기가 작은 비슷한 영역부터 먼저 합병이 된다.
4) fill 유사도
2개의 영역을 결합할 때 얼마나 잘 결합이 되는지를 나타낸다. fill이라는 용어가 붙은 이유는 2개의 영역을 합칠 때 gap이 작아지는 방향으로 유도를 하기 위함이다. 가령 1개의 영역이 다른 영역에 완전히 포함이 되어 있는 형태라고 한다면, 그것부터 합병을 시켜야 hole이 만들어지는 것을 피할 수 있게 된다.
아래 그림을 보자.
이 그림에서 r1과 r2를 합쳤을 경우 r1과 r2를 합친 빨간색의 bounding box 영역(BBij)의 크기에서 r1과 r2 영역의 크기를 뺐을 때 작은 값이 나올수록 2 영역의 결합성(fit)이 좋아지게 된다.
fill 유사도는 이런 방향으로 합병을 유도하기 위한 척도라고 보면 된다. 아래 수식을 보면 명확하게 알 수 있다. Fit이 좋을수록 결과적으로 1에 근접한 값을 얻을 수 있기 때문에 그 방향으로 합병이 일어나게 된다.
유사도를 사용하는 방식
이렇게 구한 4개의 유사도를 결합 시켜서 사용을 하며, 수식은 아래와 같다.
위 수식에서 a1~a4는 해당 유사도를 사용할 것인지 말 것인지를 결정한다. 참고로 파이썬 데모 코드를 보면 아래와 같이 편리하게 유사도를 선택하여 쓸 수 있는 메뉴가 있다.
https://github.com/belltailjp/selective_search_py
?
이번 Class에서는 주로 Selective Search에서 hierarchical grouping에서 영역을 결합할 때 무엇을 근거로 하는지 살펴보았다. 다양화를 위해 다양한 컬러 공간을 사용하고, 또한 효과적인 영역 병합을 위해 유사도 척도를 사용하는 것을 확인하였다. 다음 Class는 Selective Search의 마지막회로 이전회에서 다루지 못한 부분에 대하여 살펴볼 예정이다.
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
2. Selective Search [4]
Semantic Segmentation ? “Selective Search (part4)”
[Part VII. Semantic Segmentation] 2. Selective Search [1] ~ 2. Selective Search [3] 을 통하여 Selective Search(이후는 줄여서SS로 표기)에 대하여 상세히 살펴 보았다. J.R.R Uijlings 팀이 발표한 논문 "Selective Search for Object Recognition"은, 나중에 Selective Search라는 이름만 널리 알려졌기 때문에, object가 있을법한 후보 영역을 ?찾는 내용만 있는 것 같지만, 실제로는꽤 좋은 성능을 ?보이는 검출 알고리즘도 소개가 되어 있다.
2013년 ILSVRC에 “UvA(University of Amsterdam) 구조”라는 이름으로 출전을 하여, 검출(detection) 분야에서 1위를 차지 하였고, 분류(classification) 분야에서도 그 이전에 발표된 AlexNet 보다 더 좋은 성능을 보여준다. (ILSVRC 2013 결과는 다음 링크 참고)
http://www.image-net.org/challenges/LSVRC/2013/results.php
ImageNet Large Scale Visual Recognition Competition 2013 (ILSVRC2013)
www.image-net.org
UvA 구조는 SS를 사용하여 후보 영역을 추출한 뒤, 그 후보 영역에 대하여 OpponentSIFT 및 RGB-SIFT와 같은 알고리즘을 사용하여  feature를 추출하고 디스크립터를 만들었으며, 히스토그램 방식의 벡터를 SVM을 통해 검출을 수행하는 방식을 사용하고 있다. 검출이나 분류의 방법은 나중에 딥러닝을 사용한 방법이 대세가 되면서, 또 딥러닝을 사용한 결과가 너무 좋게 되면서 주목에서 사라지게 되었지만, 후보를 검출하는 SS 방법만은 널리 알려지게 된다.
특히 그 이듬해에 발표된 R-CNN(Rich feature hierarchies for accurate object detection)에 SS 부분이 후보 영역을 추천하기 위한 용도로 사용이 되는데, R-CNN이 워낙 뛰어난 성능을 보였기 때문에 같이 주목을 받게 되었고, SPPNet 및 Fast R-CNN에서도 후보 영역 추천에 SS를 사용하였다.
하지만 그 이후 Faster R-CNN 처럼, 후보 영역 검출까지 동일한 CNN 망을 사용하는 구조로 CNN의 흐름이 바뀌면서 SS의 인기가 떨어지게 된다.
이번 Class는 J.R.R. Uijlings 팀이 발표한 논문에서 SS한 결과를 검출에 적용한 방식에 대하여 좀 더 상세하게 살펴볼 예정이다.
SS 방식을 사용한 Object Detection 방식
?
앞서 설명한 것처럼, Uijlings 팀이 SS 방식만 발표한 것이 아니라, object detection 방법도 같이 발표를 하였다. 지금 널리 쓰이는 것처럼, 딥러닝 기반의 방식이 아니라, “feature detection + SVM”이라는 기본 방식을 사용하였다.
하지만 SS의 성능이 매우 뛰어났기 때문에, “feature detection + SVM” 처럼 전통적인 검출 방식을 사용하더라도 2013년 기준으로는 만족할만스러운 결과를 얻을 수 있었다. 잘 알려진 것처럼, 딥러닝이 대세가 되기 전까지 흔히 Learning algorithm으로는 SVM(Support Vector Machine)을 많이 사용하였다.
[비고]  SVM에 대한 설명은 “읽기 쉬운 머신 러닝” section의 SVM 편을 참고?
[머신러닝] 9. 머신러닝 학습 방법(part 4) - SVM(1)
쉽게 읽는 머신 러닝 ? 학습방법 (part 4) - SVM(1) IDC의 자료에 따르면, 2020년 경에는 290...
blog.naver.com
[머신러닝] 10. 머신러닝 학습 방법(part 5) - SVM(2)
쉽게 읽는 머신 러닝 ? 학습방법 (part 5) ? SVM(2) 아마존을 이끌고 있는 ‘제프 베조스’...
blog.naver.com
Uijlings팀이 object detection을 위해 사용한 기본적인 접근법은 아래와 같다.
SIFT 기반의 feature descriptor를 사용한 BoW(Bag of words) 모델
4레벨의 spatial pyramid
분류기로 SVM을 사용
전체적인 구조는 아래 그림과 같고 2단계 과정을 통하여 학습을 수행한다. 그림이 워낙 작기 때문에 뒷부분에서는 작은 영역으로 나누어 살펴볼 예정이다.
?
?
개념 이해: Bag of Words 모델
?
아래 그림들을 보면, 영상은 그 밑에 있는 주요 성분으로 구성이 되어 있는 것을 알 수가 있다.
?
이 주요 성분들의 공간적인 위치를 특별히 따지지 않더라도, 영상을 구성하는 특정한 성분들의 분포를 이용하여 아래 그림의 오른쪽처럼 히스토그램으로  나타낼 수가 있게 된다. 다른 말로 표현을 하면, 특정 성분들의 조합이 object에 따라 확연히 다른 분포를 보인다면, 크기, 회전, 조명 등에 영향을 받는 object를 검출하기 위해 실제 영상을 직접 비교하는 대신에, 성분들의 분포만을 이용해 해당 object 여부를 판별할 수 있게 되는 것이다.
?
여기서 주요 성분은 영상을 구성하는 특징(feature)이라고 볼 수 있으며, 이 특징들의 집합이 바로 코드북(codebook)이 된다. 이 코드북에 있는 각각의 성분들의 크기나 분포가 해당 object를 분류나 검출할 수 있게 하는 수단이 된다. 이런 방식을 BoW(Bag of Words)라고 부르며, 컴퓨터 비전뿐만 아니라 자연어 처리나 스팸 필터링 등에 널리 사용이 되고 있다.
그렇다면 영상을 구성하는 주요 성분인 코드북은 어떻게 만들어낼까? 여기에 많이 사용되는 방식이 SIFT이며, SIFT를 사용하여 디스크립터를 만들어내고 이 디스크립터의 분포를 벡터화 하면 된다.
BoW를 사용하는 경우는 뒷단의 분류기로는 SVM이나 Nearest-neighbor 모델을 많이 사용한다. BoW를 너무 자세하게 설명하는 것은 본 블로그의 범위를 넘어서기 때문에 설명이 아주 잘된 뉴욕대의 자료를 소개한다.
http://cs.nyu.edu/~fergus/teaching/vision_2012/9_BoW.pdf
?
Feature Representation
?
딥러닝이 아닌 전통적인 방식에서는 특징을 잘 추출하고 그것을 잘 나타내는 것이 매우 중요하다. 특징을 추출한 후 대부분 뒷단에서는 SVM을 사용하기 때문에, SVM이 good과 fail을 잘 구별할 수 있도록 양질의 vector를 추출하는 것이 매우 중요하다.
SIFT(Scale Invariant Feature Transform)는 속도는 느리지만 local feature를 이용한 패턴 매칭에는 탁월한 성능을 보이는 알고리즘이다. 2004년 Lowe가 발표한 SIFT는 기본적으로 밝기(intensity) 정보의 gradient 방향성에 대한 히스토그램 특성을 구한 후 그것을 128차원 벡터로 표시한다.
원래의 SIFT는 흑백 영상에만 적용이 가능하기 때문에, 이후에 컬러 영상에 적용할 수 있는 개선된 SIFT 알고리즘이 많이 발표가 되었으며, 대표적인 방식으로는 HSV-SIFT, OpponentSIFT, HueSIFT, W-SIFT, rgSIFT 등이 있다. 이것들에 대한 특성 비교는 다음 논문을 참고하면 좋을 것 같다. 참고로 아래 논문 역시 본 논문의 저자팀이 발표한 것이다. (“Evaluation of Color Descriptors for Object and Scene Recognition”)
본 논문에서는 SIFT와 2개의 color SIFT (OpponentSIFT와 RGB-SIFT)를 사용하였다. BoW 코드북의 크기는 4000이고, 다양한 크기에 대응이 가능할 수 있도록 4-레벨(1x1, 2x2, 3x3, 4x4)의 피라미드를 적용하여 길이가 360,000개인 벡터를 만들어 냈다.
Traing 과정
?
“BoW + SVM” 방식은 2단계 학습 과정을 거친다. SVM 학습의 결과가 효과적이려면, true와 false가 확실한 벡터보다는 경계면 근처에 있는 벡터들을 잘 활용을 해줘야 한다. (앞서 소개한 SVM 자료 참고)
1. 초기 학습
초기 학습에는 이미 알고 있는 ground truth 데이터로부터 “positive example”을 구하고, 추천된 후보 영역 중 ground truth 데이터와의 overlap이 20~50% 수준에 있는 경계 근처의 데이터를 “negative example”로 정하여 SVM을 학습시킨다.
앞서 살펴본 그림 중 왼쪽 부분에 해당하는 영역이다.
?
2. 학습 결과 튜닝
위와 같이 초기 학습을 마친 뒤에는 SVM 학습을 튜닝시키는 과정을 거치게 되는데, 이 때 필요한 것이 false positive(제대로 검출하지 못한 것)이다. False positive를 찾아내고 추가로 negative example을 더하여 SVM을 더 정교하게 튜닝하는 과정을 거친다. 이에 해당하는 부분은 원래 전체 구조 그림의 오른쪽에 해당이 된다.
?
?
?
실험 결과
?
논문에서 확인할 수 있는 것처럼, 결과가 상당히 좋다는 것을 확인할 수 있다. 먼저 실험 결과를 확인하는 척도로서 사용하는 것이 ABO(Average Best Overlap)인데 이것은 알고 있는 ground data와의 최고의 overlap에 대한 평균을 구한 것이다.
?
아래 그림에서 녹색선은 ground truth이고, 빨간색은 이 알고리즘의 결과이다. ABO의 수치가 높을수록 결과가 좋다고 볼 수 있다.
?예측 가능하듯이, 다양한 전략을 섞어 사용하는 경우에 결과가 좋게 나오는 것을 확인할 수 있다. Fast는 연산 시간에 중점을 둔 버전이고, Quality는 성능에 집중한 버전으로 약 4배 이상의 연산 시간차가 나는 것을 확인할 수 있다.
?
앞서 발표된 다른 논문들과의 비교 실험에서도 SS가 뛰어난 결과를 보이는 것을 알 수 있다.
?
총 4회에 걸쳐 Selective Search에 대하여 상세하게 살펴봤다. R-CNN, SPPNet, Fast/Faster R-CNN에서 질문이 많았던 것을 고려하여 좀 자세한 설명을 하는 편이 좋다고 생각하였기 때문이다. 원래는 Selective Search를 마치고 위 검출 알고리즘을 살펴보는 것이 올바른 순서이겠지만, 이미 위 4가지 방법은 앞선 Class에서 살펴보았기 때문에 추가 설명은 생략한다.
?
다음 Class부터는 본격적으로 딥러닝에 기반한 Semantic Segmentation 기법들을 살펴보고자 한다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
3. FCN [1]
Semantic Segmentation ? “FCN(Part1)”
?
FCN은 Fully Convolutional Network의 약어로 2015년에 발표가 되었으며, 논문의 제목은 “Fully Convolutional Networks for Semantic Segmentation”이다. 별다른 어려운 기법을 사용하지 않았음에도 불구하고 semantic segmentation에서 워낙 뛰어난 성능을 보였기 때문에 불과 2년도 안 되는 기간 동안에 약 1300회 정도나 인용이 되었다.
전통적으로 버클리대학교는 영상 detection 분야에서 뛰어난 논문을 많이 발표하였다. 이 논문을 발표한 Jonathan Long도 버클리 출신이며, R-CNN/Fast R-CNN이라는 뛰어난 논문을 발표한 Ross Girshick 역시 버클리 출신이다.
?
?
Classification, Detection, Semantic Segmentation
?
Classification이란 영상에서 특정 대상이 있는지를 확인하는 기술을 말하며, 가령 아래와 같은 그림이 있다면, “tabby cat”이 나오게 된다. 통상은 여러단의 (convolution + pooling)으로 구성된 네트워크 뒤에 fully connected layer가 오는 형태가 일반적이다.
앞선 class에서 살펴보았던 AlexNet, GoogLeNet, VGGNet 등 거의 대부분의 유명한 신경망들이 비슷한 구조를 갖는다. Fully connected layer를 거치고 나면 위치나 공간에 관련된 정보는 모두 사라지게 된다.
Detection은 Classification과 달리 특정 대상이 있는지 여부만을 가리는 것이 아니라, 위치 정보를 포함한다. 보통은 bounding box라고 부르는 사각형의 영역을 통해 대상의 위치 정보까지 포함하게 된다.
앞서 살펴본 R-CNN, SPPNet, Fast R-CNN, Faster R-CNN 등이 모두 여기에 속한다. 아래 그림은 Fast R-CNN의 구조이며, Detection은 대상의 여부뿐만 아니라, 위치 정보를 포함해야 하기 때문에 아래 그림의 오른쪽처럼, class 여부를 가리는 softmax 부분과 위치 정보를 구하는 bbox regressor로 구성이 된다.
?
Semantic Segmentation이란 단순하게 bounding box 등을 이용하여 검출을 원하는 대상을 나타내는 것이 아니라, 보통은 픽셀 단위의 예측을 수행하여 의미 있는 단위로 대상을 분리해낸다.
아래 그림은 개와 고양이가 있는데, 오른쪽 그림처럼 개와 고양이를 픽셀 단위로 구별해는 기술이 바로 semantic segmentation 기술이다.
?
Semantic segmentation은 영상속에 무엇(what)이 있는지를 확인하는 것(semantic)뿐만 아니라 어느 위치(where)에 있는지(location)까지 정확하게 파악을 해줘야 한다. 하지만 semantic과 location은 그 성질상 지향하는 바가 다르기 때문에 이것을 조화롭게 해결해야 semantic segmentation의 성능이 올라가게 된다.
그 동안 semantic segmentation을 위한 많은 논문들이 발표가 되었고, 어려운 개념까지 적용된 기법들이 있었지만, 그 결과가 아주 만족스럽지는 못했다. FCN이 발표가 되면서 크게 성능 향상을 이룩할 수 있게 되었고, 이후 FCN에 자극을 받은 많은 논문들이 나오게 된다.
?
?
FCN과 fully convolutional model
?
FCN이 주목한 부분은 classification에서 성능을 검증 받은 좋은 네트워크 (AlexNet, VGGNet, GoogLeNet) 등을 이용하는 것이다. 이들 대부분의 classification을 위한 네크워크는 뒷단에 분류를 위한 fully connected layer가 오는데, 이 fully connected layer가 고정된 크기의 입력만을 받아들이는 문제가 있다.
또 한가지 결정적인 문제는 fully connected layer를 거치고 나면, 위치 정보가 사라지는 것이다. Segmentation에 사용하려면 위치정보를 알 수 있어야 하는데 불가능하기 때문에 심각한 문제가 된다.
하지만, FCN 개발자들이 주목한 부분은 fully connected layer를 1x1 convolution으로 볼 수 있다는 점이다. 이것은 이미 [Part V. Best CNN Architecture] 7. OverFeat 에서 OverFeat 구조를 설명하는 부분에 자세하게 설명을 하였으니, 거기를 참고하면 된다.
사실 OverFeat 개발자들이 fully convolutional network 개념을 먼저 적용을 하였지만, 이들은 ?개념을 classification과 detection에만 활용을 하였다.
Fully connected layer를 1x1 convolution으로 간주하면 (이것을 FCN 개발자들은 convolutionization 이라고 불렀음) 위치 정보가 사라지는 것이 아니라  남게 된다.
아래 그림은 AlexNet의 뒷단에 사용되던 3개의 fully connected layer를 1x1 convolution으로 간주한 경우를 보여주며, 위치 정보가 남아 있기 때문에 오른쪽의 heatmap 그림에서 알 수 있는 것처럼, 고양이에 해당하는 위치?의 score 값들이 높은 값으로 나오게 된다.
?
또한 이제는 모든 network가 convolutional network으로 구성이 되기 때문에 더 이상 입력 이미지의 크기 제한을 받지 않게 된다.
요약하면, fully connected layer를 1x1 convolution으로 간주함에 따라 위치 정보(공간 정보)를 유지할 수 있게 되었고, 전부가 convolutional network으로 구성이 되기 때문에 입력 영상의 제한을 받지 않게 되었다.
또한 patch 단위로 영상을 처리하는 것이 아니라, 전체 영상을 한꺼번에 처리를 할 수 있어서 겹치는 부분에 대한 연산 절감 효과를 얻을 수 있게 되어, 속도가 빨라지게 된다. 이것은 R-CNN의 속도를 개선하기 위해 Fast R-CNN이나 Faster R-CNN이 convolution 부분을 ?딱 한번만 계산하도록 하여 연산 시간을 절감시킨 것과 유사하다. ([Part V. Best CNN Architecture] 8. ResNet [4] Fast-RCNN, 8. ResNet [5], Faster-RCNN 참고)
?
?
Upsampling (Decovolution)
?
그런데 여러 단계의 (convolution + pooling)을 거치게 되면, feature-map의 크기가 줄어들게 된다. 픽셀 단위로 예측을 하려면, 줄어든 feature-map의 결과를 아래 그림처럼, 다시 키우는 과정을 거쳐야 한다.
?
1x1 convolution을 거치면서 얻어진 score 값을 원 영상의 크기로 확대하는 가장 간단한 방법은 bilinear interpolation을 사용하면 된다.
하지만 end-to-end 학습의 관점에서는 고정된 값을 사용하는 것이 아니라 학습을 통해서 결정하는 편이 좋다. 논문에서는 backward convolution, 즉 deconvolution을 사용하며, deconvolution에 사용하는 필터의 계수는 학습을 통해서 결정이 되도록 하였다. 이렇게 되면, 경우에 따라서 bilinear 한 필터를 학습할 수도 있고 non-linear upsampling도 가능하게 된다.
단순하게 score를 upsampling하게 되면, 어느 정도 이상의 성능을 기대하기가 어렵다. 그래서 FCN 개발자들은 아래 그림과 같이skip layer 개념을 활용하여 성능을 끌어올렸다. 이 부분은 다음 Class에서 자세하게 다룰 예정이다.
Skip layer의 기본 개념은 (convolution + pooling)의 과정을 여러 단계를 거치면서 feature-map의 크기가 너무 작아지면 detail한 부분이 많이 사라지게 되기 때문에 최종 과정보다 앞선 결과를 사용하여 detail을 보강하자는 것이다.
여러 단계의 결과를 합쳐주는 과정을 거치면 아래 그림과 같이 더 정교한 예측이 가능해지게 된다. stride가32인 경우는 자세하게 구별할 수 없지만, skip layer를 활용한 stride 8에서는 꽤 정교한 예측이 가능하게 된다.
?
?
결과
아래 그림을 통해 FCN의 성능을 확인할 수 있다. 왼쪽에 있는 Image에 대하여 Ground Truth 값과 FCN을 통해서 얻을 결과를 비교하면, 꽤 괜찮은 성능을 얻을 수 있다는 것을 확인할 수 있다.
이번 Class에서는 FCN의 기본 개념에 대하여 살펴보았다. Classification 망의 뒷부분에 있는 fully connected layer를 1x1 convolution으로 간주하는 기본 아이디어를 기반으로 속도 및 성능을 얻을 수 있게 되었다. 다음 Class부터 FCN의 주요 개념들을 보다 상세하게 살펴보고자 한다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
3. FCN [2]
Semantic Segmentation ? "FCN (part2)"
?
[Part VII. Semantic Segmentation] 1. Image Segmentation 기본이론 [1] ~ 2. Selective Search [4] 까지 살펴본 것처럼, 일반적인 image segmentation은 입력으로 영상 데이터를 받아들여, 영역이나 특정 구조로 구분된 출력을 내는 것을 말하며, 구분하는 기준으로는 영상에 존재하는 에지나 컬러 속성 등을 ?흔히 사용한다. 아래 그림처럼, 의미 있는 속성으로 구별할 뿐이지 그것이 무엇인지 파악을 할 필요까지는 없다.
하지만 Semantic segmentation은 픽셀 단위까지 밀도 있게 예측(dense prediction) 하여야 하며, 아래 그림처럼 영상 속에 무엇이 포함 되어 있는지 파악을 해내야 하기 때문에 훨씬 더 어렵다고 볼 수 있다. 출력으론 (제한된 숫자의) object class가 포함된 픽셀 단위의 영역이 나온다.
Semantic segmentation은 로봇 비전이나 로봇이 장면을 이해하는데 사용될 수도 있고, 자율 주행이나 효율적인 진단을 위한 의료 영상에도 적용이 가능하며, 궁극적으로는 장면 이해(scene understanding)을 위한 기반 기술이 된다.
딥러닝 기술을 이용한 영상 분류 기술이 이미 인간을 수준을 넘어서기는 하였지만, 아직은 1000여개 수준의 제한된 class 범위 내에서 이루어진 것이고, object detection이나 semantic segmentation 기술 역시 제한된 범위 내에서 된 것이기 때문에 아직은 갈 길이 멀기는 하지만, 최근 몇 년 사이에 비약적인 발전이 있었기 때문에 앞으로  얼마나 더 많은 발전이 있을지 기대가 된다.
지난 [Part VII. Semantic Segmentation] 3. FCN [1] 에서는 FCN(Fully Convolutional Network)의 기본적인 개념에 대하여 살펴보았다. 이번 Class에서는 좀 더 세부적인 부분에 대하여 살펴볼 예정이다.
?
?
?
Classification 모델을 이용해 semantic segmentation을 구현
이미 이 부분은 [Part VII. Semantic Segmentation] 3. FCN [1] 에서 상세하게 설명하였다. AlexNet, VGGNet, GoogLeNet처럼 잘 알려진 classification network에서 뒷부분에 위치하는 fully connected layer를 1x1 convolution으로 간주하면, 망 전체를 convolution network으로 구성할 수 있게 된다.
FCN의 저자인 Jonathan Long의 slide에 좋은 그림이 있어, 그 자료를 바탕으로 설명한다. 아래 그림처럼, classification network(AlexNet을 간략하게 표현)이 있다.
?
위 classification network은 fully connected layer 특성 문제로 인해 고정된 크기의 입력만을 받아들여야 하는데, 뒷부분에 있는 fully connected layer를 1x1 convolution으로 생각하면 아래 그림처럼 표현이 가능해진다.
?
?
?
이제는 망 전체가 convolution layer로 구성이 되었기 때문에 더 이상 입력 이미지 크기에 제한을 받지 않으므로 위 network은 ?더 이상 특정 영상 크기에 연연하지 않게 되었기 때문에, 아래처럼 볼 수 있게 된다.
?
더 이상 입력 영상의 크기에 대한 제한이 없으며, 각 layer를 거치면서 feature map의 크기가 줄어들게 된다. 즉, 맨 마지막에 있는 feature-map의 1 pixel 값은 원영상에서 32x32 크기에 해당하게 된다.
?
Semantic segmentation에서는 픽셀 단위로 조밀하게 예측(dense prediction)을 해줘야 하는데 맨 마지막 layer에서는 feature-map의 크기가 매우 작아졌기 때문에 원영상 크기로 복원하는 작업이 필요하다.
논문에서는 이것을 skip connection 이라는 방법을 이용하여 해결하였다.
?
?
?
?
Dense prediction
?
여러 단계의 (conv + pooling)을 거치면서 Feature-map의 크기가 원영상의 1/32 크기로 줄어들었는데, 픽셀 단위로 조밀한 예측하려면 다시 원영상 크기로 복원을 하는 과정을 거쳐야 한다.
논문에서는 shift-and-stitch 방식을 이용하는 것도 검토를 하였지만, upsampling을 사용하는 것이 효과적이라는 결론을 내리고 shift-and-stitch 방식을 사용하지 않았다. 하지만 2016년에 발표된 Fisher Yu의 논문 "Multi-scale context aggregation by dilated convolution"을 보면 효과적이라는 것을 (물론 그 논문에서는 dilated convolution 뿐만 아니라 성능을 올리기 위해 다른 trick도 적용) 입증하였고, Caffe나 TensorFlow에도 dilated convolution이 지원이 된다.
(Dilated convolution은 차후 Class에서 자세하게 살펴볼 예정이다.)
?
FCN에서는 1/32 크기에서의 feature(즉 score)만을 사용하는 것이 아니라, 1/16과 1/8에서의 값도 같이 사용하는 방식을 취하였다.
이것을 논문에서는 "deep jet"이라고 칭하였다. 이전 layer는 마지막 layer보다 세밀한 feature를 갖고 있기 때문에 이것을 합하면 보다 정교한 예측이 가능해진다.
?
1/16과 1/8 크기 정보를 이용하려면, 이후 (conv + pool) 단계를 거치지 않으면 된다. 즉, 건너 뛰면 되기 때문에 "skip layer" 혹은 "skip connection"이라고 부르며, 아래 그림과 같은 형태가 된다.
?
?
?
1/32에서 32배만큼 upsample한 결과를 FCN-32s라고 하며, FCN-16s는 아래 그림과 같이 pool5의 결과를 2배 upsample한 것과 pool4의 결과를 합치고 다시 그 결과를 16배 upsample하면 되고, FCN-8s는 FCN-16s의 중간 결과를 2배 upsample한 결과와 pool3에서의 예측을 합친 것을 8배 upsample 하는 식이다.
?
?
위 그림이 좀 복잡하기도 하고, 실제로 FCN의 결과는 주로 FCN-8s의 결과를 사용하기 때문에, FCN-8s만을 따로 표현을 해보면 아래와 같은 단순한 형태의 그림이 된다.
?
이처럼, skip과 중간 결과를 합치는 과정을 거치게 되면 아래 그림처럼 점점 더 정교한 예측이 가능하게 된다.
?
실제로 PASCAL VOC 2011 데이터를 이용하여 실험한 결과는 아래 표와 같다. FCN-32s 결과와 FCN-8s의 결과는 정밀가 개선되는 것을 확인할 수 있다.
?
이번 Class에서는 FCN의 dense prediction 방법 및 classification을 위한 망이 어떻게 fully convolution network로 변하는지에 대하여 살펴보았다. FCN은 복잡하지는 않지만, 생각의 전환을 통하여 매우 효율적인 방법으로 semantic segmentation으로 가는 길을 열었다.
?
물론 FCN의 결과를 그대로 사용하는 대신에 "FCN + CRF(Conditional Random Field)"를 사용하여 FCN의 결과를 좀 더 정밀하게 튜닝하는 방법을 사용하면, 아래 그림처럼 좀 더 정교한 예측도 가능해진다. (이것은 추후에 자세하게 다루기로 한다)
?
FCN의 단순하면서도 뛰어난 성능은 다른 연구자들에게 훌륭한 자극이 되었고, 이후에 FCN의 결과를 개선하는 논문들이 많이 나온다. 다음 Class에서는 Fisher Yu의 "dilated convolution"에 대하여 상세하게 살펴보고자 한다.
​
​
[Machine Learning Academy_Part Ⅶ. Semantic Segmentation]
4. Deconvolutional Network
?
Semantic Segmentation ? "Deconvolutional Network"
?
우리는 이미 ZFNet ([Part V. Best CNN Architecture] 4. ZFNet [1] ~ 4. ZFNet [3]) 개발자들의 중요한 업적 중 하나인 visualization 기법을 통하여, 여러 단의 convolutional layer를 통하여 무엇을(local feature에서 global feature까지) 얻을 수 있는 것인지 확인하였다.
또한 여기에서 한걸음 더 나아간 FCN([Part VII. Semantic Segmentation] 3. FCN [1] ~ 3. FCN [2]) 개발자들의 연구를 통하여, 여러 단의 convolutional layer중 후반부에 있는 convolutional feature들을 결합하면semantic segmentation에 필요한 중요한 정보를 ?얻을 수 있기 때문에 "conv+pooling"으로 구성된 CNN 망이 classification/detection뿐만 아니라 segmentation에도 유용함을 확인하였다.
FCN 개발자들이 발표한 fully convolutional network의 개념은 다른 많은 연구자들에게 큰 자극이 되었으며, 그 중 Fisher Yu가 발표한 "Multi-scale context aggregation by dilated convolution"은 FCN에 대한 분석 및 약간의 구조 변경을 통하여FCN의 성능을 좀 더 끌어올렸으며, 일명 "dilated convolution"이라고도 불린다. 원래는 이번 Class에서 살펴볼까 했는데, 논문의 발표시점을 고려하여, 다른 논문의 방식을 먼저 검토한 뒤 다음 Class에서 살펴볼 예정이다.
FCN이 픽셀 단위로 조밀한 예측을 하기는 하지만 detail에 취약한 점이 있는데, 이러한 FCN의 문제점을 개선한 논문 중 우리 나라 연구원들이 발표한 것이 있다. 이것은 바로2015년 포항공대의 노현우씨 팀이 발표한 "Learning deconvolutional network for semantic segmentation" 라는 논문인데, 이들은 FCN에서 "conv+pooling"을 거치면서 해상도가 작아지는 문제를 반복적인 up-convolution(deconvolution)을 통해 해결을 시도하였다. 짧은 시간에 302회나 인용이 되었으니, 꽤 훌륭한 논문이라고 볼 수 있다.
?
이번 Class에서는 노현우씨 등이 발표한 논문 및 presentation 자료를 바탕으로 deconvolutional network에 대하여 살펴볼 예정이다.
FCN의 문제점
?
앞서 살펴본 것처럼, FCN은 classification용으로 충분히 검증을 받은 망을 픽셀 수준의 조밀한 prediction이 가능한 semantic segmentation에 적용하기 위해 up-sampling 로직을 추가하였고, skip layer 개념을 적용하여 떨어지는 해상도를 보강하였다. 뛰어난 성능으로 개념 증명에 성공을 하였지만, 맨 처음 새로운 시도를 적용하였기에 여러 모로 개선 가능한 포인트들이 존재한다.
?
FCN에 존재하는 한계나 제한점은 다음과 같다.
사전에 미리 정한 receptive field를 사용하기 때문에 너무 작은 object가 무시되거나 엉뚱하게 인식될 수 있으며, 큰 물체를 여러 개의 작은 물체로 인식하거나 일관되지 않은 결과가 나올 수도 있다. (아래 논문에서 사용한 그림을 참고)
여러 단의 "conv+pooling"을 거치면서 해상도가 줄어들고, 줄어든 해상도를 다시 upsampling을 통해 복원하는 방식을 사용하기 때문에, detail이 사라지거나 과도하게 smoothing 효과에 의해 결과가 아주 정밀하지 못하다.
?
?
?
FCN의 문제점 극복을 위한 시도 ? 새로운 architecture
?
FCN에서는 픽셀 단위의 조밀한 예측을 위해 upsampling과 여러 개의 후반부 layer의 conv feature를 합치는 방식을 사용하였는데, 이렇게 되면 앞서 살펴본 것과 같은 문제점이 발생하게 된다.
그래서 이들은 convolutional network에 대칭이 되는 deconvolutional network을 추가하였으며, 이를 통해 upsampling 해상도의 문제를 해결하려고 하였다.
?
?
?
?
이들이 사용한 기본 망은 VGG16을 기반으로 하였다. Deconvolutioal Network의 개념은 ZFNet 개발자들이 내부 layer의 feature를 시각화(visualization)하는 작업을 할 때 개념이 소개가 되었으며, max-pooling의 위치를 기억하여 정 위치를 찾아가기 위한 switch variable 개념을 비슷하게 사용하였다. ([Part V. Best CNN Architecture] 4. ZFNet [1] ~ 4. ZFNet [3] 참고)
?
?
단순하게 bilinear interpolation을 이용한 upsampling이 아니라 ?여러 layer의 "unpooling+deconvolution"에 기반하기 때문에 훨씬 정교한 복원이 가능해진다. Deconvolutional layer의 filter의 계수 역시 학습을 통해서 결정이 되며, 계층 구조를 갖는 deconvolution layer를 통해 다양한 scale의 detail 정보를 살릴 수 있게 된다.
아래 그림은 논문에서 제시하고 있는 decovolutional network의 각 layer에서 얻어지는 activation을 시각화 시킨 그림인데, deconvolutional network의 후반부로 갈수록 좀 더 정밀한 결과가 얻어지는 것을 확인할 수 있고, 배경 부분에 있던 noise 성 activation이 후반부로 갈수록 사라지는 것을 확인할 수 있다.
아래 그림에서 (a)는 학습에 사용한 이미지이고, (b)는 마지막14x14 deconv layer, (c)는 28x28 unpooling layer에서의 activation을 보여준다.
?
(d)는 마지막 28x28 deconv layer, (e)는 56x56 unpooling layer에서의 activation이다. 좀 더 세밀해지는 것을 확인할 수 있다.
(f)는 마지막 56x56 deconv layer, (g)는 112x112 unpooling layer에서의 activation이다. 역시 좀 더 세밀해지는 것을 확인할 수 있다.
?
(h)는 마지막 112x112 deconv layer, (i)는 224x224 unpooling layer, (j)는 마지막 224x224 deconv layer에서의 activation이다. 매우 조밀한 예측이 되는 것을 확인할 수 있다.
논문 저자들의 주장처럼, 단순한 decovolution이나 upsampling을 사용하는 대신에 coarse-to-fine deconvolution 망을 구성함으로써 보다 정교한 예측이 가능함을 확인할 수 있다. Unpooling을 통해 가장 강한 activation을 보이는 위치를 정확하게 복원함에 따라 특정 개체의 더 특화된(논문의 표현은 exampling-specific) 구조를 얻어낼 수 있고, deconvolution을 통해 개체의 class에 특화된 (class-specific) 구조를 추출해 낼 수 있게 된다.
아래 그림은 FCN 방법과 decovolutional network 방법을 비교한 결과이다. 확실히 좀 더 정밀한 segmentation이 가능하다는 것을 확인할 수 있다.
하지만 얻는 것이 있으면 잃는 것도 있어야 하는 것처럼, 파라미터의 갯수가 많은 VGG-16에 기반하고 있기 때문에 연산량 측면의 부담은 피할 길이 없어 보인다. Presentation 정보에 따르면, PASCAL VOC 2012 데이터를 data augmentation을 통해 약 300만장의 이미지를 만들었으며, Titan X GPU를 사용하여 학습에 6일이 걸렸다고 한다.
?
2단계 학습법과 Batch Normalization
?
앞서 망의 구조를 살펴본 것처럼, VGG-16 망 2개를 쌓아놓은 것이나 마찬가지이기 때문에 망의 깊이가 매우 깊어져 overfitting이 발생하여 학습이 어렵거나 최적의 학습 결과를 얻기가 어려워질 수 있다.
이처럼 깊은 망에서 생기는 overfitting 문제를 피하기 위하여 모든 convolutional layer 및 deconvolutional layer의 출력에batch normalization을 적용하였으며, 이것은 학습의 최적화에 절대적인 영향을 끼쳤다고 한다.
(BN은 [Part VI. CNN 핵심 요소 기술] 1. Batch Nomalization [1] ~ 1. Batch Nomalization [2] 참고)
또한 효율적인 학습을 위해 2단계의 학습법을 사용하였다.
?
1단계는 먼저 쉬운 이미지를 이용하여 학습을 시키는데, 가급적이면 학습을 시킬 대상이 영상의 가운데에 위치할 수 있도록 하였고, 크기도 대체적으로 변화가 작도록 설정을 하였다. 1차 학습은 pre-training의 용도로 생각할 수 있으며, data augmentation을 거친 약 20만장의 이미지를 사용하였다.
?
2단계는 270만장의 이미지를 사용하였으며, 좀 더 다양한 경우에 대응할 수 있도록 다양한 크기 및 위치에 대응이 가능할 수 있도록 하였다.
?
결과 검토 및 개선
?
구조적 특성으로 인해, deconvolutional network은 좀 더 정밀한 segmentation에서 좋은 특징을 보이고, FCN은 전체적인 형태를 추출하는 것에 적합하기 때문에 이 둘을 섞어 사용을 하면 더 좋은 결과를 얻을 수 있다.
?
그래서 FCN과 Deconvolutioal network의 결과의 평균을 구하거나, 이 결과에 추가적으로 CRF(Conditional Random Field)를 적용하여 좀 더 결과를 개선할 수 있다.
?
아래 그림은 FCN 보다 Deconvolutional network의 결과가 좋은 경우이다. 확실하게 detail 쪽 성능은 FCN보다 좋을 것을 확인할 수 있다. 아래 그림에서 EDconvNet은 FCN과 Deconvolutional network의 결과를 결합(Ensemble)한 것이며, EDconvNet+CRF는 EDConvNet의 결과에 CRF를 적용한 것이다.
다음 그림은 Deconvolutional network의 결과가 FCN보다 나쁜 경우를 보여주는 것이다. 확실히 FCN이 전체적인 모양에 관심을 갖는 것이라면, Deconvolutional network은 detail에 집중한다는 것을 확인할 수 있고 이런 경우라면 여러 방법을 섞어 쓰는 것이 효율적임을 알 수 있다.
?
다음 그림은 FCN과 Deconvolutional network의 결과가 모두 좋지 못한데 모델 결합을 하면 결과가 개선되는 경우를 보여주는 예이다.
?
이번 Class에서는 FCN보다 정교한 예측을 위한 deconvolutional network에 대하여 살펴보았다. 정교함에서는 deconvolutional network이 좋은 결과를 내지만 때로는 좋지 못한 결과도 나오기 때문에 뭐가 절대적인 방법이라고 할 수 없다.
?
FCN이 발표된 이래로 FCN의 문제점을 개선하는 많은 논문이 나오는데 그 중 앞서 이야기한 dilated convolution을 다음 Class에서 살펴볼 예정이다.
​
​
[Part Ⅶ. Semantic Segmentation]
5. Dilated Convolution
Semantic Segmentation ? “Dilated Convolution”
FCN 개발자들이 발표한 fully convolutional network의 개념은 다른 많은 연구자들에게 큰 자극이 되었으며, 그 중 Fisher Yu가 발표한 “Multi-scale context aggregation by dilated convolution”은 FCN에 대한 분석 및 약간의 구조 변경을 통하여FCN의 성능을 좀 더 끌어올렸으며, 일명 “dilated convolution”이라고도 불린다.
이번 Class에서는 Fisher Yu의 dilated convolution에 대하여 살펴볼 예정이다.
여기서는 dilated convolution이라는 단어를 Fisher Yu의 segmentation 방법을 지칭하는 것과 실제 dilation 개념이 적용된 convolution과 섞어서 부르고 있음을 알린다.
?
Dilated Convolution 이란?
?
Dilated convolution이라는 말은 Fisher Yu가 처음 언급한 말은 아니며, FCN을 발표한 Jonathan Long의 논문에서 잠깐 언급이 있었지만, FCN 개발자들은 dilated convolution 대신에 skip layer와 upsampling 개념을 사용하였다.
?
그 후 DeepLab 개발팀인 Liang-Chieh Chen의 논문 “Semantic image segmentation with deep convolutional nets and fully connected CRFs”에서 dilated convolution이 나오지만, Fisher Yu 팀과는 조금 다른 방법으로 사용을 하였다. DeepLab 팀의 방법은 다음 Class에서 살펴 볼 예정이다.
?
Dilated convolution의 개념은 wavelet decomposition 알고리즘에서 "atrous algorithm"이라는 이름으로 사용이 되었으며, DeepLab 팀은 구별하여 부르기 위해 atrous convolution이라고 불렀는데, Fisher Yu는 이것을 dilated convolution이라고 불렀으며, 지금은 주로 이 이름으로 불리는 것 같다.
(참고, atrous는 프랑스어이며, 정확히 쓰면 a trous이고, a trous는 구멍(hole) 이라는 뜻임. 그러나 a와 trous를 붙여쓰는 논문이 많아서, 여기서는 그냥 붙여썼음)
?
Dilated convolution이란 아래 그림과 같이, 기본적인 convolution과 유사하지만 빨간색 점의 위치에 있는 픽셀들만 이용하여 convolution을 수행하는 것이 다르다. 이렇게 사용하는 이유는 해상도의 손실 없이receptive field의 크기를 확장할 수 있기 때문이다. atrous convolution이라고도 불리는 이유는 전체 receptive field에서 빨간색 점의 위치만 계수가 존재하고 나머지는 모두 0으로 채워지기 때문이다.
아래 그림에서 (a)는 1-dilated convolution이라고 부르는데, 이것은 우리가 기존에 흔히 알고 있던 convolution과 동일하다. (b)는 2-dilated convolution이라고 부르며, (b)의 빨간색 위치에 있는 점들만 convolution 연산에 사용하며, 나머지는 0으로 채운다. 이렇게 되면 receptive field의 크기가 7x7 영역으로 커지는 셈이 된다. (c)는 4-dilated convolution이며, receptive field의 크기는 15x15로 커지게 된다.
Dilated convolution을 사용하여 얻을 수 있는 이점은 큰 receptive field를 취하려면, 파라미터의 개수가 많아야 하지만, dilated convolution을 사용하면 receptive field는 커지지만 파라미터의 개수는 늘어나지 않기 때문에 연산량 관점에서 탁월한 효과를 얻을 수 있다.
위 그림의 (b)에서 receptive field는 7x7이기 때문에 normal filter로 구현을 한다면 필터의 파라미터의 개수는 49개가 필요하며, convolution이 CNN에서 가장 높은 연산량을 차지한다는 것을 고려하면 상당한 부담으로 작용한다. 하지만 dilated convolution을 사용하면 49개중 빨간점에 해당하는 부분에만 파라미터가 있는 것이나 마찬가지고 나머지 40개는 모두 0으로 채워지기 때문에 연산량 부담이 3x3 filter를 처리하는 것과 같다.
Dilated convolution을 하면 뭐가 좋지?
일단은 receptive field의 크기가 커진다는 점이며, dilation 계수를 조정하면 다양한 scale에 대한 대응이 가능해진다. 다양한 scale에서의 정보를 끄집어내려면 넓은 receptive field를 볼 수 있어야 하는데 dilated convolution을 사용하면 별 어려움이 없이 이것이 가능해진다.
?
우리가 기존에 살펴본 일반적인 CNN 망에서는 receptive field 확장을 위해 pooling layer를 통해 크기를 줄인 후 convolution을 수행하는 방식을 취했다. 기본적으로 pooling을 통해 크기가 줄었기 때문에 동일한 크기의 filter를 사용하더라도 CNN 망의 뒷단으로 갈수록 넓은 receptive field를 커버할 수 있게 된다.
Fisher Yu의 논문에서는 자세하게 설명을 하지 않았기 때문에, 아래에 있는 DeepLab 논문에 있는 그림을 참조하면 훨씬 이해가 쉬워진다.
위쪽은 앞서 설명한 것처럼 down-sampling 수단이 적용이 된 경우이며, 픽셀 단위 예측을 위해 up-sampling을 통해 영상의 크기를 키운 경우이며, 아래는 dilated convolution(atrous convolution)을 통하여 얻은 결과이다.
Fisher Yu는 Context module이라는 것을 개발하여 segmentation의 성능을 끌어 올렸으며, 여기에 dilated convolution을 적용하였다.
Front-end 모듈
FCN이 VGG-16 classification을 거의 그대로 사용한 반면에 Fisher Yu 팀은 성능 분석을 통해 조금 수정을 하였다.
이 팀의 분석 결과에 따르면, FCN 팀은 VGG-16 network의 뒷단을 그대로 사용을 하였지만, 오히려 크게 도움이 되지 않기 때문에 뒷부분을 아래 그림과 같이 수정을 하였다.
먼저 pool4와 pool5는 제거하였다. FCN에서는 pool4와 pool5를 그대로 두었기 때문에 feature-map의 크기가 1/32까지 작아지고 그런 이유로 인해, 좀 더 해상도가 높은 pool4와 pool3 결과를 사용하기 위해 skip layer라는 것을 두었다. ([Part VII. Semantic Segmentation] 3. FCN [1] ~ 3. FCN [2] 참고)
하지만 Fisher Yu는 pool4와 pool5를 제거함으로써 최종 feature-map의 크기는 원영상의1/32이 아니라 1/8수준까지만 작아졌기 때문에 upsampling을 통해 원영상 크기로 크게 만들더라도 상당한 detail이 살아 있게 된다.
또한 conv5와 conv6(fc6)에는 일반적인 convolution을 사용하는 대신에 conv5에는 2-dilated convolution을 적용하고, conv6에는 4-dilated convolution을 적용하였다..
결과적으로 skip layer도 없고 망도 더 간단해졌기 때문에, 연산 측면에서는 훨씬 가벼워졌다. 아래 표와 그림을 보면, Front-end만 수정하여도 이전 결과들 보다 정밀도가 좋아지는 것을 확인할 수 있다.
아래 표와 그림에서 DeepLab은 다음 Class에서 살펴볼 segmentation 방법이며, 여기에도 dilated convolution을 사용하긴 했지만 구조가 약간 다르다.
Context 모듈
Front-end 모듈뿐만 아니라 다중 scale의 context를 잘 추출해내기 위한 context 모듈도 개발을 하였으며, Basic과 Large 모듈이 있다. Basic type은 feature-map의 개수가 동일하지만 Large type은 feature-map의 개수가 늘었다가 최종단만 feature-map의 개수가 원래의 feature-map 개수와 같아지도록 했다.
Context 모듈은 기본적으로 어떤 망이든 적용이 가능할 수 있도록 설계를 하였으며, 자신들의 Front-end 모듈 뒤에 Context 모듈을 배치하였다.
Context 모듈의 구성은 아래 표와 같으며, 전부 convolutional layer로만 구성이 된다. 아래 표에서 C는 feature-map의 개수를 나타내고, Dilation은 dilated convolution의 확장 계수이며, convolution만으로 구성이 되지만, 뒷단으로 갈수록 receptive field의 크기가 커지는 것을 확인할 수 있다.
결과
Front-end 모듈만 적용해도 기존 segmentation 논문들보다 성능이 개선이 되었는데 context 모듈을 추가하면 추가적인 성능 개선이 있으며, CRF-RNN까지 적용을 하면 더 좋아지는 것을 알 수 있다.
아래 표에서 Front end는 Front end 모듈만 있는 경우이고 Basic/Large는 Context 모듈까지 적용된 경우이고 CRF는 CRF까지 적용된 경우이고, RNN은 CRF-RNN이 적용된 경우이다.
이번 Class에서는 Fisher Yu의 dilated convolution을 이용한 segmentation 방법에 대하여 살펴보았다. 이들은 FCN에서 사용한 VGG-16 망을 그대로 사용하지 않고, 철저한 분석을 통해 뒷부분을 수정한 Front-end 모듈을 만들어 냈다. 또한 Dilated convolution을 사용하여 망의 복잡도를 높이지 않으면서도 receptive field를 넓게 볼 수 있어 다양한 scale에 대응이 가능하게 하였다. 또한 그 개념을 활용한 Context 모듈까지 만들어 성능 개선을 꾀하였다.
다음 Class에서는 DeepLab에 대하여 살펴볼 예정이다..
​
​
[Part Ⅶ. Semantic Segmentation]
6. DeepLab [1]
Semantic Segmentation ? “DeepLab” ? part1
[Part VII. Semantic Segmentation] 5. Dilated Convolution 을 통하여Fisher Yu가 발표한 “dilated convolution”에 대하여 살펴보았다. Dilated convolution에서는 먼저 발표된 FCN과 DeepLab V1의 구조적인 문제점을 개선하였기 때문에 어떻게 보면 DeepLab V1의 구조를 먼저 검토하는 것이 좋았을지도 모르겠지만, 1년 뒤에 발표된 DeepLab V2는 dilated convolution보다 성능이 더 좋고 시기적으로도 dilated convolution 보다 나중에 발표되었기 때문에 dilated convolution을 먼저 살펴보았다.
DeepLab 팀은 2015년과 2016년에 거의 비슷한 제목과 구조를 갖는 논문을 발표한다. 2015년에 발표된 구조를 DeepLab V1이라고 하고, 그 다음에 발표된 개선된 구조를 DeepLab V2라고 한다. 이 두 구조는 atrous convolution과 fully connected CRF를 사용한다는 점은 동일하지만, multiple-scale에 대한 처리 방법이 V2에서는 개선이 되었으며, VGG-16 대신에 ResNet-101을 기본 망으로 사용하여 성능을 끌어올렸다는 점에서는 차이가 있다. (참고로 V1의 정확도가 71.6% IOU이었던 것을 V2에서는 79.7%까지 끌어 올린다.)
DeepLab V1: Semantic image segmentation with deep convolutional nets and fully connected CRFs
http://arxiv.org/pdf/1412.7062.pdf
DeepLab V2: DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs
http://arxiv.org/pdf/1606.00915.pdf
이번 Class에서는 논문 및 관련 자료들을 바탕으로 좀더 성능이 개선된 DeepLab V2의 구조에 대하여 살펴볼 예정이다. (이후부터는 꼭 구별이 필요한 경우가 아니라면, DeepLab이라고 부르면 DeepLab V2 구조를 가리킨다.)
논문의 제목에 다 나열되어 있듯이, DCNN(deep convolutional neural networks)와 atrous convolution 및 fully connected CRF 개념을 잘 활용하여 보다 정확도가 개선된 semantic segmentation이 가능하게 되었다. 재미 있는 사실은 V1에서는 hole algorithm 이라는 용어를 사용했지만, V2부터 atrous convolution으로 바꿔 부른 것을 보면, Fisher Yu 팀의 결과를 의식한 것 같다.(^^)
Classification 기반 망을 semantic segmentation에 적용할 때의 문제점
분류(classification)나 검출(detection)은 기본적으로 대상의 존재 여부에 집중하기 때문에 object-centric하며, 강력한 성능을 발휘하려면 여러 단계의 “conv+pooling”을 거쳐 말 그대로 영상 속에 존재하며 변화에 영향을 받지 않는(혹은 덜 받는) 강인한 개념만을 끄집어내야 하며, 그래서 detail보다는 global한 것에 집중을 한다.
반면에 semantic segmentation은 픽셀 단위의 조밀한 예측이 필요한데, classification 망을 기반으로 segmentation 망을 구성하게 되면 계속 feature-map의 크기가 줄어들기 때문에 detail 정보를 얻는데 어려움이 있다.
그래서 FCN 개발자는 skip layer를 사용하여 1/8, 1/16, 및 1/32 결과를 결합하여 detail이 줄어드는 문제를 보강하였으며, DeepLab과 앞서 살펴본 dilated convolution 팀은 망의 뒷 단에 있는 2개의 pooling layer를 제거하고, dilated convolution 혹은 atrous convolution을 사용하여 receptive field를 확장시키는 효과를 얻었으며, 1/8 크기까지만 줄이는 방법을 사용하여 detail이 사라지는 것을 커버하였다.
1/8까지만 사용하더라도 다음과 같은 문제점이 있다.
Receptive field가 충분히 크지 않아 다양한 scale에 대응이 어렵다.
1/8정보를 bilinear interpolation을 통해서 원 영상의 크기로 키우면, 1/32 크기를 확장한 것보다 detail이 살아 있기는 하지만, 여전히 정교함이 떨어진다.
이 문제를 DeepLab 팀과 dilated convolution팀에서는 다른 방식으로 해결을 하였으며, dilated convolution 팀은 DeepLab 팀의 atrouse convolution에서 힌트를 많이 얻은 것처럼 보인다.
Atrous Convolution
[Part VII. Semantic Segmentation] 5. Dilated Convolution 에서 살펴본 것처럼, Atrous Convolution이란 웨이브릿(wavelet)을 이용한 신호 분석에 사용되던 방식이며, 보다 넓은 scale을 보기 위해 중간에 hole(zero)을 채워 넣고 convolution을 수행하는 것을 말한다.
직관적인 이해를 돕기 위해 논문에 나온 1차원 convolution 그림을 살펴보자. 위 그림 (a)는 기본적인 convolution이며 인접 데이터를 이용해 kernel의 크기가 3인 convolution의 예이다.
(b)는 확장 계수(k)가 2인 경우로 인접한 데이터를 이용하는 것이 아니라 중간에 hole이 1개씩 들어오는 점이 다르며, 똑 같은 kernel 3을 사용하더라도 대응하는 영역의 크기가 커졌음을 확인할 수 있다.
이처럼, atrous convolution(dilated convolution)을 사용하면 kernel 크기는 동일하게 유지하기 때문에 연산량은 동일하지만, receptive field의 크기가 커지는(확장되는) 효과를 얻을 수 있게 된다.
영상 데이터와 같은 2차원에 대해서도 아래와 같이 좋은 효과가 있는 것을 확인할 수 있으며, 이 부분에 대한 설명은 [Part VII. Semantic Segmentation] 5. Dilated Convolution 을 참고하면 된다.
Atrous convolution 및 Bilinear interpolation
DeepLab V2에서는 VGG-16뿐만 아니라 ResNet-101 도 DCNN 망으로 사용을 하였으며, ResNet 구조를 변형시킨 모델을 통하여 VGG-16을 사용할 때보다 성능을 더 끌어올렸다. 앞서 이야기 한 것처럼, DCNN에서 max-pooling layer 2개를 제거함으로써 1/8크기의 feature-map을 얻고, atrous convolution 을 통해 넓은 receptive filed를 볼 수 있도록 하였다.
pooling 후 동일한 크기의 convolution을 수행하면, 자연스럽게 receptive field가 넓어지는데, 여기서는 detail을 위해 pooling layer를 제거했기 때문에, 이 부분을 atrous convolution을 더 넓은 receptive field를 볼 수 있도록 하여, pooling layer가 사리지는 문제를 해소시켰다. (앞서 살펴본 논문의 2-D 구조 그림을 살펴보면 이해가 갈 것이다.)
이후는 FCN이나 dilated convolution과 마찬가지로 bilinear interpolation을 이용해 원 영상을 복원해냈다. (아래 그림 참고)
Atrous convolution은 receptive field 확대를 통해 특징을 찾는 범위를 넓게 해주기 때문에 전체 영상으로 찾는 범위를 확대하면 좋겠지만, 이렇게 하려면 단계적으로 수행을 해줘야 하기 때문에 연산량이 많이 소요될 수 있다.
그래서 이 팀은 적정한 선에서 trade-off를 했으며, 나머지는 bilinear interpolation을 선택하였다. 하지만 bilinear interpolation만으론 정확하게 비행기를 픽셀단위까지 정교하게 segmentation을 한다는 것이 불가능하기 때문에 뒷부분은 CRF(Conditional Random Field)를 이용하여 post-processing을 수행하도록 하였다.
결과적으로 전체적인 구조는 DCNN + CRF의 형태이며, DCNN의 앞부분은 일반적인 convolution을 사용하고 뒷부분은 atrous convolution을 수행하였으며, 전체 구조는 아래 그림과 같다.
ASPP(Atrous Spatial Pyramid Pooling)
DeepLab V1과 달리 V2에서는 multi-scale에 더 잘 대응할 수 있도록 “fc6” layer에서의 atrous convolution을 위한 확장 계수를 아래 그림과 같이 {6, 12, 18. 24}로 적용을 하고 그 결과를 취합하여 사용을 하였다.
ResNet 설계자인 Kaiming He의 SPPNet 논문에 나오는 Spatial Pyramid Pooling기법에 영감을 받아 ASPP(Atrous Spatial Pyramid Pooling)으로 이름을 지었으며, 확장 계수를 6부터 24까지 변화시킴으로써 다양한 receptive field를 볼 수 있게 되었다. SPPNet에서의 방식처럼, 이전 단계까지의 결과는 동일하게 사용을 하고, fc6 layer에서 atrous convolution을 위한 확장 계수 r 값만 다르게 적용한 후 그 결과를 합치게 되면, 연산의 효율성 관점에서 큰 이득을 얻을 수 있다. 참고로 구글의 인셉션 구조도 여러 receptive field의 결과를 같이 볼 수 있게 되어 있다. (SPPNet에 대한 설명은 [Part V. Best CNN Architecture] 8. ResNet [4], Fast-RCNN 참고)
논문 저자들의 실험에 따르면, 단순하게 확장 계수 r을 12로 고정시키는 것보다. 이렇게 ASPP를 지원함으로써 약 1.7% 정도 성능을 개선할 수 있게 되었다.
위 그림에서 (a)는 DeepLab V1의 구조이며, ASPP를 지원하지 않는 경우에 fc6의 확장 계수를 12로 고정한 경우이며, (b)는 V2에서 fc6의 계수를 {6. 12, 18, 24}로 설정하고 ASPP를 수행하는 구조를 나타낸다.
성능은 아래 표와 같다. 아래 표에서 LargeFOV는 기존처럼 r = 12로 고정을 한 경우이며, ASPP-S는 r = {2, 4, 8, 12}로 좁은 receptive field를 대응할 수 있도록 작은 확장 계수 값을 갖는 경우이며, ASPP-L은 앞서 살펴본 것처럼 넓은 receptive field를 볼 수 있게 하는 경우이다.  실험에 사용한 망은 VGG-16 로 동일하다. 결과는 scale을 고정시키는 것보다는 multi-scale을 사용하는 편이 좋으며, 좁은 receptive field보다는 넓은 receptive field를 보는 편이 좋다는 사실을 알 수 있다.
Fully Connected CRF
이번 Class DeepLab의 Atrous convolution 및 ASPP에 대하여 살펴보았다. 이것만 사용하더라도 앞서 살펴본 FCN 보다 결과가 좋다는 것을 알 수 있지만, 아래 그림처럼 CRF(Conditional Random Field)를 사용한 후보정 작업을 해주면 좀 더 결과가 좋아지는 것을 확인할 수 있다.
그래서 다음 Class에서는 CRF에 대하여 좀 더 자세한 설명을 할 예정이며, 이번 Class에서 분량 문제로 살펴보지 못한 성능 개선을 위한 방법과 실험 결과에 대하여 살펴볼 예정이다.
​
​
[Part Ⅶ. Semantic Segmentation]
6. DeepLab [2]
Semantic Segmentation ? "DeepLab" ? part2
?
?
지난 [Part VII. Semantic Segmentation] 6. DeepLab [1] 에서는DeepLab에서 성능을 끌어올리기 위한 방법으로 Atrous convolution(일명 hole algorithm)을 사용하여 DCNN(Deep Convolutional Neural Network)에서 여러 단계의 sub-sampling으로 인한 해상도 저하 문제를 어느 정도 해결하면서 연산 시간 절감 효과를 얻었으며, multi-scale 대응을 위하여 fc6 단계에서 다양한 Atrous 계수를 적용한 ASPP(Atrous Spatial Pyramid Pooling)을 사용하였다는 것에 대하여 알아보았다.
1/8 크기의 해상도를 갖는 DCNN 결과를 bi-linear interpolation을 통해서 원영상의 크기로 확대하는 경우 아래의 그림처럼 해상도가 떨어지는 문제가 있다. DeepLab 구조에서는 이 문제 해결을 위해CRF(Conditional Random Field)를 사용한 후처리 과정을 통해 좋은 결과를 얻을 수 있었다. 이번 Class에서는 이 CRF에 대하여 살펴볼 예정이다.
?
왜 CRF(Conditional Random Field)가 필요한가?
?
분류(classification)와 같이 object-centric한 경우는 가능한 높은 수준의 공간적인 불변성(spatial invariance)를 얻기 위해 여러 단계의 "conv+pooling"을 통하여 영상 속에 존재하며 변화에 영향을 받지 않는(혹은 덜 받는) 강인한 특징을 추출해야 하며, 그래서 detail보다는 global한 것에 집중을 한다.
?
반면에 semantic segmentation은 픽셀 단위의 조밀한 예측이 필요한데, classification 망을 기반으로 segmentation 망을 구성하게 되면 계속 feature-map의 크기가 줄어들기 때문에 detail한 정보를 얻을 수 없다.
?
이 문제에 대한 해결책으로 FCN에서 skip connection(layer)을 사용하였고, dilated convolution이나 DeepLab에서는 마지막에 오는 pooling layer 2개를 없애고 dilated(atrous) convolution을 사용하였다.
하지만 위와 같은 방법을 사용하더라도 분명히 한계는 존재하기 때문에, DeepLab에서는 atrous convolution에 그치지 않고 CRF(conditional random field)를 후처리 과정으로 사용하여 픽셀 단위 예측의 정확도를 더 높일 수 있게 되었다.
?
Fully Connected CRF
?
?
일반적으로 좁은 범위(short-range) CRF는 segmentation을 수행한 뒤에 생기는 segmentation 잡음을 없애는 용도로 많이 사용이 되어 왔다.
?
하지만 앞서 살펴본 것처럼, DCNN에서는 여러 단계 "conv+pooling"을 거치면서 크기가 작아지고 그것을 upsampling을 통해 원영상 크기로 확대하기 때문에 이미 충분히 부드러운(smooth) 상태이며, 여기에 기존처럼 short-range CRF를 적용하면 결과가 더 나빠지게 된다
.
이것에 대한 해결책으로 2011년 스탠포드 대학교의 Philipp Krahenbuhl이 "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials"라는 논문을 발표하였는데, 이 논문은 기존에 사용이 되던 short-range CRF 대신에 전체 픽셀을 모두 연결한(fully connected) CRF 방법을 개발하여 놀라운 성능을 얻게 되었고, 이후에 많은 사람들이 fully connected CRF를 후처리에 많이 사용하게 되었다.
?
Fully connected CRF에 대한 설명은 논문을 참고하는 것도 좋지만, 아래의 링크에 있는 자료에 설명이 아주 잘 되어 있기 때문에 이 자료를 참고하면 좋을 것 같다.
?
(1) Inference in Fully Connected CRFs with Gaussian Edge Potentials
http://swoh.web.engr.illinois.edu/courses/IE598/handout/fall2016_slide15.pdf
?
(2) DeepLab?semantic image segmentation
http://www.cs.jhu.edu/~ayuille/JHUcourses/ProbabilisticModelsOfVisualCognition/Lecture20DeepNetwork2/Pages%20from%20DeepMindTalk.pdf
?
이 자료에 있는 일부 그림을 사용하여 개념 설명을 한다.
?
기존에 사용이 되던 short-range CRF는 아래 그림과 같이 local connection 정보만을 사용을 하다. 이러게 되면, 아래 그림과 같이 detail 정보를 얻을 수 없다.
?
?
반면에 fully connected CRF를 사용하게 되면, 아래 그림과 같이 detail이 살아 있는 결과를 얻을 수 있게 된다.
?
?
?
?
위와 같이, MCMC(Markov Chain Monte Carlo) 방식을 사용하게 되면 좋은 결과를 얻을 수 있지만 시간이 오래 걸리는 문제가 있기 때문에 거의 적용이 불가능하였지만, Philipp Krahenbuhl의 논문에서는 이것을 0.2초 수준으로 효과적으로 줄일 수 있는 방법을 개발해냈다. 일명 mean field approximation 방법을 적용하여 message passing을 사용한 iteration 방법을 적용하게 되면, 효과적으로 fully connected CRF를 수행할 수 있게 된다.
?여기서 mean field approximation이란, 물리학이나 확률이론에서 많이 사용되는 방법으로, 복잡한 모델을 설명하기 위해 더 간단한 모델을 선택하는 방식을 말한다. 수많은 변수들로 이루어진 복잡한 관계를 갖는 상황에서 특정 변수와 다른 변수들의 관계의 평균을 취하게 되면, 평균으로부터 변화(fluctuation)을 해석하는데도 용이하고, 평균으로 단순화된 또는 근사된 모델을 사용하게 되면 전체를 조망하기에 좋아진다.
?
CRF의 수식을 보면, unary term과 pairwise term으로 구성이 된다. 아래 식에서 x는각 픽셀의 위치에 해당하는 픽셀의 label이며, i와 j는 픽셀의 위치를 나타낸다. Unary term은 CNN 연산을 통해서 얻을 수 있으며, 픽셀간의 detail한 예측에는 pairwise term이 중요한 역할을 한다. Pairwise term에서는 마치 bi-lateral filter에서 그러듯이 픽셀값의 유사도와 위치적인 유사도를 함께 고려한다.
?
?
?
위 CRF 수식을 보면, 2개의 가우시안 커널로 구성이 된 것을 알 수 있으며, 표준편차 σα, σβ, σγ를 통해 scale을 조절할 수 있다. 첫번째 가우시안 커널은 비슷한 위치 비슷한 컬러를 갖는 픽셀들에 대하여 비슷한 label이 붙을 수 있도록 해주고, 두번째 가우시안 커널은 원래 픽셀의 근접도에 따라 smooth 수준을 결정한다. 위 식에서 pi, pj는 픽셀의 위치(position)를 나타내고, Ii, Ij는 픽셀의 컬러값(intensity)이다.
이것을 고속을 처리하기 위해, Philipp Krahenbuhl 방식을 사용하게 되면 feature space에서는 Gaussian convolution으로 표현을 할 수 있게 되어 고속 연산이 가능해진다.
DeepLab의 동작 방식
?
?
CRF까지 적용된 DeepLab의 최종적인 동작 방식은 아래 그림과 같다.
?
?
DCNN을 통해 1/8 크기의 coarse score-map을 구하고, 이것을 bi-linear interpolation을 통해 원영상 크기로 확대를 시킨다. Bilinear interpolation을 통해 얻어진 결과는 각 픽셀 위치에서의 label에 대한 확률이 되며 이것은 CRF의 unary term에 해당이 된다. 최종적으로 모든 픽셀 위치에서 pairwise term까지 고려한 CRF 후보정 작업을 해주면 최종적인 출력 결과를 얻을 수 있다.
?
?
DeepLab의 최종 결과
?
앞서 설명한 것처럼, DeepLab V1은 VGG-16을 기반으로 하고 있으며 ASPP가 없다. DeepLab V2는 ResNet-101에 기반을 하고 있으며, 아래 그림에서 볼 수 있는 것처럼 V2가 V1에 비해 결과 좋으며, CRF 수행 전후 그림을 비교하면, 역시 CRF를 수행하면detail 이 상당히 개선됨을 알 수 있다.
?
?
아래 표는 PASCAL VOC2012 데이터에 대한 실험 결과인데, ResNet-101에 CRF를 적용하면, 평균 IOU가 79.7% 수준으로 매우 높다는 것을 확인할 수 있다.
?
?
그 외에도 다양한 데이터를 바탕으로 실험한 결과 역시 매우 좋게 나왔지만, 이 결과값은 논문을 참고하면 될 것 같다.
?
이것으로 DeepLab에 대한 설명을 마치며, 다음 Class에서는 RNN(Recurrent Neural Network)을 이용한 semantic segmentation 방법에 대하여 살펴볼 예정이다.
​
​
[Part Ⅶ. Semantic Segmentation]
7. RNN, LSTM, GRU
Semantic Segmentation ? “RNN, LSTM, GRU”
[Part VII. Semantic Segmentation] 3. FCN [1] ~ 6. DeepLab [2] 를 통하여 이미 검증된Classification network의 후반부를 변형시켜 semantic segmentation을 하는 방법들에 대하여 살펴보았다. 지금까지 살펴본 network들은 모두 feed-forward 구조를 갖고 있는 전형적인 CNN 망이었다.
?
기존 방법과 달리 순환 신경망(RNN, Recurrent Neural Network)을 사용하여 semantic segmentation을 하는 방법들이 2014년 이후로 발표되고 있으며, 그 방법들에 대하여 살펴볼 예정이지만, 본격적인 검토에 앞서 RNN 및 그 변형/개선 모델에 대하여 살펴보는 것이 좋을 것 같아 이번 Class에서는 먼저 RNN, LSTM, GRU에 대하여 살펴보고 다음 Class에서 RNN 계열의 semantic segmentation에 대하여 살펴볼 예정이다.
RNN(Recurrent Neural Network)
지금까지 살펴본 대부분의 신경망은 신호의 흐름이 입력에서 출력으로만(즉, 한방향으로만) 전개되는 것들이 대부분이었다. 실제 신경망은 이런 식으로 구성이 되어 있지 않지만, 해석하기에 편리한 구조이기 때문에 흔히 사용이 되고 있다.
?
반면에 RNN은 그 이름에 있는 “recurrent”가 “순환적인”이라는 뜻을 갖고 있는 것처럼, 신호가 한쪽 방향으로 흘러가는 것이 아니라 순환 구조를 갖는다. 또한 내부에 과거의 상태를 저장하는 메모리를 갖고 있기 때문에 순차적인 문제나 맥락을 파악해야 하는 경우나 시간에 대한 의존성을 갖고 있는 문제 해결에 적합하다. 그래서 자연어 연구나 번역 쪽에 아주 적합한 구조를 갖고 있으며, 최근에는 영상 처리나 분석에도 활용이 많이 되고 있다.
?
RNN과 feed-forward 신경망의 차이는 다음 그림과 같다. 이 그림은 아래 출처에서 인용을 하였으며, 다차원 RNN에 대한 설명이 잘된 자료이기 때문에 참고를 하면 좋을 것 같다.
(그림 출처: http://www.slideshare.net/grigorysapunov/multidimensional-rnn)
RNN은 1980년대에 그 기본 이론이 정립이 되었고, 구조적인 특성으로 인해 관심을 많이 끌었지만 아쉽게도 몇 가지 문제점을 갖고 있으며, RNN이 갖고 있는 기본적인 문제점을 해결한 LSTM(Long Short-Term Memory, 장/단기 메모리)가 1997년 Hochreiter와 Schumidhumer에 의해서 발표가 되면서 비로서 실질적인 문제들을 해결할 수 있게 되었고 많은 사람들의 관심을 끌게 된다.
RNN이나 LSTM에 대한 설명은 한글로 번역된 좋은 자료도 많이 있기 때문에 이 자료를 참고하면 좋을 것 같다.
http://aikorea.org/blog/rnn-tutorial-1/   : (part1 ~ part4)
?
특히 RNN의 일반적인 개념과 LSTM에 대한 설명은 Christopher Olah의 블로그에 그림을 통해서 설명이 잘 되어 있기 때문에 여기를 참고하면 좋을 것 같다.
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
위 블로그 등에 설명이 이미 잘 되어 있기 때문에 자세한 내용은 이 블로그를 참고하면 좋을 것 같다. 다만 여기서는 핵심만 간단히 정리하기로 한다.
RNN의 문제점
?
RNN은 기본 구조를 이해하려면, Olah의 블로그의 그림처럼 시간을 기준으로 순환 구조를 풀어서 생각하면 이해가 쉽다.
RNN의 결과는 셀에 저장된 이전 state와 입력에 의해서 결정이 되는 구조이다. RNN에 대한 학습은 feed-forward 신경망과 비슷하게 역전파(back-propagation)을 통해서 학습을 시키지만 시간에 걸쳐 풀어서 해석을 해석을 하는 BPTT(Back-Propagation Through Time) 방식을 취한다.
위 그림처럼, 펼쳐놓은 상태에서 back-propagation을 시키게 되는데, 여기서 feed-forward와 달리 파라미터를 공유한다는 점에서 차이가 있다. (시간 domain에서 펼쳤을 뿐이고, 원래 망에서는 순환구조이기 때문에 동일한 파라미터를 사용한다)
일반적인 feed-forward 신경망에서도 망이 깊어지게 되면 vanishing/exploding gradient 문제로 학습시키기가 어려웠고, 그래서 많은 회피 방법들이 개발이 되었다. ([Part VI. CNN 핵심 요소 기술] 1. Batch Nomalization [1] ~ 3. Stochastic Pooling 참고)
?
RNN을 사용하여 장시간 데이터 의존도가 있는 문제(long term 메모리가 필요)를 풀고자 하는 경우, 현재 상태가 계속 이전 상태와 연관이 있기 때문에 BPTT를 통해 연산을 할 때, chain rule에 의한 연결의 길이가 매우 길어지게 되고, 결과적으로 깊은 신경망에서 생기는 vanishing/exploding gradient 문제로 인해서 학습하기가 어려워진다.
?
결과적으로 순수하게 RNN만으로는 장시간에 걸친 의존도가 있는 문제에 대해서는 풀기가 어려워진다. 뭔가 해결책이 필요해진다.
LSTM
?
LSTM은 RNN으로 잘 해결할 수 없는 장기 메모리가 필요한 문제를 해결하기 위해서 개발이 되었다. 상세한 설명은 Olah의 블로그에 워낙 잘되어 있기 때문에 거기를 참고하면 좋을 것 같다. 여기서는 RNN과 LSTM 의 구조만 간단하게 살펴보자.
?
전통적인 RNN의 구조는 위 그림과 같고, LSTM은 아래 그림처럼 좀 더 복잡한 구조로 이루어져 있으며, 장기간 메모리 역할을 수행하는 cell state(?아래 그림에서 위쪽으로 수평으로 흐르는 라인)와 연결의 강도를 조절하는 3개의 gate(forget, input, output)로 구성이 된다.
위 그림을 보면 알 수 있듯이 cell-state는 gate 조절을 통해 이전 state 정보가 현재 state로 끼치는 영향을 조절할 수도 있고, 여기에 현재 입력과 연관된 정보를 추가할 수도 있으며, 다시 출력에 끼치는 영향의 수준을 정할 수가 있게 된다.?
결과적으로는 RNN이 갖고 있던 장기간 메모리가 필요한 문제가 해결이 가능하게 되었고 많은 분야에서 성공적으로 활용이 된다.
GRU(Gated Recurrent Unit)
?
LSTM이 성공을 크게 거두면서, LSTM의 구조를 변경한 LSTM 류의 많은 구조들이 발표가 된다. 그 중 점차 주목을 받아가고 있는 구조가 바로 GRU(Gated Recurrent Unit)이다.
?
GRU는 뉴욕대의 교수로 있는 한국인 조경현 교수가 2014년에 발표한 논문 “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation ”에 처음 소개가 되었다. (얼마 되지 않았는데 1097회나 인용이 되었다)
http://arxiv.org/pdf/1406.1078.pdf
이 논문을 직접 살펴보는 것보다는 역시 뉴욕대의 정준영씨가 발표한 논문 “Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling”에 tanh 기반의 기본 RNN, LSTM과 GRU를 비교한 내용이 잘 설명이 되어 있기 때문에 이것을 참고로 한다. (GRU와 LSTM을 비교하는 내용이 대부분이지만, GRU가 대세가 되어감을 입증하듯이 448회나 인용이 되었다. ^^)
http://pdfs.semanticscholar.org/2d9e/3f53fcdb548b0b3c4d4efb197f164fe0c381.pdf
앞서 살펴본 Olah의 아름다운(?) 그림 대신 논문에서 사용한 LSTM과 GRU의 구조는 아래 그림과 같다. GRU의 구조가 LSTM 의 구조에 비해 간결하다는 것을 확인할 수 있다.
GRU의 구조를 자세히 보면, LSTM과 마찬가지로 gate를 이용하여 정보의 양을 조절하는 것은 동일하지만, gate의 제어 방식에는 차이가 있음을 알 수 있다.
?
GRU에는 update와 reset 2개의 gate가 있으며, 이것들을 이용해 LSTM과 거의 비슷한 기능을 수행한다. ?시간 t에서의 GRU의 activation은 과거의 activation ht-1과 후보 activation ht와의 interpolation을 통해서 결정이 된다. 여기서 interpolation의 비율은 update gate z를 통해서 결정이 된다.
결과적으로 보면, GRU에서는 LSTM의 forget과 input gate를 결합하여 1개의 update gate를 만들었으며, 별도의 cell state와 hidden state를 hidden state로 묶은 셈이 되었다. 그래서 LSTM보다는 간단한 구조를 갖게 되었지만, 그 성능은 정준영씨의 논문의 결과처럼 LSTM에 뒤지지 않기 때문에 점차 유명세를 타고 있다.
?
?
이번 Class에서는 RNN 기반의 semantic segmentation에 앞서 기본적인 RNN, LSTM 및 GRU의 개념에 대하여 살펴보았다. 다음 class에서는 RNN을 vision에 적용한 응용에 대하여 좀 더 살펴볼 예정이다.
​
​
[Part Ⅶ. Semantic Segmentation]
8. ReNet
?
Semantic Segmentation ? “ReNet”
[Part VII. Semantic Segmentation] 7. RNN, LSTM, GRU 에서는 순환 신경망(RNN)의 기본 개념 및 RNN의 문제를 개선한 RSTM과 GRU에 대하여 간단하게 살펴보았다. RNN을 깊이 있게 살펴보는 것이 주목적이 아니고 semantic segmentation을 구현하는 여러 가지 방법 중 RNN을 사용한 방법에 대해서 살펴보기 전에 RNN에 대한 기본 개념을 살펴보고자 하는 목적이기 때문에 이 정도로 간단하게 살펴본다.
이번 Class에서는 RNN을 사용한 semantic segmentation을 살펴보기 전에, RNN을 사용한 image classification을 구현하는데 사용이 된 ReNet에 대하여 살펴볼 예정이다. ReNet을 살펴보는 이유는 2015년에 Francesco Visin이 “ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks” 발표한 이후 RNN을 사용한 방법들에서 ReNet이 여러 곳에서 사용이 되고 있으며, 다음에 살펴볼 semantic segmentation인 ReSeg가 ReNet에 기반을 두고 있기 때문이다.
(ReNet: http://arxiv.org/pdf/1505.00393.pdf)
또한 ReNet 논문에서는 Yann LeCun이 발표한 LeNet([Part V. Best CNN Architecture] 2. LeNet 참고)과의 비교를 통하여, Visin팀의 주장처럼 RNN이 CNN의 대안이 될 수 있는지를 살펴보는 것도 흥미로울 것 같다.
ReNet
CNN이 컴퓨터 비전 분야에서 놀랄만한 성공을 거두었듯이, 그 동안 RNN은 자연어 처리나 통역과 같이 주로 순차적인 데이터를 모델링하는 쪽에서 성공을 거두어 왔었다.
이 논문에서는 그간 영상 관련 분야에서 널리 쓰여온 “convolution + pooling” layer를 4개의 1-차원 RNN layer를 통하여 구현하는 것을 기본 목표로 하고 있다. 여기서 4개의 RNN은 아래에서 위로, 위에서 아래로, 왼쪽에서 오른쪽으로 및 오른쪽에서 왼쪽으로 sliding window를 옮기듯이 처리를 하는 것을 말한다. (아래 그림 참고)
CNN에서 “convolution + pooling”을 사용하게 되면, receptive field에 해당하는 local 영역의 feature를 추출할 수 있는데 반해, 4방향의 1-D RNN에서는 전체 이미지에 대한 특정 위치에서의 activation을 볼 수 있기 때문에 좀 더 넓은 영역을 볼 수 있게 된다.
이러한 좋은 성질로 인해 나중에 나오는 다른 논문에서는 semantic segmentation에 사용을 하거나 CNN과 RNN을 섞어 사용하는 논문들도 나오게 된다. (아래 논문 제목 및 링크 참고)
?
ReSeg: A Recurrent Neural Network-based Models for Semantic Segmentation
(http://arxiv.org/pdf/1511.07053.pdf)
?
?Layer Recurrent Neural Networks
(http://openreview.net/pdf?id=rJJRDvcex)
ReNet의 동작
ReNet의 기본 동작은 위 그림(Fig 1)에서 볼 수 있는 것처럼, 전체 영상의 영역을 패치(window) 영역이 겹치지 않도록 나눈다. Patch 영역은 CNN에서처럼, 겹치게 할 수도 있지만 여기서는 연산시간을 줄이기 위해 겹치지 않는 것을 선택하였지만, 이 선택은 구현하는 자의 몫이다.
[Part VII. Semantic Segmentation] 7. RNN, LSTM, GRU 에서 살펴본 것처럼, RNN을 펼쳐서 생각을 하면(위 그림처럼), 입력(x0, x1, …, xt)으로 들어가는 값은 각각의 patch 값이며, patch 값을 받아서 현재 patch에 대한 activation과 hidden state을 계산한다.
먼저 수직 방향으로 2개의 1-D RNN 연산(아래에서 위로, 위에서 아래로)을 수행하며, 이것을 수식으로 표현을 하면 아래와 같다.
여기서 fVFWD와 fVREV는 수직 방향(순방향, 역방향)으로 hidden state의 활성함수 이며, 단순 RNN, LSTM 및 GRU 등을 사용할 수 있다.
이렇게 수직 방향으로 sweep을 마치고 난 후 얻어지는 결과를 결합하여, 임시의 복합 feature-map을 만들며, 이렇게 만들어지는 결과 vi,j는 전체 입력 feature-map의 j행(column) 패치들을 고려한 (i,j) 위치에서의 activation이 된다.
이렇게 수직 방향으로 sweeping을 한 후 얻어진 결과에 대하여 다시 수평방향으로(왼쪽에서 오른쪽, 오른쪽에서 왼쪽)으로 움직여가면서 각 위치에서의 activation hij를 구한다. 이렇게 구해진 hij는 전체 영상의 관점에서의 패치 (i,j) 위치에서의 activation이 된다.
일반적인 CNN을 수행하게 되면 자신의 위치에 있는 receptive field에 대한 activation 결과만을 사용하지만, RNN을 사용하면 수평 혹은 수직 방향으로의 연결(lateral connection)을 통해 전체 이미지를 고려할 수 있기 때문에 영상의 다른 위치에 있는 redundant한 특징을 제거하거나 병합할 수 있게 된다.
이렇게 개발된 4개의 1-D layer를 전체적으로 보면 1 RNN layer로 볼 수 있으며, 이것을 여러 단을 쌓게 되면, 여러 단의 “convolution + pooling” 과 같은 효과를 얻을 수 있다. 아래 그림은 여러 단을 RNN layer 뒤에 fully connected layer가 오는 전형적인 구조를 보여준다.
LeNet과 ReNet의 비교
논문의 저자는 CNN의 원형이라고 볼 수 있는 LeNet(Yann LeCu, Clas25 참고)과의 비교를 통하여 자신들의 ReNet이 꽤 괜찮은 성능을 보임을 입증하였다. (사실 아주 예전에 개발된 구조와 직접 비교를 한다는 것이 설득력이 부족하기는 하지만, 아직은 RNN을 사용한 영상 처리 구조가 그렇게 연구가 많이 되지 않았다는 점을 감안하면, 어느 정도는 타당한 것 같다)
ReNet과 LeNet은 패치(receptive field)에 대하여 동일한 필터를 적용한다는 점은 동일하다. 하지만 ReNet은 앞서 설명한 것처럼, 패치에 대해 좌우상하 4방향의 연결을 모두 보기 때문에 결과적으로 영상 전체를 본다는 점이 다르다.
LeNet(CNN)은 pooling을 통해서 영상의 크기를 줄여가면서, 결과적으로 spatial invariance를 얻게 되지만, RNN은 전체 이미지를 보기 때문에 pooling이 필요가 없다. 하지만 필요에 따라서 pooling을 선택적으로 사용하는 것도 무방하다.
LeNet(CNN)은 구조적인 관점에서 병렬적으로 수행하기에 좋지만, ReNet(RNN)의 동작은 순차적으로 이루어지기 때문에 병렬 연산에 적합하지 못하다. 하지만 ReNet은 파라미터의 수가 작다는 장점이 있으며, batch 연산을 할 때는 좀 더 상위 레벨의 parallelism을 사용할 수가 있으며, 병렬화에 대해서는 추가 연구가 필요하다.
모델의 구조와 실험 데이터
입력의 크기가 32x32로 요즘 사용하는 영상의 크기에 비해서 한참 작은 LeNet과 비교를 하였기 때문에, 작은 영상 데이터 집합인 MNIST(필기체 숫자), CIFAR-10(class 10개짜리 영상 데이터) 및 SVHN(구글의 StreetView 통해 획득된 숫자 데이터)를 비교 데이터로 선정하였다. (이미 이 데이터 들에 대한 설명은 이전 Class에 많이 있으니, 거기를 참고하면 된다.)
학습 데이터를 늘리기 위한 방법으로는 영상을 좌우 반전하는 clipping과 픽셀을 몇 픽셀씩 이동시키는 shifting 방법을 사용하였다.
RNN layer에는 [Part VII. Semantic Segmentation] 7. RNN, LSTM, GRU 에서 살펴본 GRU(Gated Recurrent Unit)과 LSTM(Long Short Term Memory)를 나누어 적용을 하였으며, 그들이 사용한 구조는 아래 표와 같다.
여기서 NRE는 RNN layer의 수를 나타내고, 그 아래에 있는 wp x hp는 패치의 크기를 나타낸다. (여기서는 패치 크기는 2x2만 사용을 하였으며, 실험에서는 연산 속도를 고려하여 패치를 겹치지 않도록 하였기 때문에 패치의 크기 너무 커지면 효과가 좋지 않을 것 같다). dRE는 각 RNN layer를 거쳤을 때 나오는 feature-map의 개수며, NFC는 fully connected layer의 개수이다.
이렇게 구조를 정하고 실험을 한 결과는 아래 표와 같다.
이 결과에서 볼 수 있듯이 ReNet의 결과는 기존 최고의 CNN의 결과를 압도할만한 수준은 아니지만, 그래도 꽤 괜찮은 성능을 보이는 것을 알 수 있다.
논문 저자의 주장은 굳이 성능을 조금이라도 더 끌어올리는 것이 목적이 아니라 RNN이 CNN의 대안이 될 수 있음을 보이는 것이 목적이라고 우겼으며(^^), 좀 더 많은 연구를 하면 의미 있는 결과가 나올 것이라고 전망하였다. (이것은 작은 트릭보다는 뭔가 구조적 변화가 좀더 필요할 것이라는 것을 저자들이 느끼지 않았을까 생각한다.)
이번 Class에서는 1-D RNN sub-layer 4개를 1개의 RNN layer로 사용하는 ReNet에 대하여 살펴보았다. 실험의 결과처럼 ReNet이 “convolution + pooling”의 대안이 될 가능성은 충분히 있어 보인다. 하지만 좀 더 연구가 필요할 것이라고 생각된다.
다음 Class에서는 ReNet을 기반으로 semantic segmentation에 RNN을 도입한 ReSeg 구조에 대하여 살펴볼 예정이다.
​
​
[Part Ⅶ. Semantic Segmentation]
9. ReSeg
?
Semantic Segmentation ? “ReSeg”
[Part VII. Semantic Segmentation] 8. ReNet 에서는 ReNet이라는 순환 신경망(RNN)에 대하여 살펴보았다. ReNet에서는 상하/좌우 방향으로 움직이며 영상으로부터 feature를 추출해내는 4개의 sub-layer로 구성된 RNN layer가 기존 CNN의 “convolution + pooling”의 대안이 될 수 있다는 사실을 확인할 수 있었다. 또한 이런 RNN layer를 몇 개를 적층시키면 deep CNN의 효과를 얻을 수 있게 된다.
?
?이번 Class에서는 ReNet을 발표한 저자들이 좀 더 연구를 진전시켜 그 이듬해(2016년)에 ReSeg라는 이름의 semantic segmentation 망을 발표하였는데, 그것에 대하여 살펴볼 예정이다. 논문의 제목은 “ReSeg: A Recurrent Neural Network ?based Model for Semantic Segmentation)이며, 출처는 아래 링크와 같다.
?(ReSeg: http://arxiv.org/pdf/1511.07053.pdf)?
?
ReNet이 순수하게 RNN 기반이었다면, ReSeg에서는 CNN과 RNN을 섞어 사용을 하였다. 이전 Class([Part VII. Semantic Segmentation] 3. FCN [1] ~ 6. DeepLab [2])에서 살펴보았던 것처럼, CNN의 장점은 연속되는 "convolution + pooling"을 통해 매우 강인한 feature 들을 추출해낼 수 있어 classification나 detection에서 매우 뛰어난 성능을 보이지만, 그 과정에서 해상도가 줄어들기 때문에 픽셀 수준의 조밀한 예측에는 적합하지 않다는 사실이다. 또한 receptive field의 제한으로 인해 비록 여러 단계의 pooling을 거치더라도 local한 성질을 갖는 한계로 인해, DeepLab에서는 CRF(Conditional Random Field)를 사용한 후처리(post-processing)을 통해 멀리 떨어져 있는 픽셀들과의 관계를 고려하기도 하였다.
?
?RNN의 가장 큰 장점은 feature를 추출할 때 영상 전체를 볼 수 있다는 것이며, 그래서 ReSeg의 저자들은 CNN의 local feature 추출 능력과 RNN의 원거리 혹은 전체 영상을 고려한 feature 추출 능력을 결합하여 semantic segmentation의 성능을 높이자는 아이디어로부터 ReSeg를 발전시켰다.
ReSeg의 구조 ? 앞부분은 VGG-16의 7 layer만 사용
?앞서 간단하게 설명한 것처럼, ReSeg는 CNN과 RNN을 섞어 사용을 하였다. ReSeg 팀 역시 크게 보면 FCN([Part VII. Semantic Segmentation] 3. FCN [1] ~ 3. FCN [2])으로부터 영향을 받은 것으로 보인다. ReSeg 팀은 FCN과 마찬가지로 CNN 망으로는 VGG-16을 사용하였다.
아래 그림은 다양한 VGGNet의 구조를 나타내며, 그 중 “D” 구조를 기본으로 사용을 하였으며, VGG-16을 전부 사용하는 것이 아니라 처음 7 layer(아래 그림에서 빨간색 영역)만을 사용하였다.
VGG-16을 모두 사용하면 입력 영상의 1/32까지 크기가 줄어들게 되어 해상도가 너무 낮아지는 문제가 있기 때문에, FCN 개발자들은 skip layer 개념을 사용하여 1/8해상도와 1/16 해상도의 결과를 그대로 bypass 시켜 1/32 결과와 합치는 방식을 사용하여 해상도를 높이는 방식을 사용하였다.
Fisher Yu(Dilated convolution ?방식의 개발자)는 1/8까지만 사용하고, 나머지 max-pooling은 제거하고 dilated convolution을 사용하여 떨어지는 해상도를 보강하였다.
이런 방식들과 비슷하게 ReSeg 팀도 VGG-16에서 최초의7 layer까지만 사용을 하여 입력 영상의 1/8까지만 될 수 있도록 하여, 해상도가 너무 낮아지는 문제를 피하려고 노력을 하였다. VGG-16의 앞부분만 사용을 하기 때문에 ImageNet을 이용한 pre-training 결과를 사용하도록 하였다.
ReSeg의 구조 ? 중간 부분에 ReNet을 적용
CNN layer의 뒷단에는 [Part VII. Semantic Segmentation] 8. ReNet 에서 살펴본 ReNet을 연결하여, 이미지 전체를 볼 수 있도록 하였다.
이것은 개념적으로 보면 DeepLab([Part VII. Semantic Segmentation] 6. DeepLab ~ 6. DeepLab)에서 CRF(Conditional Random Field)를 사용하여 좀 더 넓은 범위에서 영상의 context를 고려하는 것과 일맥 상통하며, 결과적으로 보면 RNN이 갖는 구조적 장점을 잘 활용하였다고 볼 수 있다.
DeepLab 개발자들은 CRF를 후처리(post-processing) 방식으로 사용하였기 때문에 CRF 과정이 학습에 분리가 되지만, 여기는 CNN 망 바로 뒤에 ReNet 망을 위치시키는 방식을 취했기 때문에 end-to-end 학습이 가능하다.
ReSeg의 구조 ? 뒷부분에는 Transposed convolution을 적용
ReNet을 적용하여 좀 더 넓은 영역을 볼 수 있게 되었지만, 최종 해상는 ReNet에서 2x2 non-overlapping patch 방식을 사용하기 때문에 입력보다 해상도가 가로/세로 방향으로 각각 1/2씩 줄어들게 된다.
FCN이나 Deeplab 등에서도 줄어든 해상도를 원영상 크기로 복원하기 위해 upsampling 방식을 사용하였다. 이들은 단순하게 bilinear interpolation을 사용하였다.
하지만 ReSeg팀은 원영상 크기로 복원했을 때의 결과를 고려하여 단순한 bilinear interpolation 대신에 Transposed convolution (혹은 fractionally strided convolution)을 적용하였다.
흔히 사용하는 stride의 크기가 1인 convolution은 주변 픽셀과의 필터링 연산을 통하기 때문에 영상의 boundary에 적절한 padding을 해주면, 원영상과 동일한 크기의 출력을 얻을 수 있으며, stride 크기가 1 이상이 되면 sub-sampling의 효과를 얻을 수 있기 때문에 출력 영상의 크기가 줄어든다.
반면에 fractionally strided convolution이란 stride의 크기가 1보다 작은 경우로 convolution을 수행하기 때문에 결과적으로 보면 원영상의 중간에 0을 끼워 넣고 convolution을 수행하여 영상의 크기를 크게 만드는 효과를 얻을 수 있으며, 결과적으로 보면 upsampling이 가능하게 된다.
Fractionally strided convolution의 동작은 위 그림과 같으며, 아래의 링크 사이트에 가면 각종 다양한 convolution에 대한 animation이 잘 표현되어 있기 때문에 참조를 해보면 좋을 것 같다.
http://github.com/vdumoulin/conv_arithmetic
ReSeg에 대한 실험 데이터
대부분의 segmentation 연구에 흔히 사용되는 PASCAL VOC 데이터를 사용했더라면 데이터가 대한 비교가 쉬웠을 텐데 이들은 다른 데이터 집합을 사용하였다.
먼저 간단하게 대상과 배경으로 구별해서 확인할 수 있는 “Weizmann Horses”와 “Oxford Flowers” 데이터 집합을 사용하였으며, 이것들은 아래 그림과 같다.
< Weizmann Horses >
< Oxford Flowers >
위 그림에서 볼 수 있듯이 2개의 데이터 집합은 테스트 영상에서 segmentation을 필요로 하는 후보군이 1개뿐이라서 비교적 단순하다. 좀 더 복잡한 테스트를 위해 거리를 배경으로 찍은 “CamVid” 데이터 집합을 사용하였으며, 이것들은 아래 그림과 같다.
ReSeg의 성능
논문에 나온 것처럼 픽셀별 예측의 정밀도를 나타내는 “Global acc”와 Ground-truth 데이터와 예측이 겹치는 부분을 보는 “Avg IoU”의 결과를 보면 상당히 좋다는 것을 알 수 있다.
비교적 복잡한 CamVid 결과에서도 아래 표와 같이 성능이 좋다는 것을 확인할 수 있다.
이번 Class에서는 CNN과 RNN의 장점을 결합한 ReSeg라는 semantic segmentation 방법에 대하여 살펴보았다. PASCAL VOC에 대한 실험 결과가 없어, 앞서 살펴본 FCN이나 DeepLab 등과 직접적인 성능 비교를 할 수 없다는 점이 좀 아쉽기는 다양한 segmentation을 방법을 살펴보는 것은 나름대로 의미가 있는 것 같다.
다음 Class에서는 2017년에 발표된 논문이며 역시 RNN을 사용하는 “Layer Recurrent Neural Network”에 대하여 살펴볼 예정이다.
​
​
[Part Ⅶ. Semantic Segmentation]
10. L-RNN
?
?
Semantic Segmentation ? “L-RNN”
[Part Ⅶ. Semantic Segmentation] 9. ReSeg 을 통하여 RNN과 CNN을 결합하여 semantic segmentation을 수행하는 ReSeg에 대하여 살펴보았다.
CNN은 여러 단계의 “convolution + pooling”을 통하여 강인한 local feature를 추출하는데 효과적이고, RNN은 영상이나 feature-map 전체를 고려할 수 있기 때문에 CNN과 RNN을 결합하면 좋은 결과를 나올 것이라는 것은 충분히 예상할 수가 있다. 핵심은 "어떻게 결합을 할 것인가"이다.
앞서 소개한 ReSeg을 통하여 CNN과 RNN의 결합의 효과 및 가능성을 확인할 수 있었다. ReSeg에서는 RNN을 별도의 layer 개념으로 사용하였지만, Oxford 대학교의 Weidi Xie 및 Andrew Zisserman은 여기서 한걸음 더 나아가, RNN을 내부 layer 처럼 (within layer) 사용할 수 있도록 “Layer RNN”이라는 개념을 고안하였으며, “Layer Recurrent Neural Network”이라는 이름으로 2017년에 발표를 하였다.
(L-RNN: http://openreview.net/pdf?id=rJJRDvcex)
이 팀은 Layer-RNN 이라는 모듈을 통하여, 기존에 잘 개발된 CNN의 어느 위치에든 결합할 수 있는 방법을 제시하였다. 또한 이렇게 개발된 L-RNN 개념을 classification과 semantic segmentation에 적용하여 그 성능을 입증하였으며, 상당히 의미가 있는 것 같아 이번 Class에서는 L-RNN에 대하여 살펴볼 예정이다.
ReNet 구조로부터 영감을 받아 L-RNN 개념을 고안
RNN의 가장 큰 장점 중 하나는 순환적인(recurrent) 성질을 이용하여 입력 영상이나 feature-map 전체를 볼 수 있기 때문에 receptive field의 제한에서 벗어날 수 있다는 점이다.
2015년 ReNet 팀이 수평(좌에서 우로, 우에서 좌로) 방향 및 수직(위에서 아래로, 아래에서 위로) 방향으로 움직이는 4개의 1-D RNN을 사용하면 다양한 스케일에서의 맥락정보(contextual information)을 추출할 수 있다는 것을 보여주었고, ReSeg를 통해 CNN과 RNN의 결합 효과도 입증을 하였다.
여러 단의 convolutional layer로 구성된 망에서 recurrence를 활용할 수 있는 방법은 크게 보면 "단과 단 사이(between layer) 방법"과 "단 내부(within layer) 방법"이 있을 수 있는데 L-RNN 팀은 layer 내부의 recurrence를 활용하는 방법을 선택하였다.
Layer-RNN의 구조
L-RNN의 기본 구조는 아래 그림과 같다. Convolution을 통해서 얻어진 local feature에 아래 그림처럼 4개의 1-D RNN sub-layer로 구성된 모듈을 통해 연산을 해주게 되면 공간적인 의존관계(spatial dependency)를 파악할 수 있게 된다.
위 그림에서 L-RNN의 구조를 자세히 들여다보면, ReNet 구조와 거의 유사하다는 것을 알 수 있다. 이들은 ReNet의 구조를 좀 더 일반화 시키는 관점에서 접근을 하였다. ReNet에서는 연산량을 줄이기 위해 2x2 크기의 겹치지 않는 patch를 구성하고 patch 단위로 처리를 하였지만, 여기서는 layer 내부에 들어가기 때문에 입력과 출력이 동일하게 되어야 하므로 patch가 아니라 픽셀 단위로 RNN 연산을 수행한다.
또한 CNN의 layer에 곧바로 적용이 가능하기 때문에 nonlinearity 를 추가할 수 있게 되어 더욱 다양한 표현이 가능해지는 장점도 있다.
앞서 살펴본 ReSeg는 앞 단에 CNN이 오고, 후반부에 ReNet이 오는 구조였다. 여기서는 RNN 부분이 별도의 layer로 사용이 되었지만, L-RNN 팀은 within layer 개념으로 만들었기 때문에 CNN의 어느 위치에나 편안하게 위치할 수 있다.
Within layer 개념으로 사용되는 Layer RNN의 기본적인 모듈은 아래 그림과 같다
?
위 그림에서 왼쪽의 CNN 모듈은 ResNet([Part Ⅴ. Best CNN Archtecture] 8. ResNet [1] ~ 8. ResNet [8] 참고)의 기본 블락이며 L-RNN의 개념을 이것과 비슷하게 생각할 수 있다. (a)의 Residual block을 (b) 처럼 바꾸게 되면, L-RNN 모듈이 된다. 위 그림에서 “x”는 CNN의 결과이며, CNN의 결과를 받아 RNN의 결과와 합치며, 합치는 방식은 위 그림처럼 3가지 방법이 있다.
위 그림에서 Batch normalization은 pre-training된 CNN 망을 사용할 때는 필요가 없지만, 그렇지 않고 망 전부를 바닥부터 학습을 시켜야 하는 경우에는 사용을 한다.
망의 내부에 RNN이 들어가기 때문에 CNN과 RNN의 결과를 합치는 방식으로는 아래와 같은 3가지가 가능한데, L-RNN 팀은 sum을 선택하였다.
Forward: RNN의 파라미터를 0으로 하는 경우에는 RNN결과를 사용하지 않고 CNN 결과만을 사용. 이 경우는 일반적인 CNN과 동일
Sum: CNN의 결과에 RNN의 결과를 더함.
Concatenate: CNN의 결과에 RNN의 결과가 추가가 되는 형태이기 때문에 output feature-map의 채널 수가 증가하여 결과적으로 파라미터가 늘어나게 된다.
Pre-training된 CNN 망에 L-RNN을 끼워넣기
Layer RNN이 추가되면 어떤 일이 일어나는지는 아래 그림을 보면 쉽게 이해할 수가 있다. 빠른 이해를 위해 1-D 관점에서만(수평방향/왼쪽에서 오른쪽) 살펴보자.
그림에서 왼쪽은 전통적인 CNN 망을 나타내며, 그림에서 Xinter라는 부분은 convolution을 마친 중간 결과이고 여기에 ReLU와 같은 non-linearity 활성 함수 f가 적용이 된다.
위 그림의 오른쪽은 L-RNN이 추가된 경우로 CNN 연산을 마친 중간 결과에 RNN 연산 결과를 더하는 형태가 된다.
이것을 수식으로 표현을 하게 되면 아래와 같다.
이 수식에서 (7)번 식은 일반적인 convolution 연산을 나타내는 수식이며, (9)번 식은 RNN이 더해진 것이다. 이 수식을 보면 극단적인 경우에 V를 0으로 하게 되면 일반적인 CNN의 식과 동일하다는 것을 알 수가 있다.
이렇게 L-RNN은 기존 CNN 망의 어느 위치라도 편안하게 추가를 할 수가 있게 되며, CNN이 갖고 있는 장점과 RNN이 갖고 있는 장점의 결합을 통해 성능을 극대화시킬 수 있게 된다.
L-RNN의 성능 평가 (Image Classification)
L-RNN은 앞서 설명한 것처럼, 기존 CNN 망의 어느 곳이든 위치할 수 있기 때문에 L-RNN을 다양한 위치에 추가한 뒤에 그 성능을 평가하였다. 실험에 사용한 다양한 구조는 아래 표와 같다. (화면상에서 글자가 너무 작게 보이면 원래 논문 참고).
다양한 구조에 따른 classification의 성능 평가는 CIFAR-10 테스트벤치를 사용하였다.
이들이 실험에 사용하는 구조는 크게 보면 2개의 그룹으로 나눌 수 있는데, 첫번째 그룹은 A~D까지가 해당이 되며, 기본적인 CNN 망에 CNN 망을 추가하면서 맨 마지막 단에만 RNN을 넣어서 CNN 망의 깊이에 따라 어떤 양상이 나타나는지를 살펴보았다. 맨마지막에 오는 RNN 모듈은 feature-map 전체를 보기 때문에 마치 fully connected layer와 비슷한 역할을 하게 된다.
실험 결과를 보면, 간단한 망이라도 L-RNN 모듈을 추가하면 성능이 의미 있는 수준으로 좋아지며, 당연한 이야기이겠지만 CNN 망이 깊어질수록 성능이 좋아진다.
두번째 그룹은 E와 F이며, CNN과 L-RNN 모듈을 섞어 사용하는 형태이다. 중간 단계에 포함된 L-RNN 모듈이 classification 결과에 어떤 영향을 끼치는지를 확인할 수 있고, 또한 이런 구조에서 망의 깊이가 어떤 영향을 끼치는지를 확인할 수 있다. CNN과 L-RNN 모듈을 섞어 사용을 하는 경우 확실하게 성능이 개선이 되는 것을 실험 결과를 통해 확인할 수 있다.
다양한 구조에 대한 실험 결과는 아래와 같다. 구조에 따른 파라미터의 개수, layer 수, 연산 시간 및 에러율을 고려하면, L-RNN 모듈을 어떻게 사용을 할 것인지 감이 온다.
위 결과를 보면 “F”의 경우 파라미터, 연산시간, 에러율 관점에서 아주 훌륭하다는 것을 확인할 수 있다. DenseNet이 가장 뛰어난 결과를 보였지만, 파라미터의 개수 및 망의 깊이까지 전체적인 고려를 해본다면 L-RNN의 효과가 매우 뛰어나다는 것을 쉽게 알 수 있다.
L-RNN의 성능 평가 (Semantic Segmentation)
L-RNN 모듈은 위치에 상관없이 삽입이 가능하기 때문에, classification뿐만 아니라 semantic segmentation 망에도 적용이 가능하다.
이들은semantic segmentation의 새로운 기념비를 쓴 FCN에 L-RNN 모듈을 적용하여 결과가 L-RNN이 추가됨에 따라 성능이 어떻게 달라지는지를 확인하였으며, 사용한 네트워크는 아래 그림과 같다.
FCN에서는 VGG16의 후반부에 오는 fully connected layer를 1x1 convolution으로 보는 방식을 취하여 영상 크기의 제한을 받지 않을 뿐만 아니라 픽셀 레벨의 예측이 가능하게 되었다. 그들은 이 부분에 L-RNN 모듈을 적용하였으며, L-RNN을 통해 feature-map 전체를 살펴볼 수가 있기 때문에 마치 CRF를 사용한 것과 같은 효과를 얻을 수 있게 된다.
실험 결과는 아래 표와 같다. L-RNN 모듈을 추가함에 따라 정확도뿐만 아니라 평균 IOU가 크게 개선되는 것을 확인할 수 있다.
Classification과 semantic segmentation에 L-RNN 모듈을 추가함으로써 성능이 크게 개선이 되는 것을 확인할 수 있었다.
결국 RNN이 순환적인 성격을 이용해 feature-map 전부를 고려할 수 있다는 점은 CRF와 일맥상통하며, CNN과의 결합을 통하여 CNN과 RNN이 갖고 있는 장점을 극대화 함으로써 성능 향상(망의 깊이, 파라미터 수, 연산시간)을 꾀할 수 있게 되었다.
지금까지 여러 Class에 걸쳐, 다양한 방법으로 semantic segmentation 방법에 대하여 살펴보았다. 이 정도면 semantic segmentation의 연구경향을 충분히 파악할 수 있을 것 이라고 생각한다.
다음 Class부터 딥러닝의 hot-topic 중 하나인 GAN(Generative Adversarial Network)에 대하여 살펴볼 예정이다.​
