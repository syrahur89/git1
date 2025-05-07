#패션 이미지 기반 성별 & 스타일 예측 프로젝트

## 사용 데이터 구조
fashion_data/
├── sample/
    ├── raw_data/
    │   ├── 1.training/
    │   ├── 2.validation/
    │   └── 3.test/
    └── labels_data/
        ├── 1.training/
        └── 2.validation/
```

## Mission 1-1: 성별 & 스타일 이미지 수 통계

- 이미지 파일명은 다음과 같은 형식식: `{W/T}_{이미지ID}_{시대}_{스타일}_{성별}.jpg`
- 이를 기준으로 성별 + 스타일 조합별 이미지 수를 집계
- 결과는 Pandas DataFrame으로 출력

## 1-2: ResNet-18 기반 성별&스타일 분류기

- ResNet-18 사용
- `pretrained=False` 조건으로 가중치 무작위 초기화
- `Top-1 Accuracy` 기준으로 성능 측정
- 이미지 전처리: `Resize(128x128)`, `Normalize`
- 에폭별 Loss, Accuracy 출력


Epoch 1: Loss = 1.4671, Accuracy = 45.32%
...
Epoch 5: Loss = 1.0023, Accuracy = 62.13%

모든 Acuuracy는 Top-1 Accuracy 기준으로 학습


## Mission 2-1: 라벨링 JSON 기반 성별&스타일 설문 통계
- JSON 파일명: `{W/T}_{이미지ID}_{시대}_{스타일}_{성별}_{설문ID}.json`
- 유효한 이미지 ID 기준으로만 통계 작성
- 결과: 각 성별-스타일 조합의 설문 응답 수 집계


## 2-2: 응답자별 스타일 선호 정보표 생성
- JSON 내부 필드:
  - `user>R_id`: 사용자 ID
  - `item>survey>Q5`: 스타일 선호 여부 (1: 비선호, 2: 선호)
- `preference_df`라는 Pandas DataFrame 형태로 저장
  - 행: 응답자 ID
  - 열: 스타일 이미지명
  - 값: Q5 응답 (1 or 2)


## Mission 3-1: 협업 필터링 개념 설명 
Validation 데이터 내 응답자의 스타일 선호 여부를 예측하기 위해 협업 필터링(Collaborative Filtering)의 두 가지 주요 방식(User-based, Item-based)을 이해하고, 그 차이점과 장단점을 설명

## User-based Collaborative Filtering

- 정의  
  나와 비슷한 취향을 가진 사용자들이 선호한 스타일을 기반으로 예측하는 방식

- 예시  
  내가 스타일 A, B를 선호했고, 나와 유사한 응답자 X가 스타일 C도 좋아하는 경우우
  → 나도 스타일 C를 선호할 가능성이 높다고 판단

- 장점
  - 사용자 간 유사성이 뚜렷할 경우 성능이 높다
  - 개개인의 다양성을 반영한 예측 가능

- 단점
  - 사용자 수나 평가 수가 적을 경우 희소성 문제 발생할 수 있음
  - 새로운 사용자의 경우 정보 부족 가능성 있음


## Item-based Collaborative Filtering

- 정의  
  사용자가 선호한 스타일과 유사한 스타일을 기반으로 예측하는 방식

- 예시  
  많은 사용자가 스타일 A와 B를 동시에 좋아한다면, 스타일 A를 좋아하는 사람은 B도 선호할 가능성이 높다고 판단

- 장점
  - 아이템 간 유사도는 비교적 안정적이고 계산 비용이 낮음
  - 새로운 사용자에게도 예측 가능하다 (기존 아이템 기반이기 때문)

- 단점
  - 아이템 간 유사도를 정의하기가 어려울 수 있음
  - 새로운 아이템에는 적용 어려움


## 결론  
- User-based는 사용자 취향 유사성 기반  
- Item-based는 스타일 간 유사성 기반  
- 이 과제에서는 Item-based Filtering이 더 적합하다고 판단하였다.

---

## 3-2: Item-based Filtering 구현 및 예측

- `preference_df`를 기반으로 코사인 유사도 계산
- 예측 점수는 다음 수식 기반:

\[
\hat{r}_{ui} = \frac{\sum_{j \in N(i)} sim(i,j) * r_{uj}}{\sum_{j \in N(i)} |sim(i,j)|}
\]

- 예측 예시 (`user_1`, 일부 항목 비워둔 상태):

```
W_00001_80_style_W.jpg: 예측 점수 1.3391 → 비선호 예측
W_00002_80_style_W.jpg: 예측 점수 1.3372 → 비선호 예측
W_00003_90_style_W.jpg: 예측 점수 1.3369 → 비선호 예측
```


