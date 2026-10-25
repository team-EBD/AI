# 실측 음식 평가 후속 설계

이 폴더의 59개 검사는 모델 출력의 구조와 계산 입력을 검증한다. 실제 사진 인식 및 중량·열량 오차를 측정하려면 별도의 정답이 필요하다. AI 저장소에는 이번 검토 시 익명 실측 사진-중량-영양 정답셋이 없어 해당 개선치를 산출하지 않았다. **결과 확인후 추가 혹은 수정필요**.

## 수집과 분할

식별정보를 제거하고 사용 권한이 확인된 사진만 별도 접근제한 저장소에 둔다. Git에는 공개 가능한 익명 manifest와 집계만 저장한다. 동일 접시의 여러 사진, 동일 제품·개인의 반복식사는 학습/튜닝과 최종 평가에 걸쳐 섞지 않는다. 튜닝셋과 미사용 최종 평가셋을 먼저 고정하고 두 버전에 동일 입력·조건을 적용한다.

평가 층: 사진만/사진+정량텍스트/텍스트만, 단품/복합 한식, 국물 포함, 포장 낱개/다중팩, 제로/오리지널, 개수 0.5/1/복수, 100g/100ml, 실측량 있음/없음, DB 정확매칭/미매칭. 표본 수와 각 층의 분모를 함께 보고한다. 운영 데이터로 얻은 실패 표본만으로 전체 정확도를 추정하지 않는다.

정답에는 저울 실측 중량, 먹기 전후 차이, 포장 동일성(바코드 또는 정확한 상품 ID), 표시 영양표의 기준량/단위와 출처, 두 검수자의 음식명 동의 여부를 기록한다. ml와 g는 밀도가 없으면 직접 비교하지 않는다. 국물·소스 섭취량을 모르면 그 항목은 오차 분모에서 제외하고 제외 건수를 명시한다.

## 익명 JSONL 스키마 예

아래는 설명용 가상 1건이며 이번 실측 결과가 아니다. `sample_id`는 사용자를 식별하지 않는 임의 ID이고 `image_ref`는 승인된 별도 저장소 객체의 익명 키다.

```json
{"sample_id":"synthetic-example-only","split":"held_out","stratum":"packaged_zero","image_ref":"private-dataset/object-id","input_mode":"photo","truth":{"canonical_food_id":"cola_zero","accepted_names":["제로 콜라"],"amount":177.5,"amount_unit":"ml","calories_kcal":0,"evidence":"package_label","nutrition_basis_amount":355,"nutrition_basis_unit":"ml","printed_total_kcal":0},"runs":[{"version":"baseline","model":"record-exact-model-version","prompt_sha256":"record-hash","status":"success","predicted_food_id":"cola_zero","predicted_name":"제로 콜라","amount":177.5,"amount_unit":"ml","calories_kcal":0,"latency_ms":1200,"label_applied":false},{"version":"candidate","model":"same-model-version","prompt_sha256":"record-hash","status":"success","predicted_food_id":"cola_zero","predicted_name":"제로 콜라","amount":177.5,"amount_unit":"ml","calories_kcal":0,"latency_ms":1250,"label_applied":false}],"user_outcome":{"draft_id":"anonymous-id","saved_within_minutes":30,"saved":true,"nutrition_edited":false,"name_edited":false,"amount_edited":false,"correction_seconds":0}}
```

한 이미지에 여러 음식이면 음식별 `truth`와 매칭된 예측을 배열로 저장하고, 누락·추가 음식을 별도 기록한다. 후보가 모두 실패한 경우에도 `status`와 실패 사유를 남겨 성공한 사진만 평가하지 않도록 한다. 표시 이름이 같다는 이유만으로 상품 ID를 같다고 간주하지 않는다.

## 지표와 분모

| 지표 | 산식·보고 방식 |
| --- | --- |
| Top-1 메뉴 정확도 | 올바른 1순위 음식 수 / 평가대상 정답 음식 수. 추론 실패·음식 누락도 오답 포함. 혼합식은 음식별 매칭 규칙을 사전 고정 |
| 음식 검출 누락/추가 | 매칭 안 된 정답 음식 수와 허위 추가 음식 수를 별도 집계 |
| 중량 MAE | 같은 g 또는 ml 단위이고 실측량이 있는 쌍의 `mean(abs(predicted-truth))`; g와 ml는 분리 |
| 열량 MAE | 정답 열량을 확인한 쌍의 kcal 절대오차 평균, 0kcal도 포함 |
| 열량 MAPE | `truth_kcal > 0`인 쌍에서 `mean(abs(predicted-truth)/truth)*100`; 0kcal는 제외 건수와 MAE를 따로 보고 |
| ±15% 이내 비율 | 양의 정답 열량 중 상대오차≤0.15인 쌍 / 해당 비교 가능 쌍. 실패율을 나란히 표시 |
| 모델 성공률 | 유효 응답 수 / 전체 호출 수. 정확도와 구별 |
| 사용자 보정률 | 초안 ID로 연결된 저장 기록 중 이름·량·영양 중 하나라도 수정한 수 / 연결된 저장 수 |
| 초안 채택률 | 정의한 시간창 내 수정없이 저장된 초안 / 연결·판정 가능한 초안. 미연결·판정불가를 별도 분모와 함께 보고 |
| 지연 | 전체 요청 및 모델/검색 구간 각각 p50/p95; timeout은 별도 비율. 단위테스트 7.40초를 운영 지연으로 사용하지 않음 |
| 검색 신뢰 | 제품/변형/영양표가 정답과 일치한 label / 적용 label; label 적용률도 나란히 보고 |

버전별 동일 표본의 차이를 paired 비교한다. 표본 수가 충분하면 이미지/상품 그룹 단위 부트스트랩 구간을 제시하고 동일 음식의 여러 사진을 독립 표본처럼 세지 않는다. 모델 버전·프롬프트 해시·검색 사용 여부·DB 스냅샷·입력 크기·재시도 횟수를 기록한다. 저장된 모델 응답 재생은 후처리 효과만 분리하고, 실제 모델 호출 비교는 비용과 데이터 사용 허용 범위를 확인한 별도 실행에서 수행한다.

## 미결 항목

- 사진 단독 정확도·29.4% 당시 오차와 현재 수치의 동일 조건 재평가: 결과 확인후 추가 혹은 수정필요.
- 중량·제품 추정 신뢰도와 실제 오차의 calibration: 결과 확인후 추가 혹은 수정필요.
- 사용자 보정 부담, 초안 이탈 이유, 추천 만족도·리텐션: 결과 확인후 추가 혹은 수정필요.
- 동일 상품명/variant의 역방향 오매칭과 URL만 있는 잘못된 검색표: 결과 확인후 추가 혹은 수정필요.
