# PADS 유사 오픈 떨림 데이터셋 후보 (2026-09-15 조사)

PADS(손목 Apple Watch, acc+gyro, 100 Hz, 469명) 외에 팀원이 나눠 맡을 수 있는 후보.
공통 CSV schema는 `elapsed_ms, acc_x_g..., gyro_x_dps...`이므로 gyro 유무를 먼저 확인.

| 데이터셋 | 출처 | 접근/라이선스 | 참가자 | 센서 · Hz | 라벨 | 형식 · 용량 | 비고 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| MJFF Levodopa Response Study | [Synapse syn20681023](https://www.synapse.org/Synapse:syn20681023), [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7865022/) | Synapse 가입 + Certified User + 이용목적 제출 | PD 28 | GENEActiv/Pebble 손목 acc(gyro 없음) 50 Hz, 17명은 양손목 Shimmer | 임상의 tremor 0-4점, task별 | CSV(timestamp, x,y,z) | 50 Hz라 우리 조건과 일치. gyro 없음. 환자 데이터 |
| Monipar | [Zenodo 8104853](https://zenodo.org/records/8104853) | CC-BY-4.0, 즉시 다운로드 | PD 21 + HC 7 | 스마트워치 acc 50 Hz | 일부 대상 tremor 라벨, MDS-UPDRS | .mat 35 MB | 작고 바로 쓸 수 있음. gyro 없음 |
| Hand acceleration PD/HC (smartphone) | [Zenodo 7273759](https://zenodo.org/records/7273759) | CC-BY-4.0 | 검진 45 + 자가보고 454 | 스마트폰 acc | 대상별 tremor 0/1 | pickle 4.8 GB | 통화 중 수집, 손목 아님. 라벨이 사람 단위 |
| ALAMEDA PD tremor | [Zenodo 10782573](https://zenodo.org/records/10782573) | CC-BY-4.0 | 미기재 | GENEActiv acc, 92개 추출 특징만 | rest/kinetic/postural tremor 0/1 | CSV 5.8 MB | 원시 시계열 없음 → 우리 파이프라인 불가 |
| Parkinson@Home Validation | [논문 2025](https://www.nature.com/articles/s41531-025-01056-2) | 요청 기반, 공개 다운로드 없음 | PD 24 + HC 24 | 손목 gyro, 비디오 라벨 | 영상 기반 tremor 구간 | - | 이상적이지만 접근 불확실 |
| mPower | [Synapse](https://www.nature.com/articles/sdata201611) | Synapse 가입 + DUA | 수천 명 | 스마트폰 acc+gyro | PD 자가보고, tremor 전용 task 없음 | JSON | 손목 아님, 떨림 라벨 약함 |

## 추천 순서

1. **Levodopa Response Study** — 손목, 50 Hz, task 단위 임상 tremor 점수. gyro만 없으므로 acc-only 변형 실험 필요.
2. **Monipar** — 가장 빨리 붙일 수 있음(35 MB, CC-BY). 소규모 검증용.
3. **Zenodo 7273759** — 규모는 크지만 손목이 아니고 라벨이 사람 단위여서 recording-level 학습에 부적합.

ALAMEDA는 특징만 있어 제외. Parkinson@Home은 저자 연락 필요.
