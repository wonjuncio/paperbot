# PaperBot

학술 저널 RSS 피드에서 논문을 자동 수집하고, AI로 관심 논문을 추천하며, Markdown / BibTeX / CSV로 내보내는 도구.

## 설치

```bash
# uv 설치 (없는 경우)
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치
uv sync
```

## 실행

```bash
uv run paperbot
```

브라우저에서 `http://127.0.0.1:8001` 대시보드가 자동으로 열린다.

## 초기 설정

첫 실행 후 **우측 상단 Preferences**에서 아래 항목을 설정한다.

| 설정 | 용도 | 필수 |
|:-----|:-----|:----:|
| **Journal List** | 논문을 수집할 저널 RSS 피드 등록. 최소 1개 이상 추가해야 Fetch 가능 | `필수` |
| **Email** | Crossref API 응답 속도 향상 (polite pool 적용) | `선택` |
| **LLM Profiles** | AI 챗봇 및 논문 추천 기능에 사용할 LLM API 키 등록 | `선택` |

## 워크플로우

1. **Fetch** — 사이드바 "Fetch Papers"로 RSS 피드에서 논문 수집
2. **Browse** — New 탭에서 논문 확인. AI 매칭 점수 badge로 관심도 파악
3. **Pick** — 관심 논문 체크박스 선택
4. **Mark as Read** — Export 드롭다운에서 "Mark as Read"로 선택된 논문을 Read 서재로 이동 (AI 추천의 기반이 됨)
5. **Export** — Export 드롭다운의 Read Library 섹션에서 Read 논문을 BibTeX / Markdown / CSV로 내보내기
6. **AI Chat** — 우측 하단 챗봇 아이콘으로 서재 논문에 대해 질문 (Preferences에서 LLM API 키 설정 필요)
7. **Semantic** — 상단 Semantic 탭에서 논문 간 의미적 관련도를 시각적으로 탐색

## 탭 구성

| 탭 | 내용 |
|-----|------|
| **New** | 새로 수집된 논문 |
| **Picked** | Export 대상으로 선택된 논문 |
| **Archive** | 보관 처리된 논문 |
| **Read** | Export 완료된 논문 (AI 서재) |
| **All** | 전체 논문 |

## CLI

대시보드 없이 터미널에서도 사용 가능:

```bash
uv run paperbot fetch          # 논문 수집
uv run paperbot list           # 새 논문 목록
uv run paperbot pick 1 2 3     # 논문 선택
uv run paperbot export         # Markdown 내보내기
```

# TODO
<details>

- db 동기화
- LLM chatbot - (short term: RAG, long term: GraphOntology)
- 레이지 로딩(Lazy Loading) & 가상 스크롤 (기능 충돌은 없는지 - 선택된 논문 유지 기능 / fetch query 개수 등)
- DB 용량을 게이지 바 + auto cleanup?
- AI Ranking: k-means cluster centroid — 다분야 서재(예: ML+생물학+경제학)에서 단일 centroid가 의미 희석되는 문제 대응. read papers를 k개 클러스터로 나눈 뒤 가장 가까운 클러스터의 centroid를 사용하면 분야별 정밀도 향상 가능

</details>

# 기여
오류 발견이나 기능 제안이 있으면 이메일로 알려주시거나 [Pull Request](https://github.com/wonjunchoii/paperbot/pulls)를 보내주세요.
