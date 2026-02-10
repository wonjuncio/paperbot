# PaperBot

학술 저널 RSS 피드에서 논문을 수집하고, Crossref로 메타데이터를 보강한 뒤, Markdown 파일로 내보내는 도구.

## 설치 (uv 사용)

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론 후
cd paperbot

# 가상환경 생성 및 의존성 설치
uv sync

# 또는 개발 의존성 포함
uv sync --extra dev
```

## 설정

1. `.env` 파일 생성:

```bash
cp .env.example .env
```

2. `.env` 편집 (선택사항):

```
CONTACT_EMAIL=you@example.com
```

이메일은 Crossref API의 "polite pool" 사용을 위해 권장된다.

3. `feeds.yaml` 편집하여 수집할 저널 RSS 추가

프로젝트 루트의 `feeds.yaml`에 피드 목록을 YAML 형식으로 적는다.

| 필드 | 필수 | 설명 |
|------|------|------|
| `name` | 예 | 피드 이름. 목록/로그에 표시되며, Crossref 검색 시 저널 힌트로도 쓰임 |
| `url` | 예 | RSS/Atom 피드 URL. 저널 사이트에서 "RSS", "Latest articles" 링크 주소를 복사하면 됨 |
| `issn` | 아니오 | 저널 ISSN (예: `2057-3960`). RSS에 DOI가 없을 때 Crossref에서 DOI를 찾을 때 이 저널만 검색해 정확도를 높임 |

예시:

```yaml
feeds:
  - name: "npj Computational Materials - Latest Articles"
    url: "https://www.nature.com/npjcompumats.rss"
    issn: "2057-3960"

  - name: "Computational Materials Science - Latest Articles"
    url: "https://rss.sciencedirect.com/publication/science/09270256"
    issn: "0927-0256"
```

저널마다 RSS URL 형식이 다르므로, 각 저널 홈페이지에서 공식 RSS 링크를 확인한 뒤 그 주소를 `url`에 넣으면 됨.

처음 실행 시 `paperbot fetch`를 하면 프로젝트 폴더에 `papers.db`가 생성된다.

## 사용법

### GUI (웹 대시보드)

인자 없이 실행하면 Streamlit 기반 웹 GUI가 열린다.

```bash
uv run paperbot
```

브라우저에서 **PaperBot Dashboard**가 뜨며, CLI와 동일한 워크플로우를 화면에서 처리할 수 있다.

| 구성 | 설명 |
|------|------|
| **사이드바** | 상태별 논문 개수(new / picked / read / archived), "Reset All Views" 버튼 |
| **Fetch New Papers** | RSS 피드에서 새 논문 수집·보강 후 DB에 저장 (CLI `fetch`와 동일) |
| **Export Picked** | 선택된 논문을 Markdown으로 내보내기 (CLI `export`와 동일) |
| **New 탭** | 새 논문 목록 테이블. Pick 체크 후 "Apply Selection"으로 선택 반영 |
| **Picked 탭** | export 대상으로 선택된 논문 목록. 각 항목에서 "Unpick"으로 선택 해제 |
| **Archive 탭** | DB에 저장된 전체 논문 목록(상태·ID·제목·저널·발행일·DOI) |

GUI에서도 `feeds.yaml`, `papers.db`, `exports/` 경로는 CLI와 동일하게 사용된다.

### CLI (터미널)

```bash
# RSS 피드에서 논문 수집
uv run paperbot fetch

# 새 논문 목록 보기
uv run paperbot list

# 논문 선택
uv run paperbot pick 1 2 3

# 선택 취소 (picked 상태인 논문만 new로 되돌림)
uv run paperbot unpick 2 3 4

# Markdown으로 내보내기 (picked 상태 → read 상태로 변경)
uv run paperbot export
```

## 명령어

### `fetch`
RSS 피드에서 새 논문을 수집하고 Crossref API로 메타데이터(저자, 초록 등)를 보강하여 로컬 DB에 저장.

```bash
uv run paperbot fetch
```

### `list`
저장된 논문 목록을 상태별로 조회.

```bash
# 새 논문 보기 (기본값)
uv run paperbot list

# 선택된 논문 보기
uv run paperbot list --status picked

# 내보내기 완료된 논문 보기
uv run paperbot list --status read

# 표시 개수 제한
uv run paperbot list --limit 20

# 정렬 옵션 (id, date, title)
uv run paperbot list --sort date
```

### `pick`
관심있는 논문을 선택 (ID로 지정). 선택된 논문만 export됨.

```bash
uv run paperbot pick 1 2 3
```

### `unpick`
선택을 취소 (해당 ID를 `new` 상태로 되돌림). **현재 `picked` 상태인 논문만** 취소됨. 지정한 ID 중 picked가 하나도 없으면 "None of the given IDs are in picked status" 알림이 뜸.

```bash
uv run paperbot unpick 2 3 4
```

### `export`
선택된(picked) 논문을 Markdown 파일로 내보냄. `exports/` 폴더에 오늘 날짜 이름(예: `2026-02-03.md`)으로 저장되며, 같은 날 여러번 실행하면 덮어씌워짐. Export 후 논문 상태는 `read`로 변경됨.

```bash
uv run paperbot export
```

**출력 예시** (`exports/2026-02-03.md`):

```markdown
## 2026-02-03

### Machine-learning interatomic potentials for atomistic simulations
- Journal: npj Computational Materials (2024)
- DOI: 10.1038/s41524-024-01234-5
- Link: https://doi.org/10.1038/s41524-024-01234-5

### Active learning for robust ML interatomic potentials
- Journal: Physical Review Materials (2025)
- DOI: 10.1103/PhysRevMaterials.9.013801
- Link: https://doi.org/10.1103/PhysRevMaterials.9.013801
```

## 워크플로우

1. `fetch` → 피드에서 논문 수집
2. `list` → 새 논문 확인 (GUI: New 탭)
3. `pick` → 관심 논문 선택 (GUI: New 탭에서 체크 후 Apply Selection)
4. `export` → Markdown 파일로 내보내기 (GUI: Export Picked 버튼)

위 단계는 **GUI**(`uv run paperbot`) 또는 **CLI** 명령어로 진행할 수 있다.

# TODO
<details>

- READ papers export 기능 구현
- sort dropdown menu (ai matching score for new paper)
- db 동기화
- 설정 창 제작 (.env 설정용, feeds.yaml 설정용 - feeds 없으면 toast 알림?)
- LLM 연동 with api key (chatbot?)

</details>

# 기여
오류 발견이나 기능 제안이 있으면 이메일로 알려주시거나 [Pull Request](https://github.com/wonjunchoii/paperbot/pulls)를 보내주세요.