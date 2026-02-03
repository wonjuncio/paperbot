# PaperBot

학술 저널 RSS 피드에서 논문을 수집하고, Crossref로 메타데이터를 보강한 뒤, Zotero에 자동으로 추가하는 도구

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

2. `.env` 편집:

```
CONTACT_EMAIL=you@example.com
ZOTERO_API_KEY=your_api_key
ZOTERO_LIBRARY_ID=your_library_id
ZOTERO_LIBRARY_TYPE=user
ZOTERO_COLLECTION_KEY=optional_collection_key
```

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

```bash
# RSS 피드에서 논문 수집
uv run paperbot fetch

# 새 논문 목록 보기
uv run paperbot list

# 논문 선택
uv run paperbot pick 1 2 3

# 선택 취소 (picked 상태인 논문만 new로 되돌림)
uv run paperbot unpick 2 3 4

# Zotero에 업로드
uv run paperbot push-zotero
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

# 업로드 완료된 논문 보기
uv run paperbot list --status pushed

# 표시 개수 제한
uv run paperbot list --limit 20
```

### `pick`
관심있는 논문을 선택 (ID로 지정). 선택된 논문만 Zotero에 업로드됨.

```bash
uv run paperbot pick 1 2 3
```

### `unpick`
선택을 취소 (해당 ID를 `new` 상태로 되돌림). **현재 `picked` 상태인 논문만** 취소됨. 지정한 ID 중 picked가 하나도 없으면 "None of the given IDs are in picked status" 알림이 뜸.

```bash
uv run paperbot unpick 2 3 4
```

### `push-zotero`
선택된(picked) 논문을 Zotero 라이브러리에 업로드.

```bash
uv run paperbot push-zotero
```

## 워크플로우

1. `fetch` → 피드에서 논문 수집
2. `list` → 새 논문 확인
3. `pick` → 관심 논문 선택
4. `push-zotero` → Zotero에 추가

# TODO
- Zotero Test (O, pdf 받기 기능 X)
- `pick` Test (O)
- `unpick` 제작 (O)
- UI 제작
- list 조건 넣기 / n days 마다 받아오기

# 기여
오류 발견이나 기능 제안이 있으면 이메일로 알려주시거나 [Pull Request](https://github.com/wonjuncio/paperbot/pulls)를 보내주세요.