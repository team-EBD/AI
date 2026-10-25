"""다중 이미지 내부 API 계약 — 외부 호출 없이 검증한다."""
import asyncio
import json

import pytest

from app.services import vision

URLS = ['http://img/first', 'http://img/second']


def test_two_images_are_ordered_parts_in_one_call(client, set_gemini, monkeypatch):
    downloads, calls = [], []
    async def download(url, timeout, max_bytes):
        downloads.append(url)
        return url.encode(), 'image/jpeg'
    monkeypatch.setattr(vision, '_download_image', download)
    stub = set_gemini(text=json.dumps({'candidates': [
        {'food_name': '김밥', 'image_index': 0},
        {'food_name': '김밥', 'image_index': 1},
    ]}))
    from app import gemini_client
    async def generate(model, contents, **kwargs):
        calls.append(contents)
        return await stub.generate_json(model, contents, **kwargs)
    monkeypatch.setattr(gemini_client, 'generate_json', generate)
    monkeypatch.setattr(gemini_client, 'image_part', lambda data, mime: data)
    result = client.post('/internal/analyze', json={
        'image_url': 'http://ignored', 'image_urls': URLS, 'max_foods': 10, 'user_text': '코스 요리',
    }).json()
    assert result['status'] == 'success'
    assert downloads == URLS
    assert len(calls) == 1
    assert calls[0][1:3] == [u.encode() for u in URLS]
    assert len(calls[0]) == 4
    assert '최대 10개까지' in calls[0][0]
    assert 'box_2d 는 반드시 그 이미지 기준' in calls[0][0]
    assert [(c['image_index'], c['food_index']) for c in result['candidates']] == [(0,0), (1,1)]


@pytest.mark.parametrize('index', [0, 1, 2])
def test_download_failure_aborts_entire_request(client, set_gemini, monkeypatch, index):
    urls = URLS + ['http://img/third']
    async def download(url, *args):
        if url == urls[index]:
            raise RuntimeError('다운로드 실패')
        return b'img', 'image/jpeg'
    monkeypatch.setattr(vision, '_download_image', download)
    stub = set_gemini(text='{}')
    result = client.post('/internal/analyze', json={'image_urls': urls}).json()
    assert result['reason'] == 'provider_error'
    assert result['failed_image_index'] == index
    assert stub.last_contents is None


@pytest.mark.parametrize('raw,reason', [([], 'not_food'), (None, 'invalid_response'), ('bad', 'invalid_response'), ([{}], 'invalid_response')])
def test_empty_and_invalid_model_responses(client, set_gemini, stub_download, raw, reason):
    stub_download()
    set_gemini(text=json.dumps({'candidates': raw}))
    result = client.post('/internal/analyze', json={'image_urls': URLS}).json()
    assert result['reason'] == reason
    assert result['failed_image_index'] is None


def test_only_second_photo_has_food(client, set_gemini, stub_download):
    stub_download()
    set_gemini(text=json.dumps({'candidates': [{'food_name': '밥', 'image_index': 1}]}))
    result = client.post('/internal/analyze', json={'image_urls': URLS}).json()
    assert result['status'] == 'success'
    assert [c['image_index'] for c in result['candidates']] == [1]


@pytest.mark.parametrize('index', [-1, 2, 99, None, True, 'bad'])
def test_invalid_image_index_falls_back_to_zero(client, set_gemini, stub_download, index):
    stub_download()
    set_gemini(text=json.dumps({'candidates': [{'food_name': '밥', 'image_index': index}]}))
    result = client.post('/internal/analyze', json={'image_urls': URLS}).json()
    assert result['candidates'][0]['image_index'] == 0


@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('depth,cap', [('standard', 3), ('clarifier', 4)])
def test_food_and_alternative_limits(client, set_gemini, stub_download, count, depth, cap):
    stub_download()
    set_gemini(text=json.dumps({'candidates': [
        {'food_name': f'음식{i} 후보{alternative}', 'food_index': i, 'confidence': .9, 'image_index': i % count}
        for i in range(16) for alternative in range(5)
    ]}))
    result = client.post('/internal/analyze', json={
        'image_urls': (URLS + ['http://img/third'])[:count], 'max_foods': count * 5, 'candidate_depth': depth,
    }).json()
    assert len(result['candidates']) == count * 5 * cap


@pytest.mark.parametrize('payload', [{}, {'image_urls': []}, {'image_urls': ['x'] * 4}, {'image_url': 'x', 'max_foods': 16}, {'image_url': 'x', 'max_foods': 0}])
def test_request_bounds(client, payload):
    assert client.post('/internal/analyze', json=payload).status_code == 422


@pytest.mark.parametrize('exc,reason', [(asyncio.TimeoutError(), 'ai_timeout'), (RuntimeError(), 'provider_error')])
def test_model_failures_are_not_attributed_to_one_photo(client, set_gemini, stub_download, exc, reason):
    stub_download()
    set_gemini(exc=exc)
    result = client.post('/internal/analyze', json={'image_urls': URLS}).json()
    assert result['reason'] == reason
    assert result['failed_image_index'] is None


def test_legacy_url_and_single_url_array_are_identical(client, set_gemini, stub_download, monkeypatch):
    stub_download()
    monkeypatch.setattr(vision.time, 'perf_counter', lambda: 1)
    set_gemini(text=json.dumps({'candidates': [{'food_name': '밥'}]}))
    legacy = client.post('/internal/analyze', json={'image_url': URLS[0]})
    modern = client.post('/internal/analyze', json={'image_urls': URLS[:1]})
    assert legacy.content == modern.content


def test_text_only_food_cannot_borrow_photo_bbox(client, set_gemini, stub_download):
    stub_download()
    set_gemini(text=json.dumps({'candidates': [
        {'food_name': '밥', 'food_index': 0, 'box_2d': [100,100,500,500]},
        {'food_name': '라면', 'food_index': 0, 'is_in_image': False, 'box_2d': [100,100,500,500]},
    ]}))
    result = client.post('/internal/analyze', json={'image_urls': URLS, 'user_text': '라면도 먹음'}).json()
    assert result['candidates'][1]['is_in_image'] is False
    assert result['candidates'][1]['bbox'] is None
    assert result['candidates'][1]['food_index'] == 1
