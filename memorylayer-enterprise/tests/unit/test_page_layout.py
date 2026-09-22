# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Image/transcript-bound layout survives ingestion without another OCR call."""
import base64
import hashlib
import io
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from memorylayer_saas.services.document.page_layout import layout_metadata, store_page_layout
from memorylayer_saas.services.transcription.result import pages_from_embed_server_response, TranscribedPage
from memorylayer_saas.services.transcription.regions import PageRegion

RAW = '<|ref|>text<|/ref|><|det|>[[100,200,800,400]]<|/det|>Service is optional.\n<|ref|>table<|/ref|><|det|>[[100,500,900,900]]<|/det|><table><tr><td>25</td></tr></table>'


def image_bytes():
    buffer = io.BytesIO()
    Image.new('RGB', (100, 200), 'white').save(buffer, format='PNG')
    return buffer.getvalue()


def page():
    return pages_from_embed_server_response({'results':[{'success':True,'page_index':0,'content':'unused',
        'raw_content':RAW,'output_contract':'deepseek_ocr','provider_used':'ocr','model_used':'model'}]})[0]


def test_layout_binds_actual_pixels_and_transcript_and_keeps_text_table_regions():
    p=page();data=image_bytes();layout=layout_metadata(p,data)
    assert layout['image_sha256']==hashlib.sha256(data).hexdigest()
    assert layout['transcript_sha256']==hashlib.sha256(p.content.encode()).hexdigest()
    assert (layout['image_width'],layout['image_height'])==(100,200)
    assert [r['label'] for r in layout['regions']]==['text','table']
    assert layout['regions'][0]['text']=='Service is optional.'
    assert layout['regions'][0]['bbox']==[.1,.2,.8,.4]
    assert '<table>' in layout['regions'][1]['text']
    assert layout_metadata(p,data)==layout
    assert p.raw_content==RAW and p.provider=='ocr'


@pytest.mark.parametrize('box',[None,(-1,0,100,100),(0,0,1001,100),(100,100,0,0),(10,10,10,20)])
def test_invalid_boxes_never_become_valid_highlights(box):
    p=TranscribedPage(0,'body','model',regions=(PageRegion('text',box,'body'),))
    assert layout_metadata(p,image_bytes())['regions'][0]['bbox'] is None


@pytest.mark.asyncio
async def test_raw_output_persists_at_a_hash_versioned_path():
    blob=Mock();blob.store_file=AsyncMock();blob.page_raw_ocr_path.side_effect=lambda w,d,p,h:f'/raw/{w}/{d}/{p}-{h}.txt'
    p=page();result=await store_page_layout(blob_storage=blob,workspace_id='ws',doc_id='doc',page_no=3,
        page_image_b64=base64.b64encode(image_bytes()).decode(),page=p)
    assert result['raw_ocr']['sha256']==hashlib.sha256(RAW.encode()).hexdigest()
    blob.store_file.assert_awaited_once_with(result['raw_ocr']['storage_path'],RAW.encode())
    blob.store_file.side_effect=OSError('storage unavailable')
    with pytest.raises(OSError):
        await store_page_layout(blob_storage=blob,workspace_id='ws',doc_id='doc',page_no=3,
            page_image_b64=base64.b64encode(image_bytes()).decode(),page=p)


def test_raw_transcript_format_still_supported_and_region_ids_change_with_content():
    old=pages_from_embed_server_response({'results':[{'success':True,'page_index':0,'content':'text'}]})[0]
    assert old.raw_content=='text'
    first=layout_metadata(page(),image_bytes())
    changed=page();object.__setattr__(changed,'regions',(PageRegion('text',(100,200,800,400),'Changed.'),))
    assert first['regions'][0]['id']!=layout_metadata(changed,image_bytes())['regions'][0]['id']


def test_quote_locator_consumes_persisted_ingestion_layout_and_expected_image():
    from types import SimpleNamespace
    from memorylayer_saas.services.document.page_layout import locate_page_quote
    p=page();layout=layout_metadata(p,image_bytes())
    stored=SimpleNamespace(transcript=p.content,metadata={'ocr_layout':layout})
    result=locate_page_quote(stored,'Service is optional.',image_sha256=layout['image_sha256'])
    assert result==[dict(region_id=layout['regions'][0]['id'],image_sha256=layout['image_sha256'],
                        bbox=[.1,.2,.8,.4],origin='ocr')]
    assert locate_page_quote(stored,'Service is optional.',image_sha256='0'*64)==[]
    stored.transcript += ' changed'
    assert locate_page_quote(stored,'Service is optional.')==[]
