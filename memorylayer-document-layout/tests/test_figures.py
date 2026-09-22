# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic figure navigation: no customer sources, coordinates or content."""
import copy

import pytest
from memorylayer_document_layout import locate_quote, text_sha256


def figure_page():
    rows = [
        ('header', 'Example document', [.1, 0, .8, .03]),
        ('sub_title', '## Sample artwork', [.1, .05, .8, .09]),
        ('image', '', [.1, .1, .4, .3]),
        ('image', '', [.5, .1, .8, .3]),
        ('sub_title', '## Service menu', [.1, .32, .8, .36]),
        ('image', '', [.2, .4, .6, .6]),
        ('text', 'Symbol legend', [.1, .62, .8, .66]),
        ('image', '', [.2, .7, .6, .8]),
        ('page_number', '7', [.4, .95, .6, .98]),
    ]
    text = '## Sample artwork\n\n[figure 1]\n\n[figure 2]\n\n## Service menu\n\n[figure 3]\n\nSymbol legend\n\n[figure 4]'
    layout = dict(version=1, coordinate_space='normalized_0_1', image_sha256='a'*64,
        transcript_sha256=text_sha256(text), regions=[dict(
            id=f'block-{i}', label=label, text=content, bbox=box)
            for i, (label, content, box) in enumerate(rows)])
    return text, layout


def locate(quote, text, layout, locator=''):
    return locate_quote(quote, text, layout, mode='figure', locator=locator, image_sha256='a'*64)


@pytest.mark.parametrize('quote,locator,ids', [
    ('[figure 3]', '', ['block-5']),
    ('[figure 2] and [figure 1]', '', ['block-2', 'block-3']),
    ('Visible artwork', 'Two images below the "Sample artwork" heading', ['block-2', 'block-3']),
    ('Visible menu', 'Image under the Service menu heading', ['block-5']),
    ('Visible images', 'Sample artwork and Service menu headings', ['block-2', 'block-3', 'block-5']),
    ('Symbol legend', 'Icons beneath it', ['block-7']),
    ('Visible icons', 'Figure below the "Symbol legend" caption', ['block-7']),
    ('Visible icons', '[figure 4]', ['block-7']),
])
def test_only_whole_recorded_figures_for_exact_markers_or_unique_anchors(quote, locator, ids):
    text, layout = figure_page()
    original = copy.deepcopy(layout)
    result = locate(quote, text, layout, locator)
    assert [r['region_id'] for r in result] == ids
    for r in result:
        recorded = next(b for b in layout['regions'] if b['id'] == r['region_id'])
        assert r == dict(region_id=recorded['id'], image_sha256='a'*64,
                         bbox=recorded['bbox'], origin='ocr')
    assert layout == original
    result[0]['bbox'][0] = 0
    assert layout == original


@pytest.mark.parametrize('case', [
    'untranscribed_text', 'partial_heading', 'between_figures', 'missing_marker',
    'mixed_missing_marker', 'missing_box', 'invalid_box', 'incomplete_group',
    'stale_transcript', 'wrong_image', 'missing_image', 'duplicate_id',
    'reordered_markers', 'repeated_marker', 'custom_rendering', 'duplicate_heading',
    'heading_below_figure', 'invalid_locator', 'zero_marker', 'huge_marker',
    'invalid_marker_with_anchor', 'opposite_direction',
])
def test_ambiguous_changed_or_incomplete_figure_geometry_is_not_guessed(case):
    text, layout = figure_page()
    quote, locator = 'Visible artwork', 'Below the Sample artwork heading'
    if case == 'invalid_marker_with_anchor': quote = '[figure 0]'
    if case == 'opposite_direction': locator = 'Illustrations above the Sample artwork heading'
    if case == 'untranscribed_text': quote, locator = 'A menu title visible only in pixels', ''
    if case == 'partial_heading': locator = 'Below artwork'
    if case == 'between_figures': quote, locator = 'Symbol legend', ''
    if case == 'missing_marker': quote, locator = '[figure 9]', ''
    if case == 'mixed_missing_marker': quote, locator = '[figure 1] [figure 9]', ''
    if case == 'zero_marker': quote, locator = '[figure 0]', ''
    if case == 'huge_marker': quote, locator = '[figure ' + '9'*5000 + ']', ''
    if case in ('missing_box', 'incomplete_group'): layout['regions'][3]['bbox'] = None
    if case == 'invalid_box': layout['regions'][3]['bbox'] = [0, 0, 4, 4]
    if case == 'stale_transcript': text += ' Changed.'
    if case == 'wrong_image': layout['image_sha256'] = 'b'*64
    if case == 'missing_image': layout['regions'].pop(2)
    if case == 'duplicate_id': layout['regions'][3]['id'] = 'block-2'
    if case == 'reordered_markers': text = text.replace('[figure 1]', '[figure X]').replace('[figure 2]', '[figure 1]').replace('[figure X]', '[figure 2]')
    if case == 'repeated_marker': text = text.replace('[figure 2]', '[figure 1]')
    if case == 'custom_rendering': text = text.replace('[figure 1]', '[illustration 1]')
    if case == 'duplicate_heading':
        text = text.replace('Service menu', 'Sample artwork')
        layout['regions'][4]['text'] = '## Sample artwork'
    if case == 'heading_below_figure': layout['regions'][1]['bbox'] = [.1, .8, .8, .9]
    if case == 'invalid_locator': locator = None
    if case != 'stale_transcript': layout['transcript_sha256'] = text_sha256(text)
    assert locate(quote, text, layout, locator) == []


def test_image_caption_and_title_rendering_follow_ocr_contract():
    text, layout = figure_page()
    layout['regions'][1].update(label='title', text='Sample artwork')
    layout['regions'][2]['text'] = 'Photograph caption'
    text = text.replace('[figure 1]', '[figure 1]\n\nPhotograph caption')
    layout['transcript_sha256'] = text_sha256(text)
    assert [r['region_id'] for r in locate('[figure 1]', text, layout)] == ['block-2']
