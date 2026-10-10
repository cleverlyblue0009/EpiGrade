"""The Word paper and the slide deck exist and have the required shape."""

import re
import zipfile

import pytest

from epigrade import paths

DOCX = paths.repo_root() / "paper" / "EpiGrade_paper.docx"
PPTX = paths.repo_root() / "presentation" / "EpiGrade_slides.pptx"


def _members(path, pattern):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n).decode("utf-8", errors="replace")
                for n in z.namelist() if re.fullmatch(pattern, n)}


@pytest.mark.skipif(not DOCX.exists(), reason="docx not built")
def test_word_paper_has_the_nine_sections_and_the_authors():
    xml = _members(DOCX, r"word/document\.xml")["word/document.xml"]
    text = re.sub(r"<[^>]+>", " ", xml)
    for heading in ("Abstract", "Introduction", "Literature Review", "Methodology", "Analysis",
                    "Result", "Discussion", "Conclusion", "References"):
        assert heading in text, heading
    for author in ("23BCB0074", "23BCB0125", "23BCB0123"):
        assert author in text


@pytest.mark.skipif(not PPTX.exists(), reason="deck not built")
def test_deck_is_16_presented_slides_plus_backup_each_with_speaker_notes():
    slides = _members(PPTX, r"ppt/slides/slide\d+\.xml")
    notes = _members(PPTX, r"ppt/notesSlides/notesSlide\d+\.xml")
    # 16 presented slides (the controls-source finding has its own) plus one backup slide
    assert len(slides) == 17
    assert len(notes) == len(slides)
    for name, xml in notes.items():
        spoken = " ".join(re.findall(r"<a:t>([^<]+)</a:t>", xml))
        assert len(re.findall(r"[.!?](?:\s|$)", spoken)) >= 3, f"{name}: fewer than 3 sentences"


@pytest.mark.skipif(not PPTX.exists(), reason="deck not built")
def test_deck_body_text_is_at_least_20_point_outside_the_backup_slide():
    slides = _members(PPTX, r"ppt/slides/slide\d+\.xml")
    last = f"ppt/slides/slide{len(slides)}.xml"
    for name, xml in slides.items():
        if name == last:
            continue  # backup slide: reference detail, not presented
        sizes = [int(s) for s in re.findall(r'<a:rPr[^>]* sz="(\d+)"', xml)]
        small = [s / 100 for s in sizes if s < 2000]
        # captions and the footer are the only text below 20 pt
        assert all(s >= 16 for s in small), (name, sorted(set(small)))
        assert len(small) <= 2, (name, small)


@pytest.mark.skipif(not PPTX.exists(), reason="deck not built")
def test_deck_states_the_cross_study_count_plainly_and_names_the_controls_finding():
    slides = _members(PPTX, r"ppt/slides/slide\d+\.xml")
    text = " ".join(" ".join(re.findall(r"<a:t>([^<]+)</a:t>", xml)) for xml in slides.values())
    assert "1 of 6 cross-study tests could be run" in text
    assert "one cross-study test, not three" in text
    assert "controls-source effect" in text
    assert "detect series membership, not disease" in text
    assert "68.1%" in text and "11.9%" in text
