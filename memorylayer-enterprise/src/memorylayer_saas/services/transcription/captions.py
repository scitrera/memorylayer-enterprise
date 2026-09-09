"""Context-grounded captions for figures a grounded OCR model leaves silent.

The OCR model marks an illustration with a box and emits no text, so the
transcript records only that *something* was there. A caption closes that gap:
the transcript then says what the figure contains, and a downstream consumer can
decide from the text alone whether it needs to fetch the actual crop.

Captions are generated from the crop **plus the surrounding page text**, because
an ungrounded caption of a document figure is nearly useless -- "a photograph of
a building" versus "exterior of the Macaw Landing Grille dining outlet". The
neighbouring title and body regions supply that grounding, and they are already
in hand from the layout parse.

Routing goes through the standard **LLM profiles** mechanism
(``LLMService.complete(request, profile=...)``), not the document-chat
prompt-embeds client: this is an ordinary multimodal completion, and the default
profile is a multimodal OpenAI-compatible model. ``LLMMessage.images`` is the
vision seam -- OpenAI-compatible providers serialize it into the multimodal
content-block list, and providers without vision ignore it.
"""

from __future__ import annotations

import asyncio
import base64
from logging import Logger

from memorylayer_server.models.llm import LLMMessage, LLMRequest, LLMRole
from memorylayer_server.models.generation import GenerationActivity

from .provider import DEFAULT_TRANSCRIBE_CONCURRENCY
from .regions import LABEL_TEXT, LABEL_TITLE, PageRegion

DEFAULT_FIGURE_CAPTION_PROFILE = "default"
#: Thinking is suppressed for captions (see DEFAULT_FIGURE_CAPTION_REASONING_EFFORT),
#: after which a caption costs ~25 tokens. The budget is nonetheless generous: it
#: is a bound on the damage if a provider ignores reasoning_effort and reasons
#: anyway, not a target. A caption-sized cap truncates such a model mid-thought
#: and yields raw chain-of-thought instead of a sentence.
DEFAULT_FIGURE_CAPTION_MAX_TOKENS = 512

#: Captioning is a description task, not a reasoning task, and the default
#: profile is a thinking model whose chain-of-thought lands in ``content``
#: (``reasoning_content`` comes back empty, so nothing downstream can separate
#: it). Measured on the live default profile: with thinking on, 6 of 7 captions
#: were raw chain-of-thought at ~700-1024 tokens each; with "none" they are
#: clean sentences at ~21 tokens.
#:
#: This is the portable knob -- OpenAI-compatible providers pass it through
#: (Fireworks qwen3 honors it while rejecting ``enable_thinking``), and the
#: Anthropic/Google providers map unknown values to a default thinking budget
#: rather than erroring, so an unsupported provider degrades to "still thinks"
#: instead of failing. Set the env key to empty to omit the field entirely.
DEFAULT_FIGURE_CAPTION_REASONING_EFFORT = "none"
DEFAULT_FIGURE_CAPTION_CONTEXT_CHARS = 600
DEFAULT_FIGURE_CAPTION_TEMPERATURE = 0.2

#: Constrained on purpose. The caption goes into the transcript, gets embedded,
#: and is read by an LLM later -- so it must be short, factual, and free of the
#: "This image shows..." preamble that wastes tokens in every single one.
DEFAULT_FIGURE_CAPTION_INSTRUCTION = (
    "This is a figure taken from a document page. Using the surrounding page text "
    "above only as context, write ONE short factual sentence (at most 25 words) "
    "describing what the figure actually depicts. Describe only what is visible. "
    "Do not begin with 'This figure', 'This image', or 'The image shows'. "
    "If the figure is a chart, say what it plots. Reply with the sentence alone."
)

#: Labels whose text is worth feeding to the captioner as context.
_CONTEXT_LABELS = (LABEL_TITLE, LABEL_TEXT)


def build_figure_context(
    regions: list[PageRegion] | tuple[PageRegion, ...],
    figure_index: int,
    *,
    max_chars: int = DEFAULT_FIGURE_CAPTION_CONTEXT_CHARS,
) -> str:
    """Collect page text around ``regions[figure_index]`` as caption context.

    Walks outward from the figure, nearest first, so the closest caption-like
    text wins the budget. The most recent preceding title is always included --
    it is usually the section the figure illustrates.
    """
    parts: list[str] = []
    budget = max_chars

    for region in reversed(regions[:figure_index]):
        if region.label == LABEL_TITLE and region.content:
            parts.append(region.content)
            budget -= len(region.content)
            break

    before = [r for r in regions[:figure_index] if r.label in _CONTEXT_LABELS and r.content]
    after = [r for r in regions[figure_index + 1:] if r.label in _CONTEXT_LABELS and r.content]

    for region in list(reversed(before)) + after:
        if budget <= 0:
            break
        if region.content in parts:
            continue
        parts.append(region.content[:budget])
        budget -= len(region.content)

    return "\n".join(parts).strip()


class FigureCaptioner:
    """Caption figure crops via the configured LLM profile.

    ``llm_service`` is the standard ``LLMService`` (``EXT_LLM_SERVICE``);
    ``profile`` selects the provider profile, and ``model`` is an optional
    override left ``None`` so the profile's own default model applies.
    """

    def __init__(
        self,
        *,
        llm_service,
        logger: Logger,
        profile: str = DEFAULT_FIGURE_CAPTION_PROFILE,
        model: str | None = None,
        instruction: str = DEFAULT_FIGURE_CAPTION_INSTRUCTION,
        max_tokens: int = DEFAULT_FIGURE_CAPTION_MAX_TOKENS,
        context_chars: int = DEFAULT_FIGURE_CAPTION_CONTEXT_CHARS,
        temperature: float = DEFAULT_FIGURE_CAPTION_TEMPERATURE,
        reasoning_effort: str | None = DEFAULT_FIGURE_CAPTION_REASONING_EFFORT,
        concurrency: int = DEFAULT_TRANSCRIBE_CONCURRENCY,
    ):
        self._llm = llm_service
        self.logger = logger
        self.profile = profile
        self.model = model
        self.instruction = instruction
        self.max_tokens = int(max_tokens)
        self.context_chars = int(context_chars)
        self.temperature = float(temperature)
        self.reasoning_effort = reasoning_effort or None
        #: Captions in flight at once. A separate pool from page
        #: transcription: captions hit the LLM profile, pages hit the OCR
        #: lane, so one must not consume the other's budget.
        self.concurrency = max(1, int(concurrency))

    def build_request(self, figure_png: bytes, context: str) -> LLMRequest:
        prompt = (
            "Surrounding page text:\n%s\n\n%s" % (context, self.instruction)
            if context
            else self.instruction
        )
        return LLMRequest(
            messages=[
                LLMMessage(
                    role=LLMRole.USER,
                    content=prompt,
                    images=[base64.b64encode(figure_png).decode("ascii")],
                )
            ],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

    async def caption(self, figure_png: bytes, context: str = "") -> str | None:
        """Caption one crop. Returns ``None`` on any failure -- a missing caption
        degrades the transcript, a raised exception would lose the document."""
        try:
            response = await self._llm.complete(
                self.build_request(figure_png, context),
                profile=self.profile,
                activity=GenerationActivity.SYNTHESIS,
            )
        except Exception as e:  # noqa: BLE001 — captions must never fail an ingest
            self.logger.warning("Figure caption failed: %s", e)
            return None
        return _tidy(getattr(response, "content", None))

    async def caption_page_figures(
        self,
        regions: list[PageRegion] | tuple[PageRegion, ...],
        crops: list[tuple[PageRegion, bytes]],
    ) -> dict[int, str]:
        """Caption a page's crops, keyed by 1-based figure number.

        ``crops`` comes from :func:`.figures.crop_figures` and is in page order,
        so its ordering matches the ``[figure N]`` numbering in the transcript.
        Figures that fail to caption are simply absent from the result and keep
        their bare placeholder.
        """
        positions = {id(region): index for index, region in enumerate(regions)}
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run(region: PageRegion, data: bytes) -> str | None:
            async with semaphore:
                return await self.caption(
                    data,
                    build_figure_context(
                        regions, positions.get(id(region), 0), max_chars=self.context_chars,
                    ),
                )

        results = await asyncio.gather(*(run(region, data) for region, data in crops))
        return {
            figure_no: caption
            for figure_no, caption in enumerate(results, start=1)
            if caption
        }


def _tidy(text: str | None) -> str | None:
    """Collapse the caption to a single clean line."""
    if not text:
        return None
    caption = " ".join(text.split()).strip().strip('"')
    return caption or None
