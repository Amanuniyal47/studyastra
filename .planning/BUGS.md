# Bug Tracking & Issue Register

## Active Issues

| Issue | Description | Status | Target Stage |
|---|---|---|---|
| **BUG-01** | Notes generation not working | **Resolved** | Stage 8: Hierarchical Academic Study Notes Generation |
| **BUG-02** | Flashcard generation not working | **Resolved** | Stage 7: Flashcard Generation & Interactive Study Deck |

---

## BUG-01: Notes Generation Not Working

### Description
Notes generation from uploaded PDFs is currently failing or incomplete. It must produce comprehensive, structured academic study notes organized by chapters, topics, and subtopics, extract crucial definitions and mechanisms, preserve source page references, gracefully handle large PDFs without context truncation, and allow users to view and download the notes.

### Status
`Resolved` (Stage 8 implementation complete & verified)

### Existing Implementation Status
- **Backend Module:** `src/generation/study_notes_generator.py` contains batch-aware `generate_study_notes()` with prompt-overhead-aware chunk packing within `NOTES_MAX_BATCH_TOKENS = 2400`, recursive hierarchical map-reduce reduction in `_reduce_notes_hierarchically()`, citation integrity validation in `validate_and_filter_page_citations()`, structural heading validation in `validate_notes_structure()`, and publication-quality Markdown formatting via `export_notes_to_markdown()`.
- **UI Integration:** `app.py` implements a dedicated **Study Notes** tab with document selection, full/page-range/topic scope controls, page range validation, progress reporting, metadata metrics, session state caching, and single-click Markdown download button.
- **Unit Tests:** `tests/test_study_notes_generator.py` (11 passing tests) and `tests/test_app_ui.py` (5 passing UI tests in `TestStage8StudyNotesUI`). Full test suite: 172 passed.

### Root Cause
1. **Unbounded Prompt Concatenation:** Previous implementation stuffed all document chunks into a single prompt without token budgeting, overflowing Qwen2.5 7B's 4096-token context budget on multi-page PDFs.
2. **Missing Hierarchical Map-Reduce Reduction:** When documents exceeded context limits, no recursive synthesis engine existed to combine batch notes hierarchically into an integrated academic overview.
3. **Citation Hallucinations:** Previous notes generation did not validate bracketed page citations against source chunks supplied in each batch, allowing ungrounded page references.
4. **Missing UI & Export Controls:** Study notes were buried inside the PDF Summary tab as a secondary button with no dedicated tab, no scope selection (Full Document, Page Range, Topic), and no file download button.

### Resolution
1. Added `_build_notes_batches()` with prompt-overhead-aware chunk packing enforcing `NOTES_MAX_BATCH_TOKENS = 2400`.
2. Added `_reduce_notes_hierarchically()` implementing multi-level recursive map-reduce grouping (pairing >= 2 notes per subset) guaranteeing convergence without context overflow.
3. Added `validate_and_filter_page_citations()` scrubbing hallucinated page citations.
4. Added `validate_notes_structure()` ensuring academic heading hierarchy, definitions, and takeaways.
5. Added `export_notes_to_markdown()` with complete document metadata, metrics, and source attribution.
6. Created a dedicated **Study Notes** tab in `app.py` with Full Document, Page Range (with start/end validation), and Topic focus scopes, execution progress reporting, metadata display, and single-click Markdown export.
7. Fixed chunk schema handling in topic retrieval (`c.get("metadata", c).get("page", 1)`).
8. Verified with 11 targeted unit tests, 5 AppTest integration tests, full 172-test project regression suite, and live end-to-end generation with local Qwen2.5 7B via Ollama.

### Final Resolution Status
`Resolved`

---

## BUG-02: Flashcard Generation Not Working

### Description
Flashcard generation from uploaded PDFs is currently failing or producing incomplete results. It must generate high-yield question-and-answer pairs across configurable counts, cover concepts evenly across documents, reject duplicates, strictly cite source references, provide revision/editing capabilities, validate JSON schemas safely, and handle malformed outputs with bounded retries.

### Status
`Resolved` (Stage 7 implementation complete & verified)

### Existing Implementation Status
- **Backend Module:** `src/generation/flashcard_generator.py` contains batch-aware `generate_flashcards()`, robust JSON extraction, schema validation, deduping, and Markdown/CSV export functions.
- **UI Integration:** `app.py` implements dedicated `_render_flashcard_tab()` with single-card carousel, front/back flip, navigation, in-place editing with persistence, Markdown and Anki CSV download buttons, and full deck list view.
- **Unit Tests:** `tests/test_flashcard_generator.py` (12 passing tests) and `tests/test_app_ui.py` (5 passing UI tests in `TestStage7FlashcardUI`).

### Root Cause
Investigated and confirmed in Stage 7 Phase 1:
1. **Unbounded Context & Missing Multi-Batch Distribution:** `generate_flashcards()` did not split or sample chunks into token-budgeted batches (`max_batch_tokens` was accepted but ignored). It concatenated all document chunks into a single prompt string, exceeding Ollama's 4096-token context window on PDFs with more than ~10 chunks, causing context truncation, generation timeouts, or failure.
2. **Brittle Schema & Root Array Parsing:** `extract_flashcard_json_payload()` rejected valid JSON where the root was an array `[...]` rather than `{"flashcards": [...]}`. It only recognized exact keys `"front"` and `"back"`, discarding valid cards with synonyms like `"question"`/`"answer"`, `"term"`/`"definition"`, or `"prompt"`/`"response"`.
3. **Overly Restrictive Answer Length & Chunk Formatting:** `parse_and_validate_flashcards()` rejected concise answers under 5 characters (e.g. "ATP", "DNA", "0", "4", "RNA") and string-formatted `source_chunk_ids` (e.g. `"c1"` instead of `["c1"]`), discarding valid flashcards instead of normalizing them.
4. **Missing UI In-Place Card Revision:** The Streamlit interface displayed cards as static markdown with no mechanism to edit question or answer text, and no state persistence across reruns.
5. **Missing Deck Export Functionality:** No options existed to export or download flashcards as Markdown (.md) or Anki-compatible CSV (.csv).

### Files Modified & Created
- `src/generation/flashcard_generator.py` (Multi-batch distribution, refined schema parsing, bounded retries, markdown & anki export functions)
- `app.py` (In-place card revision, Markdown and Anki CSV download buttons, deck viewer improvements)
- `tests/test_flashcard_generator.py` (Expanded to 12 unit tests covering batching, root lists, synonyms, concise answers, retries, exports)
- `tests/test_app_ui.py` (Added `TestStage7FlashcardUI` with 5 AppTest integration tests)

### Required Tests (All Passed)
1. Multi-batch chunk coverage and token budgeting for large PDFs (`test_multi_batch_chunk_distribution`).
2. Schema validation, root array parsing, synonym keys, and partial response recovery (`test_extract_flashcard_json_payload_root_list`, `test_extract_flashcard_json_payload_cards_synonym_key`).
3. Concise valid answer preservation (`test_parse_and_validate_concise_answers_and_synonyms`).
4. Deduplication against existing cards (`test_parse_and_validate_flashcards`).
5. Source reference grounding and deterministic page derivation (`test_generate_flashcards_end_to_end`).
6. Bounded retries on malformed batches (`test_bounded_retry_on_malformed_batch`).
7. Streamlit AppTest card rendering, flip, navigation (`test_01_flashcard_generation_render_and_card_counter`, `test_02_flashcard_flip_and_navigation`).
8. Streamlit AppTest in-place editing and rerun persistence (`test_03_flashcard_in_place_editing_and_persistence`).
9. Export controls and downloads (`test_04_flashcard_export_controls`, `test_export_flashcards_to_markdown`, `test_export_flashcards_to_anki_csv`).
10. Automatic cache invalidation on document deletion / vector store reset (`test_05_flashcard_invalidation_on_delete_and_reset`).

### Acceptance Criteria
1. Generate flashcards from uploaded PDFs. -> **Met**
2. Create concise question-and-answer pairs. -> **Met**
3. Support configurable flashcard counts. -> **Met**
4. Avoid duplicate and repetitive cards. -> **Met**
5. Cover important concepts from the uploaded material. -> **Met**
6. Preserve source references wherever possible. -> **Met**
7. Allow users to view and revise generated flashcards. -> **Met**
8. Validate generated JSON against the required schema. -> **Met**
9. Handle malformed or incomplete model responses safely. -> **Met**
10. Strictly use `get_llm_provider("ollama")` with Qwen2.5 7B. -> **Met**

### Final Resolution Status
`Resolved` (Stage 7)

**Resolution Details:**
- Implemented multi-batch chunk distribution and prompt budgeting (`_build_flashcard_batches`, `_plan_flashcard_batches`) ensuring prompts stay strictly within `FLASHCARD_MAX_BATCH_TOKENS` without context truncation.
- Enhanced JSON parser `extract_flashcard_json_payload()` to accept top-level JSON arrays `[...]`, alternative synonym keys (`question`/`answer`, `term`/`definition`, `prompt`/`response`), and string-formatted `source_chunk_ids`.
- Relaxed answer validation to accept concise valid terms/definitions (`len(back) >= 1`) such as "ATP", "DNA", "4", "RNA" while preserving grounding and Jaccard similarity deduplication.
- Built interactive Streamlit study deck (`app.py`) featuring single-card carousel view, front/back flip interaction, previous/next navigation, card counter (`Card X of Y`), and in-place question/answer revision with session persistence.
- Added full export capabilities (`export_flashcards_to_markdown`, `export_flashcards_to_anki_csv`) with one-click download buttons.
- Verified with 12 targeted unit tests, 5 Streamlit AppTest integration tests, and live end-to-end Qwen2.5 7B inference through local Ollama. Zero regressions across 161 total test suite cases.
