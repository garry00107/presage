"""
Injector — selects which staged memories to inject into the LLM context.

Two-phase selection:
  Phase A: Select StagedMemory objects (memory-level selection)
           Uses tier priority: AUTO first, then HOT if budget remains
  Phase B: Select chunks from selected memories (chunk-level selection)
           Uses 0/1 knapsack on pre-chunked units — NEVER truncates content

The injector is purely computational — no I/O, no side effects.
It receives staged memories and returns an InjectionPlan.
"""

import structlog
from core.staging.models import InjectionPlan, StagedChunk, StagedMemory, SlotTier
from core.types import MemoryID
from math_core.knapsack import knapsack_01
from config.settings import settings

log = structlog.get_logger(__name__)


class Injector:
    """
    Converts staged memories into an InjectionPlan respecting token budget.

    Design invariant: chunk content is NEVER modified.
    The knapsack selects whole chunks only (Phase 2 chunker guarantee).
    """

    def plan(
        self,
        staged: list[StagedMemory],
        token_budget: int | None = None,
        soft_trigger: str | None = None,
    ) -> InjectionPlan:
        """
        Build an injection plan from available staged memories.

        Args:
            staged:       all non-expired StagedMemory objects from cache
            token_budget: max tokens to inject (defaults to settings value)
            soft_trigger: user message text for HOT tier trigger matching

        Returns:
            InjectionPlan with selected chunks and rendered context.
        """
        budget = token_budget if token_budget is not None else settings.max_inject_tokens
        if not staged or budget <= 0:
            return InjectionPlan(
                chunks=[], tokens_used=0, tokens_budget=budget,
                memories_injected=0, staged_memories=[],
            )

        # Phase A: select which StagedMemory objects to consider
        selected_memories = self._select_memories(staged, soft_trigger)

        if not selected_memories:
            return InjectionPlan(
                chunks=[], tokens_used=0, tokens_budget=budget,
                memories_injected=0, staged_memories=[],
            )

        # Phase B: knapsack chunk selection across all selected memories
        all_chunks = self._gather_chunks(selected_memories)
        chosen_chunks = knapsack_01(all_chunks, budget)

        tokens_used = sum(c["tokens"] for c in chosen_chunks)
        parent_ids  = {c["parent_id"] for c in chosen_chunks}

        log.debug(
            "injector.plan_built",
            memories_considered=len(selected_memories),
            chunks_considered=len(all_chunks),
            chunks_selected=len(chosen_chunks),
            tokens_used=tokens_used,
            tokens_budget=budget,
        )

        # Convert back to StagedChunk for InjectionPlan
        staged_chunks = [
            StagedChunk(
                chunk_id=c["id"],
                parent_id=c["parent_id"],
                content=c["content"],
                tokens=c["tokens"],
                score=c["score"],
                source_type=c["source_type"],
                chunk_index=c["chunk_index"],
                source=c.get("source", ""),
            )
            for c in chosen_chunks
        ]

        return InjectionPlan(
            chunks=staged_chunks,
            tokens_used=tokens_used,
            tokens_budget=budget,
            memories_injected=len(parent_ids),
            staged_memories=selected_memories,
        )

    # ── Phase A: Memory selection ──────────────────────────────────────────────

    def _select_memories(
        self,
        staged: list[StagedMemory],
        soft_trigger: str | None,
    ) -> list[StagedMemory]:
        """
        Select memories to inject based on tier priority.

        Rules:
          AUTO tier: always included (highest confidence)
          HOT tier:  included if soft_trigger matches any annotation tag,
                     OR if AUTO tier left budget headroom
          WARM tier: only included on explicit request (Phase 7 feature)
        """
        auto  = [s for s in staged if s.tier == SlotTier.AUTO]
        hot   = [s for s in staged if s.tier == SlotTier.HOT]

        selected = list(auto)

        # Include HOT memories if soft trigger matches
        if soft_trigger and hot:
            triggered = self._soft_trigger_match(hot, soft_trigger)
            selected.extend(triggered)
        elif hot and len(auto) == 0:
            # No AUTO memories ready: fall through to best HOT memory
            selected.extend(hot[:1])

        # Deduplicate by combined_score, keep highest scoring per parent
        selected = self._deduplicate(selected)

        # Sort by combined_score
        selected.sort(key=lambda s: s.combined_score, reverse=True)

        return selected

    def _soft_trigger_match(
        self,
        hot: list[StagedMemory],
        trigger: str,
    ) -> list[StagedMemory]:
        """
        Match HOT memories whose annotation tags appear in the trigger text.
        A memory 'fires' if any of its annotation tags is mentioned.
        """
        trigger_lower = trigger.lower()
        matched = []
        for sm in hot:
            tags = sm.prediction.annotation_tags
            for tag in tags:
                # Strip tag namespace prefix for matching
                # "symbol:verify_token" → match "verify_token" in trigger
                keyword = tag.split(":", 1)[-1].lower()
                if keyword and keyword in trigger_lower:
                    matched.append(sm)
                    break
        return matched

    def _deduplicate(self, memories: list[StagedMemory]) -> list[StagedMemory]:
        """Keep only highest-scoring StagedMemory per parent_id."""
        best: dict[str, StagedMemory] = {}
        for sm in memories:
            for chunk in sm.chunks:
                pid = chunk.parent_id
                if pid not in best or sm.combined_score > best[pid].combined_score:
                    best[pid] = sm
        return list(best.values())

    # ── Phase B: Chunk gathering for knapsack ──────────────────────────────────

    def _gather_chunks(
        self, memories: list[StagedMemory]
    ) -> list[dict]:
        """
        Flatten all chunks from selected memories into knapsack format.
        Deduplicates by chunk_id across memories.
        """
        seen: set[str] = set()
        all_chunks: list[dict] = []

        for sm in memories:
            for c in sm.chunks:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                all_chunks.append({
                    "id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "tokens": max(c.tokens, 1),
                    "score": c.score,
                    "source_type": c.source_type,
                    "source": c.source,
                })

        return all_chunks

