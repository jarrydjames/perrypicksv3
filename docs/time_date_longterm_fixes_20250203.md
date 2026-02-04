# Long-term Sustainable Time/Date Architecture

**Date:** 2025-02-03  
**Status:** PROPOSED ARCHITECTURE IMPROVEMENTS

---

## Executive Summary

This document proposes sustainable, long-term architectural improvements to eliminate time/date issues in PerryPicks automation system.

---

## Problem Statement

### Current Issues
1. System clock out of sync with reality
2. Relative dates (`date='today'`) causing confusion
3. Manual datetime parsing with timezone bugs
4. No validation that scheduled times are actually in the future
5. ISO format offset issues (`+00:00` vs `Z`)
6. Timezone conversions scattered throughout codebase

### Impact
- PRE_GAME triggers not firing
- HALFTIME predictions not posting
- Games appearing in wrong order
- Manual intervention constantly required

---

## Proposed Architecture

### Principle 1: UTC-First Architecture

**Rule**: All internal operations use UTC. Convert to local timezone ONLY for display.

### Principle 2: Explicit Date Scheduling

**Rule**: Never use relative dates in production. Always use explicit YYYY-MM-DD.

### Principle 3: Pendulum for Timezone Handling

**Rule**: Replace `datetime` with `pendulum` for all timezone operations.

---

## Implementation Plan

### Phase 1: Immediate Fixes ✅ DONE

#### 1.1 Fix HALFTIME Trigger Logic ✅
- **Problem**: Old scheduled triggers blocking new game-state triggers
- **Solution**: Only check for FIRED triggers, not scheduled ones
- **Status**: Implemented in `worker/triggers.py` and `core/storage.py`

---

## Timeline

| Phase | Tasks | Duration | Priority |
|-------|-------|----------|----------|
| Phase 1 | HALFTIME fix | 1 day | ✅ DONE |
| Phase 2 | Pendulum migration | 3-5 days | HIGH |
| Phase 3 | Date scheduling improvements | 2-3 days | HIGH |

---

**Document Date:** 2025-02-03  
**Documented By:** Perry (code-puppy)