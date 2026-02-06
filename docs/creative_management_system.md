# Creative Management System for PerryPicks

## Goal
Convert structured model outputs (predictions, confidence intervals, edge metrics, matchup context) into on-brand creative assets automatically using Z.AI image generation.

## Recommended Architecture

### 1) Prediction Event Layer
- **Trigger source**: when a new pregame prediction record is finalized (or updated), emit an internal event.
- Event payload should include:
  - game_id, date/time (tip-off + timezone)
  - teams, spread/total context, model pick(s)
  - confidence/probability + uncertainty band
  - audience segment (X/Twitter, Instagram story, Telegram, Discord, app card)
  - experiment flags (tone/style version)

### 2) Creative Orchestrator Service
Build a separate `creative_orchestrator` job/service that consumes prediction events and handles:
1. **Content strategy selection**
   - Choose template family by channel and confidence tier.
   - Example tiers:
     - High confidence (e.g., >= 63%): assertive creative.
     - Mid confidence: balanced analytical tone.
     - Lower confidence: "watchlist" framing.
2. **Prompt builder**
   - Assemble prompt from brand-safe blocks instead of freeform generation.
   - Include guardrails like: avoid guaranteed-win language, avoid betting claims, display uncertainty.
3. **Asset generation (Z.AI API)**
   - Request multiple candidate images (e.g., 3 variants).
   - Keep deterministic seeds for repeatability and A/B testing.
4. **Policy + quality gate**
   - Run text overlays through banned phrase checks.
   - Validate readability/contrast and layout constraints.
5. **Human-in-the-loop queue (optional but recommended initially)**
   - Slack/Discord approval workflow for first N weeks.
6. **Publishing adapter**
   - Output to per-channel queues or directly to social posting tooling.

### 3) Brand System as Configuration
Store branding as code/config in version control:
- palette, font stacks, logo treatment
- tone guide (voice, prohibited wording, CTA style)
- composition rules (safe margins for platform crops)
- reusable prompt snippets and negative prompts
- channel templates with size specs

This avoids drift and allows clean rollback.

### 4) Data Model You’ll Need
Use three core entities:
1. **PredictionCreativeRequest**
   - Input prediction context + target channel.
2. **CreativeVariant**
   - Prompt, seed, model params, generated asset URI, quality metrics.
3. **CreativePublishRecord**
   - Where/when published, engagement metrics, linked back to prediction_id.

This creates end-to-end traceability from prediction to published post.

### 5) Observability & Feedback Loop
Track:
- generation latency
- acceptance/rejection rate by template
- engagement by creative style and confidence tier
- downstream conversion metrics (clickthrough, app opens, subscription actions)

Feed performance back into strategy rules so winning template/prompt combinations are automatically prioritized.

## Suggested Content Pipeline (Practical)
1. Prediction completes in existing pipeline.
2. Emit a `prediction.ready` event.
3. Orchestrator enriches with team metadata and schedule context.
4. Select template + prompt profile.
5. Call Z.AI for image candidates.
6. Render overlay text (pick, confidence, key stat, timestamp).
7. Run moderation + readability checks.
8. Auto-approve high quality or route to manual queue.
9. Publish and log metrics.

## Backend Management Portal (How Humans Interact)

Yes—this system should include a backend portal so operators can manage strategy, review assets, and control publishing safely.

### Core Portal Modules
1. **Creative Inbox (Review Queue)**
   - List all generated variants by game/date/channel.
   - Side-by-side compare candidates with quick actions:
     - Approve
     - Reject (with reason)
     - Request Regenerate
     - Edit copy/CTA before publish
   - Bulk actions for slate-level approvals.

2. **Campaign & Template Manager**
   - Manage brand templates, prompt profiles, and channel presets.
   - Version history for prompts/template configs.
   - Draft -> QA -> Published states to prevent accidental live updates.

3. **Publishing Control Center**
   - Calendar/timeline view of scheduled posts.
   - "Hold all" and channel-level pause toggles.
   - Last-minute replacement flow when prediction updates invalidate assets.

4. **Brand & Policy Console**
   - Central place to manage:
     - banned phrases
     - required disclaimers by region/channel
     - voice/tone guardrails
   - Preview checker that highlights policy violations before approval.

5. **Asset Library**
   - Searchable store of all generated and published creatives.
   - Filters: team, model confidence tier, channel, prompt version, date range.
   - Track lineage: prediction -> variant -> published post -> engagement.

6. **Experimentation & Analytics Dashboard**
   - A/B performance by template/prompt family.
   - Approval rate and rejection reasons.
   - Engagement and conversion metrics tied to creative IDs.

### Recommended User Roles & Permissions
- **Admin**: full access; policy, publishing, and user management.
- **Editor/Creative Lead**: approve/reject/edit creatives; manage templates.
- **Analyst**: view metrics and run experiments; no publish rights.
- **Operator**: execute scheduled publishing and monitor queues.
- **Viewer**: read-only for stakeholders.

Use RBAC and immutable audit logs for all approvals, edits, and policy overrides.

### Suggested Review Workflow
1. Prediction event generates variants.
2. Variants land in Creative Inbox with auto quality score.
3. If score >= threshold and no policy flags: auto-approve (optional).
4. Otherwise route to reviewer with violation/rejection hints.
5. Reviewer can approve, edit copy, or regenerate with one click.
6. Approved asset is scheduled/published and tracked.

### Operational Safeguards in Portal
- Kill switch: global and per-channel pause.
- Freshness warnings: block publish if prediction timestamp is stale.
- Incident mode: fallback to static branded template when generation is degraded.
- SLA monitor: alert when queue delays threaten pre-game publish windows.

### API Endpoints the Portal Will Need (Example)
- `GET /creative/queue`
- `POST /creative/{id}/approve`
- `POST /creative/{id}/reject`
- `POST /creative/{id}/regenerate`
- `PATCH /creative/{id}/copy`
- `POST /publishing/schedule`
- `POST /publishing/pause`
- `GET /analytics/creative-performance`

This portal layer is what makes the automation controllable and trustworthy in production.

## Limitations / Concerns

### 1) Compliance & Risk Language
- Sports-prediction creative can accidentally imply certainty.
- You should enforce policy checks for terms like "lock," "guaranteed," or ROI promises.
- Regional gambling-ad compliance differs; geo-aware policy may be required.

### 2) Hallucinated or Inconsistent Visuals
- Image generators can output irrelevant logos, wrong team colors, distorted text, or unofficial branding.
- Keep generated backgrounds abstract and render final text/logos in your own deterministic layer (post-processing), not inside raw generation.

### 3) Brand Drift Over Time
- Without strict templating, quality and tone will drift as prompts evolve.
- Maintain prompt/version registry and require reviews for major style changes.

### 4) Latency at Peak Times
- Pre-game windows are bursty; generation queues can bottleneck.
- Use fallback templates and degrade gracefully to non-generated static cards if SLA is missed.

### 5) Cost Control
- Multi-variant generation across channels can become expensive quickly.
- Use tiered generation policy:
  - high-value games: 3-5 variants
  - standard games: 1-2 variants
  - low-value games: static template only

### 6) Measurement Ambiguity
- Creative performance is confounded by game popularity and posting time.
- Use controlled A/B experiments and normalized metrics.

### 7) Data Integrity
- If predictions update near tip-off, stale creative can be posted.
- Add freshness checks and invalidation rules before publishing.

## Additional Features Worth Adding

1. **Narrative Modes**
   - Different story modes: data-heavy, hype, educational, risk-aware.
   - Auto-select mode by audience/channel.

2. **Confidence-Aware CTAs**
   - CTA style adjusts based on uncertainty ("lean" vs "strong pick").

3. **Auto-Thread Generation**
   - Create a main image + 2 follow-up slides:
     - key matchup factor
     - model confidence explanation
     - "what would invalidate this pick"

4. **Personalized Feeds**
   - Generate team-follow specific variants for users who follow particular teams.

5. **Retro/Result Cards**
   - Automatically produce post-game "model result" recaps to build trust and transparency.

6. **Creative Knowledge Base**
   - Store top-performing prompts/templates and automatically propose weekly updates.

7. **Safety Kill Switches**
   - Pause posting when upstream data quality checks fail or model drift alarms trigger.

## Phased Rollout Plan
- **Phase 1 (2 weeks)**: Portal MVP + manual approval only, 1-2 channels, strict templates.
- **Phase 2 (2-4 weeks)**: Partial auto-publish on high-confidence quality scores + analytics dashboard.
- **Phase 3**: Full optimization loop with A/B routing and dynamic template selection.

## Minimum Viable Technical Stack
- Queue/Event Bus: Redis Streams / SQS / Kafka
- Orchestrator: Python worker (Celery/RQ/FastAPI background jobs)
- Storage: Postgres (metadata) + object store (assets)
- Portal Backend: FastAPI/Django + RBAC + audit log table
- Portal Frontend: React/Next.js admin console
- Policy checks: simple rule engine + moderation API
- Analytics: dbt/warehouse dashboards linked to creative IDs
