from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import streamlit as st

from src.creative.portal_store import (
    CreativePortalStore,
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_REJECTED,
)


def _tier_from_confidence(conf: float) -> str:
    if conf >= 0.63:
        return "high"
    if conf >= 0.56:
        return "medium"
    return "watchlist"


def _safe_pred_payload(pred: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "game_id": pred.get("game_id"),
        "home_name": pred.get("home_name"),
        "away_name": pred.get("away_name"),
        "pred": pred.get("pred", {}),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def render_creative_portal(last_pred: Dict[str, Any] | None) -> None:
    st.markdown("### 🎨 Creative Management Portal")
    st.caption("Manage review queue, template profiles, policy controls, scheduling, and creative analytics.")

    store = CreativePortalStore()

    tab_inbox, tab_policy, tab_publish, tab_assets, tab_analytics = st.tabs(
        ["Creative Inbox", "Template & Policy", "Publishing", "Asset Library", "Analytics"]
    )

    with tab_inbox:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Create request from current prediction", use_container_width=True):
                if not last_pred or not last_pred.get("game_id"):
                    st.warning("Run a prediction first so a game_id/payload exists.")
                else:
                    pred = last_pred.get("pred", {}) or {}
                    confidence = float(pred.get("home_win_prob") or 0.5)
                    req_id = store.create_request(
                        game_id=str(last_pred.get("game_id")),
                        channel="twitter",
                        confidence_tier=_tier_from_confidence(confidence),
                        prediction_payload=_safe_pred_payload(last_pred),
                        created_by="streamlit_operator",
                    )
                    seed = random.randint(1000, 999999)
                    prompt = (
                        "Basketball matchup poster, high contrast, clean modern sports design, "
                        "no gambling certainty language, add space for overlay text"
                    )
                    overlay = (
                        f"{last_pred.get('away_name', 'Away')} @ {last_pred.get('home_name', 'Home')} | "
                        f"Model confidence: {confidence:.1%}"
                    )
                    store.add_variant(
                        request_id=req_id,
                        variant_label="v1",
                        prompt_text=prompt,
                        overlay_copy=overlay,
                        model_name="z.ai-image",
                        seed=seed,
                        image_uri=f"generated://{last_pred.get('game_id')}/{seed}",
                        quality_score=0.75,
                        actor="streamlit_operator",
                        negative_prompt="distorted text, trademarked logos, guaranteed-win language",
                    )
                    st.success(f"Created creative request #{req_id} and initial variant.")

        with c2:
            with st.expander("Manual request", expanded=False):
                game_id = st.text_input("Game ID", key="cp_manual_game")
                channel = st.selectbox("Channel", ["twitter", "instagram", "discord", "telegram"], key="cp_manual_channel")
                conf = st.slider("Confidence", 0.0, 1.0, 0.57, 0.01, key="cp_manual_conf")
                if st.button("Create manual request", key="cp_create_manual", use_container_width=True):
                    req_id = store.create_request(
                        game_id=game_id or None,
                        channel=channel,
                        confidence_tier=_tier_from_confidence(conf),
                        prediction_payload={"manual": True, "confidence": conf, "game_id": game_id},
                        created_by="streamlit_operator",
                    )
                    st.success(f"Manual request created: #{req_id}")

        queue = store.list_queue()
        if not queue:
            st.info("No creative variants in review queue yet.")
        else:
            st.dataframe(queue, use_container_width=True)
            variant_ids = [row["variant_id"] for row in queue]
            chosen_variant = st.selectbox("Select variant", variant_ids, key="cp_selected_variant")
            selected = next((row for row in queue if row["variant_id"] == chosen_variant), None)
            if selected:
                st.code(selected.get("prompt_text") or "", language="text")
                st.write(f"Overlay: {selected.get('overlay_copy') or ''}")
                st.write(f"Image URI: {selected.get('image_uri') or ''}")

                a1, a2, a3 = st.columns(3)
                with a1:
                    if st.button("Approve", key="cp_approve", use_container_width=True):
                        store.set_variant_status(variant_id=int(chosen_variant), status=STATUS_APPROVED, actor="reviewer")
                        st.success("Variant approved.")
                with a2:
                    reason = st.text_input("Reject reason", key="cp_reject_reason")
                    if st.button("Reject", key="cp_reject", use_container_width=True):
                        store.set_variant_status(
                            variant_id=int(chosen_variant),
                            status=STATUS_REJECTED,
                            actor="reviewer",
                            reason=reason or "Needs refinement",
                        )
                        st.warning("Variant rejected.")
                with a3:
                    if st.button("Regenerate", key="cp_regen", use_container_width=True):
                        new_seed = random.randint(1000, 999999)
                        store.add_variant(
                            request_id=int(selected["request_id"]),
                            variant_label=f"regen-{new_seed}",
                            prompt_text=selected.get("prompt_text") or "",
                            overlay_copy=selected.get("overlay_copy") or "",
                            model_name="z.ai-image",
                            seed=new_seed,
                            image_uri=f"generated://{selected.get('game_id')}/{new_seed}",
                            quality_score=max(0.6, float(selected.get("quality_score") or 0.7) - 0.02),
                            actor="reviewer",
                        )
                        st.success("New variant generated.")

    with tab_policy:
        st.markdown("#### Prompt Profiles")
        with st.form("cp_prompt_profile"):
            profile_name = st.text_input("Profile name", value="default_sports_card")
            profile_channel = st.selectbox("Channel", ["twitter", "instagram", "discord", "telegram"])
            prompt_template = st.text_area(
                "Prompt template",
                value="Dynamic basketball background, modern typography zones, compliant sports prediction framing",
            )
            negative_prompt = st.text_area(
                "Negative prompt",
                value="distorted text, fake logos, guaranteed-win statements, trademarked marks",
            )
            active = st.checkbox("Active", value=True)
            submitted = st.form_submit_button("Save profile")
            if submitted:
                store.upsert_prompt_profile(
                    profile_name=profile_name,
                    channel=profile_channel,
                    prompt_template=prompt_template,
                    negative_prompt=negative_prompt,
                    is_active=active,
                )
                st.success("Prompt profile saved.")

        st.dataframe(store.list_prompt_profiles(), use_container_width=True)

        st.markdown("#### Policy Controls")
        current_policy = store.get_policy(
            "global_policy",
            {
                "banned_phrases": ["lock", "guaranteed", "risk-free", "100%"],
                "required_disclaimer": "For entertainment and informational purposes only.",
            },
        )
        banned = st.text_area("Banned phrases (comma-separated)", value=", ".join(current_policy.get("banned_phrases", [])))
        disclaimer = st.text_area("Required disclaimer", value=current_policy.get("required_disclaimer", ""))
        if st.button("Save policy"):
            store.upsert_policy(
                "global_policy",
                {
                    "banned_phrases": [x.strip() for x in banned.split(",") if x.strip()],
                    "required_disclaimer": disclaimer,
                },
            )
            st.success("Policy updated.")

    with tab_publish:
        kill_switch = store.get_system_toggle("global_pause", "false") == "true"
        new_kill_switch = st.toggle("Global publish pause", value=kill_switch)
        if new_kill_switch != kill_switch:
            store.set_system_toggle("global_pause", "true" if new_kill_switch else "false")
            st.info("Publishing toggle updated.")

        queue = store.list_queue()
        approved = [q for q in queue if q.get("status") == STATUS_APPROVED]
        if not approved:
            st.info("No approved variants ready for scheduling.")
        else:
            st.dataframe(approved, use_container_width=True)
            ids = [row["variant_id"] for row in approved]
            variant_to_schedule = st.selectbox("Approved variant", ids, key="cp_schedule_variant")
            schedule_dt = st.datetime_input(
                "Schedule UTC",
                value=datetime.now(timezone.utc) + timedelta(minutes=15),
                format="YYYY-MM-DD HH:mm",
            )
            notes = st.text_input("Scheduling notes", value="auto-scheduled from portal")
            if st.button("Schedule publish", use_container_width=True):
                chosen = next((row for row in approved if row["variant_id"] == variant_to_schedule), None)
                if chosen:
                    store.schedule_publish(
                        variant_id=int(variant_to_schedule),
                        channel=chosen.get("channel") or "twitter",
                        scheduled_ts_utc=schedule_dt.astimezone(timezone.utc).isoformat(timespec="seconds"),
                        actor="operator",
                        notes=notes,
                    )
                    st.success("Publish scheduled.")

    with tab_assets:
        assets = store.list_assets(limit=200)
        st.dataframe(assets, use_container_width=True)

    with tab_analytics:
        metrics = store.summarize_metrics()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Variants", metrics["total_variants"])
        m2.metric("Approved", metrics["approved"])
        m3.metric("Rejected", metrics["rejected"])
        m4.metric("Approval Rate", f"{metrics['approval_rate']:.1%}")

        st.markdown("#### Top Rejection Reasons")
        st.dataframe(metrics["top_rejection_reasons"], use_container_width=True)
