"""Seeded fixture: the §13 worked example.

soft-token-attention-collapse — the LLM ignores injected soft tokens because the
projector is frozen during LoRA/CE training. A confirming hypothesis (unfreeze +
joint loss) and a *refuted sibling* (rmsnorm-logscale-init alone) are both seeded
so tests can assert negative-result ranking and calibration.
"""

from __future__ import annotations

from vibescience_mcp.core import Store


def seed(vault_root) -> Store:
    s = Store(vault_root)

    # --- tag vocabulary (registered basis, like diagnostics) ---
    # `projector-freeze` also carries a synonym to exercise alias resolution.
    for tid, desc in [
        ("blip2-qformer", "BLIP-2 Q-Former style bridging."),
        ("contrastive", "Contrastive objectives."),
        ("projector", "The soft-token projector module."),
        ("rmsnorm", "RMSNorm parameterisation."),
    ]:
        s.register_tag(tid, axis="topic", description=desc)
    s.register_tag("attention-collapse", axis="problem",
                   description="The LLM stops attending to injected soft tokens.")
    s.register_tag("projector-freeze", axis="problem",
                   description="Projector frozen during training, blocking gradients.",
                   aliases=["frozen-projector"])

    # --- diagnostics (fixed basis) ---
    s.register_diagnostic("linear probe r2", unit="r2", direction="higher_better",
                          id="linear_probe_r2", topic_tags=["projector"],
                          description="How linearly decodable the target is from soft tokens.")
    s.register_diagnostic("HR MAE", unit="bpm", direction="lower_better", id="hr_mae_bpm",
                          description="Heart-rate mean absolute error in bpm.")
    s.register_diagnostic("attention mass on soft tokens", unit="frac",
                          direction="higher_better", id="attn_mass_soft_tokens",
                          topic_tags=["projector"],
                          description="Fraction of attention the LLM puts on injected soft tokens.")
    s.register_diagnostic("attention entropy", unit="nats", direction="neutral",
                          id="attn_entropy", description="Entropy of the attention distribution.")

    # --- interventions ---
    s.register_intervention("unfreeze projector", id="unfreeze-projector",
                            description="Allow captioning-loss gradients to reshape soft tokens.",
                            topic_tags=["projector", "blip2-qformer"])
    s.register_intervention("joint contrastive + captioning", id="joint-contrastive-captioning",
                            description="Train contrastive and captioning objectives jointly.",
                            topic_tags=["contrastive"])
    s.register_intervention("rmsnorm logscale init", id="rmsnorm-logscale-init",
                            description="Initialize RMSNorm log-scale to widen soft-token norms.",
                            topic_tags=["rmsnorm"])

    # --- papers ---
    s.add_paper("BLIP-2", arxiv_id_or_url="2301.12597", id="blip2",
                key_claims=["Q-Former bridges frozen image encoder and LLM."],
                topic_tags=["blip2-qformer"])
    s.add_paper("CoCa", arxiv_id_or_url="2205.01917", id="coca",
                key_claims=["Joint contrastive + captioning improves representations."],
                topic_tags=["contrastive"])

    # --- problem ---
    s.create_problem(
        "soft token attention collapse", id="soft-token-attention-collapse",
        description=("The LLM ignores injected soft tokens; suspected cause is a frozen "
                     "projector during LoRA/CE training, so captioning-loss gradients "
                     "never reshape the soft-token representations."),
        topic_tags=["blip2-qformer", "contrastive"],
        problem_tags=["attention-collapse", "projector-freeze"],
        paper_refs=["blip2", "coca"],
    )

    # --- CONFIRMING hypothesis ---
    s.propose_hypothesis(
        problem_id="soft-token-attention-collapse",
        id="projector-freeze-blocks-caption-grad",
        statement=("The projector being frozen blocks captioning-loss gradients from "
                   "reshaping soft tokens; unfreezing it + joint loss fixes collapse."),
        rationale="Analogous to the BLIP-2 Q-Former vs CoCa design gap.",
        interventions=["unfreeze-projector", "joint-contrastive-captioning"],
        predicted_effects=[
            {"diagnostic_id": "attn_mass_soft_tokens", "direction": "up",
             "magnitude_note": "should rise well above noise floor"},
            {"diagnostic_id": "hr_mae_bpm", "direction": "down",
             "magnitude_note": "downstream task should improve"},
        ],
        plan="Unfreeze projector, add joint contrastive+captioning loss, retrain LoRA.",
        topic_tags=["contrastive", "projector", "blip2-qformer"],
        problem_tags=["attention-collapse", "frozen-projector"],  # alias → projector-freeze
        papers=["blip2", "coca"],
    )
    e1 = s.start_experiment("projector-freeze-blocks-caption-grad",
                            git_ref="fix/soft-token-norm-scale@a1b2c3d",
                            external_run="wandb.ai/anton/eiv/runs/xyz",
                            id="unfreeze-projector-joint-loss")
    s.record_diagnostics(e1.id, [
        {"diagnostic_id": "attn_mass_soft_tokens", "before": 0.02, "after": 0.31},
        {"diagnostic_id": "hr_mae_bpm", "before": 8.1, "after": 4.0},
    ])
    s.close_experiment(e1.id, notes="Prediction matched on both diagnostics.")

    # --- REFUTED sibling (must be retained + surfaced by recall) ---
    s.propose_hypothesis(
        problem_id="soft-token-attention-collapse",
        id="rmsnorm-logscale-alone-fixes-collapse",
        statement="Raising rmsnorm-logscale-init alone fixes the attention collapse.",
        rationale="Maybe soft-token norms are just too small to be attended to.",
        interventions=["rmsnorm-logscale-init"],
        predicted_effects=[
            {"diagnostic_id": "attn_mass_soft_tokens", "direction": "up",
             "magnitude_note": "norms up → attention up"},
        ],
        plan="Bump rmsnorm log-scale init, retrain, measure attention mass.",
        topic_tags=["rmsnorm"],
        problem_tags=["attention-collapse"],
    )
    e2 = s.start_experiment("rmsnorm-logscale-alone-fixes-collapse",
                            git_ref="exp/rmsnorm-logscale@d4e5f6a",
                            id="rmsnorm-logscale-alone")
    # observed: attention mass did NOT rise — it fell slightly → refutes
    s.record_diagnostics(e2.id, [
        {"diagnostic_id": "attn_mass_soft_tokens", "before": 0.02, "after": 0.015},
    ])
    s.close_experiment(e2.id, notes="No effect; attention mass actually dropped.")

    return s
