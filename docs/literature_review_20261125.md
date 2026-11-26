# Literature review: multimodal OCT + SLO models and where to go next (2026-11-25)

Written after run plan steps 8 and 11 to 13 showed that our diffusion denoiser does not use the SLO, and Matthias asked to use the SLO in any way that makes the model multimodal. Figures below are as reported by the authors (abstracts and summaries; full papers not read unless stated).

## Our position

- Data: pilot of 531 Spectralis scans (25 B-scans, 496 × 512, plus a 768 × 768 SLO and the position of every B-scan on the SLO). Labels: sex and eye only. The full pull should bring 5,000 to 10,000 scans with dates and ICD diagnoses from DataDirect.
- What works: the OCT-only diffusion model (`w2z1_sqrt24`) gives frozen features that predict the eye at AUROC 0.90, against about 0.5 from random weights.
- What does not: five ways of feeding the SLO to the denoiser (cross-attention with 64 or 256 tokens, the scan-line adapter, the SLO as a slice of the volume, and the same with its own noise step). None lowers the denoising error or makes generated OCTs follow the SLO. The real OCT en-face view correlates with its SLO at only 0.15 to 0.2 after high-pass, so the SLO says little about what the denoising loss measures.

## 1. Multimodal OCT + SLO/IR foundation models

| Model | Modalities | How the modalities are linked | Pre-training data |
|---|---|---|---|
| MIRAGE (npj Digit Med 2025) | OCT and SLO, paired | Multimodal masked autoencoder (MultiMAE): one shared ViT encoder, modality-specific input layers and decoders, plus retinal-layer pseudo-labels as an extra task. Works with either modality alone at inference. | 261k samples, 42k patients (VIBES) |
| OCTCube-M (Nat Biomed Eng 2026; the paper Matthias shared) | 3D OCT with IR or FAF | 3D masked autoencoder for OCT, then COEP, a contrastive loss between OCT volume and en-face image. OCT-to-IR retrieval Recall@1 0.64 (RETFound 0.46). | 26,685 OCT volumes; 26,685 OCT-IR pairs |
| OphMAE (preprint 2026) | 3D OCT and en-face OCT | Masked autoencoder with cross-modal masking and a similarity-consistency loss. The authors argue that contrastive losses suit image-text pairs better than OCT and en-face images, because the two carry unequal information. | 183,875 pairs, 32,765 patients |
| Sükei et al. (Sci Rep 2024) | 3D OCT and fundus photo | CLIP and CLOOB contrastive pre-training. Top-1 retrieval is low (about 0.1 with all samples per patient). | 153,306 pairs |
| EyeCLIP (npj Digit Med 2025) | 11 eye modalities and text | Reconstruction plus image-image and image-text contrastive losses | 2.77M images |
| SLO-Net (2024, multiple sclerosis) | OCT and IR-SLO | Separate networks, combined for the diagnosis. Adding the SLO improved diagnosis over OCT alone. | single site |

What this says for us:
- No published model makes OCT and SLO multimodal through a diffusion conditioner. They all use masked reconstruction across modalities (MIRAGE, OphMAE) or contrastive alignment (OCTCube-M, Sükei). Our negative result fits that pattern.
- The contrastive run (`w2z1_align`) is the OCTCube-M recipe on a diffusion backbone, plus a link at each place along the scan lines that none of these papers uses. OphMAE's warning applies: the SLO and the OCT share only part of their content, so alignment may be weak. Sükei's low retrieval at 153k pairs suggests that whole-scan retrieval on our 53 validation scans will be close to chance; the same-place cosine is the more sensitive measure.
- Masked cross-modal prediction (MIRAGE, OphMAE) is the main alternative if alignment fails. The closest version for us is predicting the SLO along the scan lines from the OCT encoder, as an extra head beside the denoising loss.
- Late fusion, separate encoders joined at the head, is the simplest floor and has shown a gain (SLO-Net).

## 2. Diffusion models as feature learners

- Chen, Liu, Xie and He (ICLR 2025, l-DAE): the features of a diffusion model come from denoising, not from the diffusion process. One noise level is nearly as good as a full schedule. Linear probes still trail contrastive and masking methods (ViT-B on ImageNet: about 66.6% vs 68.0% for MAE and 76.7% for MoCo v3).
- Masked diffusion (Pan et al.): the representation does not come from generative power; adding masking helped segmentation on medical and natural images.
- REPA (Yu et al., ICLR 2025): aligning the early layers of a diffusion model with a pre-trained self-supervised encoder (DINOv2) made training over 17.5 times faster and improved generation. The diffusion model learns better features when it gets them from a strong encoder.
- Medical: frozen diffusion features with light probes or finetuning are the common use (D-Cube for classification; a 3D diffusion pre-training for segmentation; OrthoDiffusion for musculoskeletal MRI).

What this says for us: our features come from denoising, so the open question is how they compare with masked-autoencoder OCT models trained on much more data. REPA suggests a direct way to borrow from them: align our denoiser's encoder with a public OCT encoder (OCTCube weights are on Hugging Face) while keeping the diffusion loss. That needs a licence check and an input-size adapter (OCTCube uses 60 × 256 × 256 crops; our grid is 24 × 512 × 512).

## 3. JEPA for 3D medical images

- 3D work is recent: COJEPA (brain MRI, JEPA plus a contrastive term, 2,286 scans), Rad-JEPA 3D (CT, about 120,000 scans), and comparisons of masked autoencoders against JEPA for brain-MRI disease detection.
- LeJEPA (Balestriero and LeCun, 2025) replaces the usual guards against collapse (stop-gradient, teacher-student EMA) with one regulariser, SIGReg, that pushes embeddings toward an isotropic Gaussian. It has one hyperparameter and drop-in code.

What this says for us: our `align_jepa` run uses the EMA-teacher form (as in BYOL and I-JEPA). With 372 training scans, collapse or very slow learning is a real risk. If its same-place cosine does not separate own from other SLO, SIGReg is the cleaner next try. COJEPA's combination of JEPA with a contrastive term is close to running our two losses together.

## 4. Tasks the full pull can support

- **Retinal age gap** (Matthias's next target): an OCT model in UK Biobank reached MAE 3.27 years, and each 5-year rise in the gap went with about 8% higher mortality (about 44,600 participants). A fundus-photo model linked each year of gap to about 2% higher mortality. A 2025 Japanese study found links with diabetes, hypertension and hyperlipidaemia, but the gap did not predict later onset. Needs age at scan, which the pull should give (birth date or age plus scan date).
- **Systemic disease from linked records:** AlzEye linked 1.57M OCTs at Moorfields to national admission records. RETFound fine-tuned on it predicted heart failure, myocardial infarction and stroke within 3 years, with UK Biobank as external test. Our DataDirect ICD link is the same kind of design. A thesis on AlzEye notes that group-level differences are easier to show than individual prediction.
- **Longitudinal progression:** conversion to neovascular AMD is the best-studied task (Yim et al., Nat Med 2020; Morph-SSL, self-supervision across visits; 3DTINC, non-contrastive learning over time). It needs repeat scans per eye with dates, which the pull may contain.
- **Data scale:** the published OCT foundation models use 26,000 to 260,000 volumes; ours will have 5,000 to 10,000. Gains from self-supervision at a few thousand scans are plausible but task-dependent; I found no OCT study with a learning curve at that size.

## 5. Suggested order of work

1. **Read the three alignment runs** (`w2z1_align`, `w2z1_align_jepa`, `w2z1_align_strong`) on the same-place cosine, own vs another scan's SLO. If own is clearly higher, the model is multimodal in the OCTCube-M sense and the SLO features can go to the heads.
2. **Add public OCT foundation models as baselines** (OCTCube or OCTCube-M, RETFound-OCT, MIRAGE), frozen, on the same heads and splits. Any paper will need this comparison, and it is cheap once the weights run on the lab computer. Licences and input formats need checking first.
3. **If alignment is weak:** (a) cross-modal prediction, the OCT encoder predicting SLO features along the scan lines (MIRAGE/OphMAE style); (b) LeJEPA's SIGReg in place of the EMA teacher; (c) late fusion with a separately pre-trained SLO encoder as the floor.
4. **REPA from a public OCT encoder** into our denoiser, if the baselines in step 2 show their features are clearly better than ours.
5. **Plan the full-pull tasks now:** retinal age (needs age at scan), ICD-based systemic outcomes within a fixed window (needs scan and diagnosis dates), and AMD progression if repeat visits exist. Ask Peter's table for patient-level IDs so both eyes and repeat visits of one person stay in the same split (the pilot has one scan per anon_id).

## Sources

- MIRAGE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12462498/ ; https://arxiv.org/html/2506.08900v3
- OCTCube-M: https://arxiv.org/abs/2408.11227v2 ; https://source.washu.edu/2026/06/new-ai-tools-could-help-eye-doctors-diagnose-retinal-disease-faster/
- OphMAE: https://arxiv.org/abs/2605.02714
- Sükei et al.: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11538269/
- EyeCLIP: https://arxiv.org/abs/2409.06644v1
- SLO-Net: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11262482/
- l-DAE: https://arxiv.org/abs/2401.14404v1
- Masked diffusion: https://arxiv.org/html/2308.05695v4
- REPA: https://arxiv.org/abs/2410.06940v3
- D-Cube: https://arxiv.org/pdf/2411.11087 ; diffusion pre-training for 3D segmentation: https://arxiv.org/abs/2501.19265v1 ; OrthoDiffusion: https://arxiv.org/html/2602.20752v1
- COJEPA: https://arxiv.org/abs/2607.11962 ; Rad-JEPA 3D: https://www.alphaxiv.org/abs/2607.26196v1 ; MAE vs JEPA for brain MRI: https://arxiv.org/pdf/2606.13315
- LeJEPA: https://arxiv.org/abs/2511.08544 ; code: https://github.com/rbalestr-lab/lejepa
- OCT age and mortality: https://pubmed.ncbi.nlm.nih.gov/37733221/ ; fundus age gap: https://research.monash.edu/en/publications/retinal-age-gap-as-a-predictive-biomarker-for-mortality-risk/ ; Japanese cohort: https://link.springer.com/article/10.1007/s10384-025-01205-3
- RETFound: https://www.nature.com/articles/s41586-023-06555-x
- AlzEye: https://pmc.ncbi.nlm.nih.gov/articles/PMC8928293
- AMD progression: https://www.nature.com/articles/s41591-020-0867-7 ; Morph-SSL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7616690/ ; 3DTINC: https://arxiv.org/pdf/2312.16980
- SSL scaling: https://arxiv.org/html/2405.14788v1 ; BioVFM-21M: https://arxiv.org/pdf/2505.09329
