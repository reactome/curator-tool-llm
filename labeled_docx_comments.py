"""
label_pink1_reactions.py

Build a Word (.docx) document that keeps the 300-token RESULTS chunk format and,
for every extracted reaction, HIGHLIGHTS the sentence it was pulled from and
attaches a Word COMMENT naming the reaction.

The location hint for each reaction is taken from its `evidence` field in the
extraction JSON: each evidence string is a sentence copied from the chunk text,
so we search for it inside the chunk, highlight it, and anchor a comment to it.

Usage:
    python label_pink1_reactions.py
Output:
    results/pink1_reactions_annotated.docx
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent
EXTRACTION_JSON = ROOT / "results" / "znfx1_2prev1next_extraction.json"
OUTPUT_DOCX = ROOT / "results" / "znfx1_reactions_annotated.docx"

# ---------------------------------------------------------------------------
# The RESULTS text, split into the same 300-token chunks used for extraction.
# Line breaks are preserved so hyphenated word-wraps can be rejoined.
# ---------------------------------------------------------------------------
CHUNKS = ["""
    --- chunk 0 (157 words) ---
    ZNFX1 is composed of helicase and E3 modules linked
    by a zinc-finger spacer
    Although ZNFX1 is an important factor in innate immunity, its mo-
    lecular mechanism is unexplored. The specific PAMP that acti-
    vates ZNFX1 is not fully understood, and the functional details
    of its helicase and E3 enzymatic activities are unclear. To address
    these points, we performed an in-depth structure-functional
    analysis of the human protein (Figure 1A), recombinantly
    Figure 1. ZNFX1 has a flexible bipartite structure
    (A) Domain organization of human ZNFX1.
    (B) SDS-PAGE, mass photometry, and negative-stain EM analysis of purified recombinant ZNFX1. The expected mass is 220 kDa.
    (C) Cryo-EM composite map of ZNFX1 showing the helicase core of ZNFX1.
    (D) Negative-stain EM analysis of ZNFX1. Selected 2D classes are shown as well as a 3D reconstruction with AF2 models docked inside.
    (E) Structural features and domain architecture of ZNFX1. IDR, intrinsically disordered region.
    See also Figures S1, S2, and S3.
    ll
    OPEN ACCESS

    --- chunk 1 (171 words) ---
    5996 Cell 188, 5995–6011, October 16, 2025
    Article
    expressed in insect cells. Mass photometry and negative-stain
    electron microscopy (EM) showed that the purified protein exists
    as a monomer (Figure 1B). To interrogate the structure of ZNFX1,
    we applied an integrative approach combining EM analysis with
    AlphaFold2 (AF2) predictions14 and molecular dynamics simula-
    tions (Figures 1C–1E, S1, and S2; Table S1). We first determined a
    4.3 A˚ cryo-electron microscopy (cryo-EM) structure of ZNFX1,
    which enabled us to build a model of its compact core assembled
    around the helicase domain. Negative-stain class averages re-
    vealed this core is tethered by a semi-rigid linker to the C-terminal
    segment that houses the E3 motif. The linker samples a contin-
    uum of conformations, from nearly linear to sharply bent, enabling
    the E3 module in the bent state to approach and sometimes con-
    tact the helicase scaffold. We were able to generate an EM enve-
    lope for the sharply bent conformation and dock AF2 building
    blocks into this density to obtain a model of the entire protein.

    --- chunk 2 (150 words) ---
    The central core of ZNFX1 is a UPF1-family helicase module,
    containing the characteristic 1A, 1B, 1C, and 2A sub-domains
    (Figures 1E and S3A).15,16 Analogously to UPF1-family helicases
    SEN117 and Aquarius,18 ZNFX1 has an N-terminal armadillo
    repeat (ARM) domain. Additionally, there is a small helical domain
    (1Z) residing above the 1C coiled coil and forming the contact site
    with the C-terminal segment of the protein (Figure 1E). The 1Z and
    ARMdomains are the most flexible appendices ofthe helicase, as
    indicated by their relatively low local resolution in the cryo-EM
    map (Figure S1). The helicase module is followed by a semi-rigid
    spacer, composed of a series of 12 interlocked zinc-finger do-
    mains (ZF1–ZF12), coordinating 20 zinc ions (Figure S3B). ZF1
    is closely attached to the helicase core, and along with ZF2–
    ZF3, it is well resolved in the cryo-EM reconstruction. Structural
    flexibility at the ZF4–ZF6 segment of the spacer yields the bend-

    --- chunk 3 (182 words) ---
    able joint visible in negative-stain 2D classes and some cryo-EM
    2D classes (Figure S1). Zinc fingers ZF7–ZF12 are aligned to an
    extended helical scaffold, visible at low resolution in our nega-
    tive-stain reconstruction, comprising the C-terminal segment of
    the protein, where the largest zinc finger, ZF12, forms a hinge re-
    gion. At the very C terminus, the highly conserved RZ domain
    (Figure S3C) is predicted to be loosely attached to the helical
    scaffold by a short linker. While we were only able to resolve
    the E3 module in its closed state, where it interacts with the 1Z
    domain, our cryo-EM and negative-stain data indicated that it
    mostly exists in a flexible, detached state. To investigate the
    conformational flexibility of ZNFX1 at a molecular level, we per-
    formed molecular dynamics simulations of the ZF chain
    (Figure S2B). These simulations revealed that the rigidly linked
    zinc-finger units keep the local architecture intact; however, the
    chain as a whole can undergo substantial end-to-end motions.
    This constrained flexibility enables the C-terminal segment and
    its E3 motif to sample a large space but at a defined distance

    --- chunk 4 (153 words) ---
    and orientation from the helicase domain. Overall, our structural
    analysis of ZNFX1 shows a flexible, multi-domain structure
    composed of three major parts: a UPF1-like helicase core, a flex-
    ible ZF1–ZF12 spacer, and aC-terminal scaffold harboring the RZ
    domain as the E3 active site.
    ZNFX1 is an active E3 ligase stimulated by long ssNAs
    To characterize the basic E3 properties of ZNFX1, we performed
    autoubiquitination assays with the E1 activating enzyme UBA1
    and the promiscuous E2 enzyme UBE2D2. We observed effi-
    cient autoubiquitination, reflected by the formation of large poly-
    ubiquitin chains at the top of the gel (Figure 2A). To test if this ac-
    tivity was dependent on the RZ domain, we mutated its reactive
    cysteine to alanine (C1860A [CA]) and observed that autoubiqui-
    tination activity was impaired. As ZNFX1 is activated by the
    dsRNA mimic HMW poly(I:C),9 we investigated whether NAs
    may stimulate its E3 ligase activity. In the presence of HMW

    --- chunk 5 (146 words) ---
    poly(I:C), we observed a 3-fold increase in the RZ-dependent
    auto-ubiquitination activity (Figures 2A and 2B) and a complete
    shift of the ZNFX1 band into the gel pocket when monitored by
    Coomassie staining. Consistent with a previous report,9 low mo-
    lecular weight (LMW) poly(I:C) failed to induce the E3 ligase func-
    tion of ZNFX1 (Figure 2B).
    HMW poly(I:C) is a heterogeneous mixture of dsRNA and
    ssRNA. We therefore generated 500 nucleotide (nt) long dsRNA
    and ssRNA to determine which type of NA is sensed (Figure 2B).
    Whereas dsRNA had no effect, ssRNA activated E3 function
    more strongly than HMW poly(I:C). Similarly, we observed no
    activation by dsDNA, but a 6-fold activation by single-stranded
    DNA (ssDNA). These findings reconcile how a UPF1-family heli-
    case, which would be expected to bind ssNAs,19 can sense
    HMW poly(I:C). To confirm the binding preferences of ZNFX1,
    we performed fluorescence polarization measurements using

    --- chunk 6 (180 words) ---
    20 nt NAs (ssDNA-20, dsDNA-20, ssRNA-20, and dsRNA-20)
    (Figure 2C). In agreement with the E3 activation data, we saw
    no binding of dsNAs but strong binding to ssNAs. ssDNA bound
    with a low nanomolar KD, while the affinity for ssRNA was subna-
    nomolar. The affinity increased with ssNA length, and binding to
    a 50 nt ssDNA (ssDNA-50) had a KD below 500 pM.
    The result that ZNFX1 is activated by HMW but not LMW poly
    (I:C) suggested a role in sensing long viral NA stretches. To test
    whether there is length dependence in E3 activation, we per-
    formed auto-ubiquitination assays with a series of ssDNA mole-
    cules ranging from 20 to 500 nt, always at the same mass con-
    centration (Figure 2D). Despite the strong binding of ZNFX1 to
    20 and 50 nt ssDNA, as measured by fluorescence polarization,
    we observed no activation by these molecules. Weak activation
    was seen for an 80 nt ssDNA, and then activation further
    increased with length up to ssDNA with 500 nt. Finally, we tested
    an NA mimicking physiological viral genomic ssNA. We incu-

    --- chunk 7 (149 words) ---
    bated ZNFX1 with the 7,249 nt ssDNA M13 phage circular
    genome and observed stronger activation than with all other
    NAs tested. In conclusion, the E3 activity of ZNFX1 is activated
    by ssNAs in a length-dependent manner, with the highest activity
    induced by long ssNA stretches.
    ZNFX1 ubiquitinates the 2′OH hydroxyl groups on ssRNA
    ZNFX1 has previously been proposed to function in innate im-
    mune signaling,9 which frequently uses K63-linked ubiquitin
    chains.20 Using label-free mass spectrometry, we observed
    that autoubiquitination products were predominantly K48- and
    K6-linked chains, associated with protein degradation21 and
    xenophagy,22 with almost no K63-modified ubiquitin present
    (Figure 2E).
    RNF213, the E3 ligase functionally related to ZNFX1, has been
    shown to target non-proteinaceous substrates and bacterial
    lipids, ubiquitinating a hydroxyl group in the lipid A moiety of
    ll
    OPEN ACCESS
    Cell 188, 5995–6011, October 16, 2025 5997
    Article
    LPS.8 As ZNFX1 shares the same catalytic RZ domain, and

    --- chunk 8 (147 words) ---
    ssRNA has hydroxyl nucleophiles (Figure 2F), we tested whether
    ZNFX1 could directly ubiquitinate ssRNA. We generated fluores-
    cently labeled 500 nt ssRNA and performed ubiquitination reac-
    tions, monitoring band shifts of the fluorescently labeled ssRNA
    (Figure 2G). We observed the formation of a band above the
    ssRNA consistent with the size of a mono-ubiquitinated species.
    Ubiquitination of ssRNA was ATP and ZNFX1 dependent and
    Figure 2. ZNFX1 is an RNA-ubiquitinating E3 ligase stimulated by long ssNAs
    (A) Ubiquitination assay using either WT ZNFX1 or a CA variant in the absence or presence of HMW poly(I:C). The upper panel shows autoubiquitination of ZNFX1
    by Coomassie staining. The lower panel shows ubiquitinated products by measuring the signal of DyLight488-ubiquitin. Representative of two experimental
    replicates.
    (B) Autoubiquitination assay with different NA stimulators. On the right, the fluorescent autoubiquitination signal from triplicate experiments is quantified relative to
    the reaction without stimulator.

    --- chunk 9 (160 words) ---
    (C) Fluorescence polarization experiments performed in triplicate showing the affinity of ZNFX1 for the indicated NAs. Dissociation constants are shown with 95%
    confidence intervals. Error bars show standard deviation.
    (D) Autoubiquitination assay using different lengths of ssDNA as stimulators of ZNFX1.
    (E) Mass spectrometry analysis of ubiquitin linkages in autoubiquitinated ZNFX1 in the presence of poly(I:C). Quantification shows the number of modified
    peptides compared with total peptides.
    (F) Structure of ssRNA showing the position of hydroxyl groups.
    (G) RNA ubiquitination assay by ZNFX1 monitoring Cy5-labeled ssRNA-500. After the ubiquitination reaction, one sample was quenched with EDTA and treated
    with the deubiquitinase USP21.
    (H) RNA ubiquitination assay followed by incubation with Tris pH 10 for the indicated time.
    See also Figure S4.
    ll
    OPEN ACCESS
    5998 Cell 188, 5995–6011, October 16, 2025
    Article
    could be removed by the promiscuous deubiquitinase USP21.
    To determine how this ubiquitin was attached to RNA, we
    exposed the sample to basic conditions, expected to induce hy-

    --- chunk 10 (167 words) ---
    drolysis of ester linkages but not amide linkages. While the au-
    toubiquitination band was unaffected, the RNA-Ub species
    was removed (Figure 2H), suggesting that the modification is
    through an ester linkage to a hydroxyl group, as found on the
    ribose. To confirm that ZNFX1 has hydroxyl ubiquitination activ-
    ity, we used the model substrate maltoheptaose and observed
    ZNFX1-catalyzed ubiquitination of the sugar (Figure S4A).
    HUWE1, used as a control ubiquitin ligase known to have highly
    promiscuous E3 activity, was unable to ubiquitinate either malto-
    heptaose or ssRNA (Figure S4B). Recent studies have shown
    that the E3 ligase DTX3L can ubiquitinate the 3′OH of NAs termi-
    nating in adenine, as this resembles the DELTEX E3 family sub-
    strate ADP ribose.23,24 We investigated whether ZNFX1 had any
    specificity for the termini of NAs or any modifications of these,
    which would be indicative of a viral RNA origin. We observed
    no effect of 5′ RNA modifications or RNA circularization, which
    would remove the 3′ hydroxyl, on either ZNFX1 activation or

    --- chunk 11 (150 words) ---
    RNA ubiquitination activity (Figure S4C). Together, these data
    reveal that the only other hydroxyl on RNA, the 2′ ribose hydroxyl,
    is the target of ZNFX1, which otherwise is rather unselective in
    modifying ssNA molecules.
    ATP-driven translocation along NAs is required to
    induce E3 activity
    We next wanted to understand how ssNAs interact with ZNFX1
    to stimulate its E3 activity. UPF1-family helicases couple ATP
    binding and hydrolysis with ssNA binding and 5′ to 3′ transloca-
    tion.19,25,26 We observed that ZNFX1 had strong ATPase activity
    in the presence of ssDNA (Figure 3A), which is consistent with
    ZNFX1 being an ssNA translocase. To test this idea, we per-
    formed a DNA exchange experiment to measure dissociation
    from 100 nt ssDNA (ssDNA-100), where dissociation could
    result from either spontaneous unbinding or ATP-dependent
    Figure 3. ZNFX1 translocase activity is required for maximal E3 activation
    (A) NADH-coupled assay showing ATPase activity of ZNFX1 with excess ssDNA-20.

    --- chunk 12 (153 words) ---
    (B) DNA exchange experiments measuring ZNFX1 dissociation from ssDNA-100 by the loss of fluorescence polarization. Individual data points are shown.
    (C and D) E2∼Ub discharge assays showing the stimulatory effect of ATP. Loss of E2∼Ub in triplicate experiments is quantified in the plots below each gel with
    error bars showing standard deviation. The reactions in (D) were performed for 5 min in the presence of M13.
    ll
    OPEN ACCESS
    Cell 188, 5995–6011, October 16, 2025 5999
    Article
    Figure 4. ZNFX1 has a distinct ubiquitin-transfer mechanism
    (A) Autoubiquitination assay of ZNFX1 in the presence of M13 ssDNA with either UBE2D2 or UBE2L3 as E2 enzymes.
    (B) E2∼Ub discharge assays using either WT ubiquitin or ubiquitin with the I44A mutation in the presence of M13 ssDNA, comparing ZNFX1 with RNF213.
    (C) AF2 prediction of the complex formed between ZNFX1 and UBE2D2 showing only the E3 module of ZNFX1. Important functional residues on ZNFX1 are
    indicated.

    --- chunk 13 (132 words) ---
    (D) Autoubiquitination assay of ZNFX1 variants in the presence of M13 ssDNA. CA, C1860A; QA, Q1623A; FFAA, F1574A/F1575A; IS, I1835S; HA, H1881A.
    (E) Proposed ubiquitin-transfer mechanism of ZNFX1.
    (F) AF2 prediction of the complex formed between the ZNFX1 RZ domain and ubiquitin.
    (G) E2∼Ub discharge experiment using excess ZNFX1 variant (1.2 μM) over E2 (1 μM) in the presence of M13 ssDNA and ATP quenched in either reducing or non-
    reducing SDS-PAGE buffer.
    (H) Maltoheptaose ubiquitination endpoint assays in the presence of M13 ssDNA.
    (I) RNA ubiquitination assay as in Figure 2G with the indicated ZNFX1 variant.
    (legend continued on next page)
    ll
    OPEN ACCESS
    6000 Cell 188, 5995–6011, October 16, 2025
    Article
    translocation off the ssDNA 3′ end. We pre-bound ZNFX1 to fluo-
    rescent ssDNA, resulting in a characteristic fluorescence polari-

    --- chunk 14 (160 words) ---
    zation increase, and then applied an excess of unlabeled ssDNA-
    100 to quench dissociated ZNFX1. In the absence of ATP or
    presence of the non-hydrolysable ATP analog AMPPNP, we
    observed very slow exchange, confirming the tight association
    of ZNFX1 to long ssDNA (Figure 3B). However, in the presence
    of ATP, the protein rapidly dissociated from ssDNA. To directly
    show that dissociation was due to ZNFX1 translocation, we
    blocked the 3′ end of the fluorescent ssDNA by biotin-streptavi-
    din, halting translocation at this end. In the presence of strepta-
    vidin, ATP-stimulated dissociation was blocked (Figure 3B). This
    was not the case when ssDNA was blocked at the 5′ end by
    biotin-streptavidin. Thus, like other UPF1-family members,
    ZNFX1 has 5′ to 3′ translocase activity.
    To test whether ATP plays a role in stimulating the NA-depen-
    dent E3 activity, we performed E2∼Ub discharge assays starting
    with purified UBE2D2∼Ub conjugate, bypassing the ATP-
    dependent E1 enzyme (Figure 3C). ATP alone had no effect on

    --- chunk 15 (180 words) ---
    the slow E2∼Ub discharge by ZNFX1. The M13 ssDNA alone
    increased the rate to some extent, but only in the presence of
    both ssDNA and ATP did we observe rapid discharge and
    concomitant smearing of the ZNFX1 band into the gel pocket.
    We considered that either ATP binding itself could induce an
    active conformation or that ATP hydrolysis may power transloca-
    tion as required to boost E3 function. To discriminate these hy-
    potheses, we tested a range of non-hydrolysable ATP analogs,
    as well as the transition state analog ADP.AlFx and the product
    ADP + Pi (Figure 3D). We only observed robust E3 activation
    for ATP itself, indicating that ATP hydrolysis rather than binding
    is required for activation. Together, our findings suggest that
    the ATP-driven translocase activity of ZNFX1 is critical to stimu-
    late its E3 ligase activity.
    ZNFX1 uses a multi-step E3 mechanism linked to its
    bipartite active site organization
    To understand the ubiquitination mechanism of ZNFX1, we first
    screened which E2 enzymes can work with ZNFX1 (Figures 4A
    and S5A). UBE2D variants acted as an E2 for ZNFX1, whereas

    --- chunk 16 (133 words) ---
    all other E2s, including the transthiolation-specific UBE2L3,
    which works with HECTs, RBRs, and RNF213, but not RING-
    type E3 ligases,27,28 did not promote auto-ubiquitination.
    To determine whether ZNFX1 functions similarly to RING-type
    ligases, which catalyze ubiquitin transfer by stabilizing the
    activated ‘‘closed’’ state of the E2-Ub conjugate,29 we generated
    UBE2D2∼Ub(I44A), where the hydrophobic patch on ubiquitin is
    mutated, preventing formation of the closed state (Figure 4B).
    For RNF213, this mutation had no effect on the rate of discharge,
    as expected for a transthiolation E3 ligase.30 However, for
    ZNFX1, the Ub(I44A) mutation completely blocked discharge,
    showing that ZNFX1 requires the closed state of the E2∼Ub con-
    jugate for ubiquitin transfer. Overall, this mechanistic profile is
    reminiscent of the ‘‘RING-Cys-Relay’’ E3 MYCBP2,31 which is
    also a transthiolation E3 enzyme that uses the closed state of

    --- chunk 17 (159 words) ---
    E2∼Ub of UBE2D enzymes.
    To explore how ZNFX1 binds UBE2D2, we predicted the com-
    plex between the two full-length proteins by AF2.32 We observed
    a high-confidence model where UBE2D2 bound to the C termi-
    nus of the ZF chain (Figures 4C, S5B, and S5C). Upon closer in-
    spection, it became clear that ZF12 is an atypical RING domain,
    which we term the Z-RING. This was not previously noticed at
    the sequence level because it lacks one of the two zinc coordi-
    nation sites, similar to the unrelated RING domains from Siz133
    and UBR434 (Figure S5D). The Z-RING domain also contains a
    potential linchpin residue, which is expected to facilitate the for-
    mation of the E2∼Ub closed state.35 In the AF2 model, the
    Z-RING uses a similar E2-binding interface as a canonical
    RING domain (Figure S5C). Two phenylalanines, F1574 and
    F1575, on ZNFX1 form a hydrophobic patch that interacts with
    F62, P61, and P95 on UBE2D2. To investigate the importance

    --- chunk 18 (151 words) ---
    of the Z-RING domain for E3 activity, we generated ZNFX1 var-
    iants where either the linchpin glutamine (QA) or the interacting
    phenylalanines (FFAA) were mutated to alanine. In autoubiquiti-
    nation assays, both variants had severely diminished activity,
    like the RZ CA mutation (Figure 4D). The FFAA variant completely
    abrogated activity, which implies that the Z-RING is the E2 bind-
    ing platform on ZNFX1. The CA variant showed no HMW polyu-
    biquitin chains, and instead, residual ubiquitination resembled
    mono-ubiquitination of ZNFX1. Thus, without the RZ domain,
    the Z-RING alone can directly but weakly catalyze autoubiquiti-
    nation. The Z-RING QA variant instead had a wild-type (WT)-like
    ubiquitination
    pattern
    with
    weakened
    intensity.
    Therefore,
    without the linchpin glutamine, ubiquitin transfer to the RZ finger
    can still occur, but it is inefficient. These data are consistent with
    a sequential ubiquitin-transfer mechanism (Figure 4E): first,
    ZNFX1 uses its Z-RING to bind E2∼Ub and activate it by stabi-

    --- chunk 19 (169 words) ---
    lizing its closed state. In a second step, the ubiquitin is trans-
    ferred to the reactive cysteine of the RZ domain, which then, in
    a third step, catalyzes substrate ubiquitination.
    To understand how the RZ domain catalyzes both lysine and
    hydroxyl ubiquitination, we generated a model of the RZ∼Ub
    thioester state using AF2 (Figures 4F and S5E). In this model,
    the N-terminal helix of the RZ domain binds to the hydrophobic
    patch of ubiquitin, positioning the ubiquitin C-terminal tail to
    interact with the catalytic cysteine of the RZ. Intriguingly, this
    model resembles the activated closed state of an E2∼Ub conju-
    gate (e.g., Figure S5C). To understand the role of this interaction
    for the E3 ligase activity of ZNFX1, we mutated an isoleucine at
    the center of this interface (I1835S, IS) and saw strongly reduced
    autoubiquitination activity (Figure 4D). We next tested which step
    of the reaction requires the RZ/Ub interaction by performing an
    E2∼Ub discharge assay under single turnover conditions to
    follow the ubiquitin-transfer reaction and using SDS-PAGE under

    --- chunk 20 (151 words) ---
    either reducing or non-reducing conditions to distinguish iso-
    peptide and thioester bonds (Figure 4G). For WT ZNFX1, we
    observed
    complete
    conversion
    of
    E2∼Ub
    to
    isopeptide
    (J) Schematic model of the molecular features contributing to RNA ubiquitination by ZNFX1.
    (K) E2∼Ub discharge experiment as in (G) in the absence or presence of M13 ssDNA and ATP.
    (L) Lysine discharge assays to measure Z-RING activity where ZNFX1-CA is mixed with E2∼Ub in the presence of lysine as a nucleophile.
    See also Figure S5.
    ll
    OPEN ACCESS
    Cell 188, 5995–6011, October 16, 2025 6001
    Article
    ZNFX1-Ub and hydrolyzed free Ub, which were both unaffected
    by reductant. For I1835S ZNFX1, we saw no products but
    instead the buildup of a trapped ZNFX1∼Ub thioester species,
    as indicated by the sensitivity of this band to reduction. Thus,
    the RZ/Ub closed state is required for efficient nucleophilic
    attack of ZNFX1∼Ub by water or lysines. RZ domains have a

    --- chunk 21 (167 words) ---
    conserved histidine adjacent to the catalytic cysteine, which
    has been proposed to activate incoming nucleophiles, as seen
    for RBR E3 ligases.36 Mutation of this histidine (H1881A, HA) re-
    sulted in a similar autoubiquitination defect and buildup of the
    ZNFX1∼Ub intermediate, confirming this idea (Figures 4D and
    4G). We considered that increasing the reactivity of the
    ZNFX1∼Ub intermediate would be particularly important for
    weaker hydroxyl nucleophiles. Using a maltoheptaose ubiquiti-
    nation endpoint assay, we first compared the I1835S and
    H1881A variants with the WT protein and the QA variant, which
    is defective in the first step of the ZNFX1 reaction (Figure 4H).
    While the QA variant could still catalyze maltoheptaose ubiquiti-
    nation, the IS and HA variants could only catalyze autoubiquiti-
    nation. We also observed that the IS and HA variants could not
    perform RNA ubiquitination (Figure 4I). Thus, the RZ/Ub closed
    state and catalytic histidine are essential for the hydroxyl ubiqui-
    tination activity of ZNFX1 (Figure 4J).
    We next investigated which of the three reaction steps is

    --- chunk 22 (157 words) ---
    modulated by ssNAs. First, we carried out E2∼Ub discharge re-
    actions under single turnover conditions (Figure 4K). In the
    absence of ssDNA and ATP, the ZNFX1∼Ub intermediate did
    not accumulate, showing that the third step, substrate ubiquiti-
    nation, is not modulated by ssNAs. To determine whether ssNAs
    modulate the first step, E2∼Ub activation, we performed a
    discharge assay with the RZ-defective CA variant and provided
    lysine instead as an artificial ubiquitin acceptor (Figure 4L). We
    observed E2∼Ub discharge onto lysine, confirming that the
    Z-RING is an active E3 domain, but discharge was not substan-
    tially affected by ssNA and ATP, showing that this is not the main
    regulated step. Thus, NAs must control the second step, ubiqui-
    tin transfer from the Z-RING-bound E2∼Ub to the RZ domain.
    ZNFX1 oligomerizes on ssDNA to yield ubiquitin-coated
    nucleoprotein particles
    The AF2 model of the ZNFX1-UBE2D2 complex gave us the first
    insight into how regulation of ubiquitin transfer between the two

    --- chunk 23 (171 words) ---
    E3 motifs could occur (Figure 4C). We noted that the RZ finger is
    located remotely from the Z-RING domain, being shielded by the
    rigid helical scaffold. The two reactive cysteines are predicted to
    be 95 A˚ apart, and the flexible RZ domain linker of 6 residues
    would be unable to bridge this distance, preventing ubiquitin
    transfer (Figure S6A). We hypothesized that ssNA binding could
    bring two ZNFX1 molecules in proximity to enable ubiquitin
    transfer in trans. To test the activation mechanism, we per-
    formed a titration experiment where we added increasing
    amounts of M13 ssDNA to an autoubiquitination reaction
    (Figure 5A). We observed a pronounced ‘‘hook’’ effect linking
    ssDNA-mediated clustering of ZNFX1 with the upregulation of
    its E3 ligase activity. At low concentrations of ssDNA, NAs are
    present in insufficient amounts to enhance the local concentra-
    tion of the E3 ligase. At moderate ssDNA levels, ZNFX1 mole-
    cules get closely packed along ssDNA strands, which is corre-
    lated
    with
    increased
    ubiquitination
    activity.
    Yet,
    as
    the
    concentration
    of
    ssDNA
    increases,
    the
    spacing

    --- chunk 24 (153 words) ---
    between
    ZNFX1 molecules on the ssDNA strands becomes larger, disfa-
    voring complex formation and active site complementation and
    resulting in decreased activity. While optimizing the purification
    of ZNFX1, we noticed that the protein forms soluble aggregates
    in the presence of imidazole (Figure S6B). We exploited this ef-
    fect to validate the regulatory mechanism of ZNFX1. We
    observed that aggregated ZNFX1 catalyzed E2∼Ub discharge
    at a faster rate than monomeric ZNFX1 (Figure S6C), showing
    that induced proximity in the absence of ssNA can acti-
    vate ZNFX1.
    Immune dsNA sensors, like cGAS, MDA5, and RIG-I, form fil-
    aments on dsNAs,37–39 which would be one way of inducing
    proximity. We thus set out to determine how ZNFX1 molecules
    arrange on activating ssNA. We first used atomic force micro-
    scopy (AFM) to resolve how ZNFX1 interacts with the best acti-
    vator, M13 ssDNA. ZNFX1 alone was monodisperse with some
    aggregates (Figure 5B). M13 ssDNA formed irregular, loosely

    --- chunk 25 (177 words) ---
    folded structures.40 When we added excess ZNFX1 to M13
    ssDNA in the presence of ATP, we observed that ZNFX1 con-
    verted the M13 ssDNA into a dense protein-DNA aggregate.
    We next used a lower ratio of ZNFX1 to DNA to visualize how
    this aggregate is built up. In these conditions, we observed indi-
    vidual smaller clusters on the M13 ssDNA. Quantification
    showed that in the presence of ATP, a higher percentage of
    M13 DNA had ZNFX1 bound, yielding overall larger ZNFX1:
    DNA clusters (Figure 5C). However, at saturating ZNFX1 concen-
    trations, the DNA appeared to be completely coated regardless
    of ATP.
    To visualize these ZNFX1-ssDNA clusters with higher resolu-
    tion, we used negative-stain EM (Figure 5D). When applying
    ZNFX1 in excess, we observed the formation of large, spherical
    nucleoprotein particles, like those seen with AFM. When we
    increased the concentration of DNA to alter the ZNFX1:DNA ra-
    tio, we observed numerous smaller clusters, suggesting that
    ZNFX1 binds cooperatively to DNA. To exclude that this clus-
    tering is due to the interconnected fold of the circular M13 ssDNA

    --- chunk 26 (157 words) ---
    itself, we repeated the experiment with a linear 500 nt ssDNA.
    Again, we observed dense aggregates, suggesting that ZNFX1
    can compact ssDNA into nucleoprotein particles.
    ZNFX1 activation involves the formation of a specific
    ZNFX1 dimer
    To gain mechanistic insight into ZNFX1 activation by ssNA-
    induced oligomerization, we reconstituted complexes with min-
    imal substrates. Notably, the 50 nt ssDNA-50 cannot activate
    ZNFX1, while the 100 nt ssDNA-100 can (Figure 2D). A possible
    scenario is that only one ZNFX1 molecule can bind to ssDNA-50,
    excluding the formation of an active E3 complex with a partner
    subunit. However, electrophoretic mobility assays with both
    the ssDNA-50 and ssDNA-100 showed similar formation of
    ZNFX1-DNA species containing multiple ZNFX1 molecules
    (Figure 6A), meaning that induced proximity alone cannot fully
    explain activation.
    To determine whether ssDNA-100 induces a specific E3 com-
    plex, we performed negative-stain EM of ZNFX1 with either
    ssDNA-50 or ssDNA-100 in a 3:1 ratio. In the presence of
    ll
    OPEN ACCESS

    --- chunk 27 (169 words) ---
    6002 Cell 188, 5995–6011, October 16, 2025
    Article
    ssDNA-100, but not with ssDNA-50 or without DNA, we now
    observed that a third of the particles were in 2D classes that
    resembled specific, well-defined ZNFX1 dimers (Figures 6B
    and S7A). When aligning the dimer classes to the class of the
    well-defined monomer, we could observe two types of dimers,
    both of which were held together by the ARM domain. To gain
    Figure 5. ZNFX1 clusters on long ssNAs
    (A) Autoubiquitination assay of ZNFX1 in the presence of increasing concentrations of M13 ssDNA. The autoubiquitination quantified from triplicate experiments is
    plotted against the calculated stoichiometry of nucleotides per ZNFX1 on the right. Error bars show standard deviation.
    (B) AFM analysis of ZNFX1 and M13 ssDNA either separately or mixed at different ratios in the presence of ATP. Images are colored by height.
    (C) Representative AFM images of limiting ZNFX1 binding to M13 ssDNA in the presence of absence of ATP. On the right, the fraction of M13 bound by ZNFX1 is

    --- chunk 28 (165 words) ---
    quantified for triplicate experiments performed under the indicated conditions. Error bars and significance are calculated from the SD and a two-tailed t test
    between replicates. All measured M13 ssDNA maximum heights are shown as violin plot distributions in the graph on the right, where the significance is
    calculated by a two-tailed t test between the distributions of all heights from all replicates.
    (D) Negative-stain analysis of ZNFX1 in the absence and presence of M13 ssDNA at different ratios or ssDNA-500.
    See also Figure S6.
    ll
    OPEN ACCESS
    Cell 188, 5995–6011, October 16, 2025 6003
    Article
    Figure 6. ZNFX1 forms a specific dimer on activating ssNA
    (A) Determination of the stoichiometry of ZNFX1 binding to short ssDNA using EMSAs.
    (B) Negative-stain 2D classes of ZNFX1 in the presence of 100 nt ssDNA. Classes are rotated to show one ZNFX1 molecule from each class in the same
    orientation.
    (C) Crystal structure of the ZNFX1 ARM domain homodimer with one molecule in blue and the other in gray.

    --- chunk 29 (155 words) ---
    (D) A model of the full-length ZNFX1 homodimer generated by superimposing two full-length ZNFX1 structures onto the ARM homodimer crystal structure.
    (E) Direct docking of the full-length dimer model into a dimeric 2D class from (B).
    (F) Autoubiquitination assay with the indicated ZNFX1 variant in the presence or absence of ssDNA-100.
    (G) Modeling of the binding of ssRNA using an ssRNA-bound structure of the related helicase UPF1 (PDB: 2XZL).
    (H) Model of how unidirectional translocation on long ssDNA could result in homodimerization.
    (I) Location of patient mutations, shown in pink, on the structure of ZNFX1. Autoubiquitination assay with the indicated ZNFX1 variants and conditions.
    See also Figure S7.
    ll
    OPEN ACCESS
    6004 Cell 188, 5995–6011, October 16, 2025
    Article
    molecular insight into ARM-mediated homodimerization, we
    generated the isolated ARM domain and determined its structure
    using X-ray crystallography (Table S2). The asymmetric unit of
    the crystal lattice was composed of a complex of two ARM do-

    --- chunk 30 (170 words) ---
    mains interacting with a hydrophobic 975 A˚ 2 interface and clas-
    sified as a stable homodimer by the PISA server41 (Figures 6C
    and S7B). By aligning the structure of full-length ZNFX1 onto
    the ARM domain homodimer, we modeled the full-length
    ZNFX1 homodimer (Figure 6D). This model fit well into one of
    the two dimer negative-stain classes (Figure 6E). Moreover, in
    the homodimer model, the E3 modules are brought together
    in an antiparallel arrangement, which could facilitate transfer in
    trans between the Z-RING of one with the RZ of the other. To
    determine the importance of this arrangement to activate E3
    function, we generated a ZNFX1 variant with the ARM domain
    deleted. While this variant had similar basal E3 activity to WT
    ZNFX1, we observed no activation by 100 nt ssDNA (Figure 6F)
    and much reduced activation by M13 ssDNA (Figure S7C).
    To understand how the ZNFX1 dimer is specifically acti-
    vated by ssNAs having a certain length, we modeled ssRNA
    into both ZNFX1 molecules using a crystal structure of RNA-

    --- chunk 31 (182 words) ---
    bound UPF1 (Figure 6G).42 Notably, the bound RNA runs in
    opposite directions for the two ZNFX1 molecules in the dimer.
    This antiparallel binding offers an explanation of both why the
    dimer does not form on the smaller 50 nt ssDNA and how uni-
    directional translocation could enhance dimer formation on
    long ssNA molecules (Figure 6H). The NA translocation path
    must be long enough to form NA loop structures, allowing
    ZNFX1 molecules that move in the same direction to run into
    each other, interact via their ARM domain, and complement
    their E3 active sites.
    While most ZNFX1-deficiency patient mutations cause delete-
    rious truncations, we noticed that two mutations affect zinc-
    coordinating cysteines within the ZF chain. We hypothesized
    that these mutations would impair the rigidity of this spacer
    and thus prevent the correct arrangement of the E3 modules
    upon NA-mediated dimerization. We thus generated ZNFX1 var-
    iants with the C1264S and C1292S patient mutations. The
    variant proteins behaved well during purification and had similar
    NA binding affinity as the WT protein (Figure S7D). However, they
    were strongly impaired in M13-stimulated E3 ligase activity

    --- chunk 32 (159 words) ---
    (Figure 6I). As well as showing the importance of the ZF chain
    for E3 ligase activity, these disease mutations imply that it is
    NA-activated E3 ligase activity rather than NA binding alone
    that is essential for the function of ZNFX1 in humans.
    ZNFX1 protects against IFN-induced cell death
    Our in vitro data show that ZNFX1 condenses ssNAs into ubiqui-
    tin-coated nucleoprotein particles. To determine whether ZNFX1
    can use this mechanism to target foreign RNA in cells, we first
    generated RPE1 ZNFX1−/−
    knockout (KO) cells (Figure S8A)
    and then transiently expressed mCherry-tagged ZNFX1 in these.
    We electroporated fluorescently labeled ssRNA-500, which
    robustly activates ZNFX1 in vitro, into the cells in the presence
    of fluorescent ubiquitin (Figures 7A and S8B). ZNFX1 colocalized
    in foci with the RNA and ubiquitin, suggesting that ZNFX1 can
    recognize foreign RNA in cells to form ubiquitin-positive conden-
    sates. We next investigated what function this might have.
    ZNFX1 has been previously linked to the innate immune system,

    --- chunk 33 (169 words) ---
    acting through MAVS to induce the IFN response upon detection
    of viral RNA.9 To investigate this function, we electroporated
    ssRNA-500 into WT and ZNFX1−/−RPE1 cells and then quanti-
    fied expression of ISGs by qPCR (Figure S8C). We observed
    ISG induction, presumably because the RNA was in vitro tran-
    scribed,43,44 but no difference between the WT and ZNFX1 KO
    cells, arguing against a role for ZNFX1 in enhancing IFN signaling
    in response to ssRNA.
    Intriguingly, we had observed that the viability of RPE1
    ZNFX1−/−cells was highly sensitive to ssRNA electroporation.45
    To quantify this, we used a cell competition assay where WT and
    KO cells with different fluorescent markers were mixed and then
    electroporated with either ssRNA-500 or buffer. To control for
    clonal effects, we employed a bulk CRISPR interference
    approach to knock down ZNFX1, comparing two ZNFX1-target-
    ing CRISPRi guides against a control guide, and observed a 5- to
    10-fold loss of cells with ZNFX1 knockdown (Figures 7B and
    S8D). We next tested whether this phenotype relates to the pro-

    --- chunk 34 (186 words) ---
    posed role of ZNFX1 to function with MAVS.9 However, knock-
    down of MAVS had no effect on cell viability in these conditions
    either alone or in combination with a ZNFX1 knockdown
    (Figure S8E), suggesting this phenotype is independent of
    MAVS. One known cause of toxicity induced by in vitro tran-
    scribed ssRNA is the strong induction of the IFN response.43,44
    To determine whether IFN signaling could be the cause of cell
    death, we repeated the assay using IFN stimulation instead of
    RNA electroporation and observed similar loss of viability in
    ZNFX1 KO cells, suggesting it is the IFN response rather than
    the RNA that is toxic to these cells (Figure 7B).
    The immune response dysregulation we observed is consis-
    tent with the autoimmune symptoms seen in ZNFX1 deficiency
    patients.10–13 To further investigate why RPE1 ZNFX1−/−cells
    were dying upon IFN stimulation, we performed RNA sequencing
    (RNA-seq) analysis of the WT and KO cells 3 days after IFN stim-
    ulation (Figures 7C and S8F; Table S3). In the KO cells, we
    observed strong upregulation of genes involved in the unfolded
    protein response, including apoptosis effectors of the integrated

    --- chunk 35 (182 words) ---
    stress response, such as DDIT3. As IFN-induced cell death is in-
    dependent of foreign RNA, it is likely ZNFX1 has a role in target-
    ing host RNA under these conditions. Indeed, we observed ubiq-
    uitin-positive ZNFX1 foci even in the absence of foreign RNA
    (Figure S8B), which agrees with previous reports that ZNFX1
    binds to host mRNA46 and is a component of stress granules,
    which also contain mRNA.10 The induction of the unfolded pro-
    tein response in the absence of ZNFX1 suggests that mRNA
    metabolism must be tightly regulated during an IFN response
    to maintain cell viability.
    Using this strong cell viability phenotype, we next tested the
    importance of the E3 ligase activity of ZNFX1 for its cellular func-
    tion. We reconstituted our KO cells with ZNFX1 constructs con-
    taining the mutations examined in vitro (Figures 7D and S8G). In
    contrast to WT ZNFX1, variants that were strongly defective for
    ubiquitination activity in vitro, such as the CA, FFAA, IS, and pa-
    tient mutations C1264S and C1292S, were unable to rescue the
    phenotype (Figure 7D). Thus, we can conclude that the ubiquiti-

    --- chunk 36 (149 words) ---
    nation activity of ZNFX1 is important for cell viability upon IFN-in-
    duced stress. Ectopic WT ZNFX1 did not fully rescue the
    ll
    OPEN ACCESS
    Cell 188, 5995–6011, October 16, 2025 6005
    Article
    Figure 7. Cellular function of ZNFX1
    (A) Confocal microscopy of RPE1 ZNFX1−/−cells transfected with mCherry-ZNFX1 and electroporated with AF647-ssRNA-500 and DyLight488-Ub.
    (B) Competition cell death assay of RPE1 cells after electroporation with 500 nt ssRNA or buffer. sgRNA-transduced and empty vector-transduced cells were
    mixed 50:50 before electroporation. The ratio of cell populations was measured by flow cytometry at days 0 and 7. The log2 fold change of the relative sgRNA-
    transduced population between days 0 and 7 is shown for three biological replicates for each condition, with the average and standard deviation also plotted.
    (C) Volcano plot of RNA-seq analysis comparing WT and ZNFX1−/−RPE1 cells 3 days after IFN-β treatment. Unfolded protein response genes with a −log10(false

    --- chunk 37 (173 words) ---
    discovery rate [FDR]) over 100 are highlighted in red. Genes with a −log10(FDR) exceeding the calculable range are plotted at 310.
    (D) Competition cell death assay of ZNFX1−/−RPE1 cells complemented with the indicated cDNAs by lentiviral transduction versus WT RPE cells 7 days after IFN-
    β or buffer treatment. Three biological replicates are shown for each condition with standard deviation plotted.
    (E) Proposed model for ZNFX1 function in cells.
    (F) Unrooted phylogenetic analysis of ZNFX1-related helicases found by an unbiased search. Red squares indicate that sequences contain an RZ domain.
    See also Figure S8.
    ll
    OPEN ACCESS
    6006 Cell 188, 5995–6011, October 16, 2025
    Article
    phenotype, presumably due to its relatively low expression level
    compared with endogenous ZNFX1 under IFN-induced condi-
    tions (Figure S8G). To confirm the importance of ZNFX1 E3 activ-
    ity under native expression conditions, we knocked the CA mu-
    tation into the genomic ZNFX1 locus and observed a loss of
    viability similar to complete ZNFX1 KO (Figure S8H). Of note,
    the ΔARM variant was able to rescue the phenotype moderately,

    --- chunk 38 (94 words) ---
    which may be due to the residual E3 ligase activity of this variant
    in the presence of long ssNAs (Figure S7C) combined with the
    higher expression of the protein in cells compared with the WT
    (Figure S8G). Interestingly, the QA variant was able to rescue
    the phenotype while the IS variant could not (Figure 7D), despite
    these having similar autoubiquitination defects (Figure 4D).
    Notably, the QA variant can still catalyze hydroxyl ubiquitination,
    whereas the IS variant cannot (Figure 4H), suggesting an impor-
    tant role for hydroxyl ubiquitination for the function of ZNFX1
    in cells. 
"""]


def normalize(text: str) -> str:
    """Rejoin hyphenated line-wraps, drop newlines, collapse whitespace."""
    text = re.sub(r"-\n", "", text)        # word-\nwrap -> wordwrap
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_span(evidence: str, hay: str):
    """
    Locate `evidence` inside `hay` (both normalized).
    Returns (start, end, quality) or None.
    quality: 1.0 exact, else fraction of evidence chars aligned (fuzzy).
    """
    ev = normalize(evidence)
    if not ev:
        return None

    idx = hay.find(ev)
    if idx != -1:
        return idx, idx + len(ev), 1.0

    # Fuzzy fallback for PDF word-wraps that lost their hyphen
    # (e.g. "phosphor\nylation" -> "phosphor ylation" != "phosphorylation").
    sm = SequenceMatcher(None, hay, ev, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    if matched < 0.6 * len(ev):
        return None
    start = blocks[0].a
    end = blocks[-1].a + blocks[-1].size
    return start, end, matched / len(ev)


def build_comment(rxn: dict) -> str:
    """Human-readable comment body describing the reaction."""
    ca = rxn.get("catalystActivity") or {}
    lines = [
        f"REACTION: {rxn.get('name', '(unnamed)')}",
        f"Type: {rxn.get('reactionType', '?')}"
        f"  |  Confidence: {rxn.get('confidence', '?')}",
        f"Input: {', '.join(rxn.get('input') or []) or '—'}",
        f"Output: {', '.join(rxn.get('output') or []) or '—'}",
    ]
    if ca.get("catalyst"):
        mf = ca.get("molecularFunction") or ""
        lines.append(f"Catalyst: {ca['catalyst']}" + (f" ({mf})" if mf else ""))
    reg = rxn.get("regulatedBy") or []
    if reg:
        reg_str = "; ".join(
            f"{r.get('regulator')} [{r.get('regulationType')}]" for r in reg
        )
        lines.append(f"Regulated by: {reg_str}")
    summ = (rxn.get("summation") or {}).get("text")
    if summ:
        lines.append(f"Summation: {summ}")
    ctx = rxn.get("context_used")
    if ctx and ctx != "none":
        lines.append(f"(extraction used context: {ctx})")
    return "\n".join(lines)


def attach_comment(doc, runs, text, author="reaction-extractor", initials="RX"):
    """Attach a comment to the given runs, or fall back to appending comment text."""
    if hasattr(doc, "add_comment"):
        try:
            doc.add_comment(runs, text=text, author=author, initials=initials)
            return
        except Exception:
            pass

    if not runs:
        return
    fallback = text.replace("\n", "  |  ")
    runs[-1].add_text(f" [{author}: {fallback}]")


def locate_evidence(evidence: str, norm_chunks):
    """
    Find `evidence` across ALL chunks; return (chunk_idx, start, end, quality)
    for the best match, or None. The JSON `chunk_index` is a DIFFERENT
    segmentation from the pasted 300-token chunks, so we cannot trust it for
    placement — we search the whole document and annotate where the sentence
    actually lives.
    """
    best = None  # (quality, chunk_idx, start, end)
    for ci, hay in enumerate(norm_chunks):
        span = find_span(evidence, hay)
        if span is None:
            continue
        s, e, q = span
        if best is None or q > best[0]:
            best = (q, ci, s, e)
    if best is None:
        return None
    return best[1], best[2], best[3], best[0]


def main():
    reactions = json.loads(EXTRACTION_JSON.read_text())
    norm_chunks = [normalize(c) for c in CHUNKS]

    # Per-chunk accumulators, keyed by chunk index.
    chunk_highlights = {i: [] for i in range(len(CHUNKS))}   # [(start, end)]
    chunk_anchors = {i: [] for i in range(len(CHUNKS))}      # [(start, end, rxn)]

    total_rxn = 0
    matched_rxn = 0
    unmatched = []

    for r in reactions:
        rxn = r["annotation_result"]
        total_rxn += 1
        ev_matches = []  # (chunk_idx, start, end, quality)
        for ev in rxn.get("evidence") or []:
            hit = locate_evidence(ev, norm_chunks)
            if hit is not None:
                ev_matches.append(hit)

        if ev_matches:
            matched_rxn += 1
            for ci, s, e, _q in ev_matches:
                chunk_highlights[ci].append((s, e))
            # Anchor the comment to the earliest evidence occurrence
            # (by chunk order, then position in the chunk).
            ev_matches.sort(key=lambda t: (t[0], t[1]))
            aci, as_, ae, _ = ev_matches[0]
            chunk_anchors[aci].append((as_, ae, rxn))
        else:
            unmatched.append((r.get("chunk_index"), rxn.get("name")))

    doc = Document()

    # ---- Title / preamble ----
    doc.add_heading("ZNFX1 — RESULTS reactions annotated onto 300-token chunks", level=0)
    intro = doc.add_paragraph()
    intro.add_run(
        "Each extracted reaction is anchored to the sentence it was pulled from "
        "(its evidence). Those sentences are "
    )
    hl = intro.add_run("highlighted")
    hl.font.highlight_color = WD_COLOR_INDEX.YELLOW
    intro.add_run(
        " in the chunk text below; open the linked Word comment on a highlight to "
        "see the reaction it produced. Source: "
    )
    intro.add_run(EXTRACTION_JSON.name).italic = True

    for ci, chunk_text in enumerate(CHUNKS):
        hay = norm_chunks[ci]
        highlight_spans = chunk_highlights[ci]
        anchors = chunk_anchors[ci]

        doc.add_heading(f"Chunk {ci}   ({len(anchors)} reaction(s))", level=1)

        # Build the set of cut points to segment the paragraph into runs.
        cuts = {0, len(hay)}
        for s, e in highlight_spans:
            cuts.add(s)
            cuts.add(e)
        for s, e, _ in anchors:
            cuts.add(s)
            cuts.add(e)
        cuts = sorted(c for c in cuts if 0 <= c <= len(hay))

        para = doc.add_paragraph()
        seg_runs = []  # list of (seg_start, seg_end, run)
        for a, b in zip(cuts, cuts[1:]):
            if b <= a:
                continue
            run = para.add_run(hay[a:b])
            if any(s <= a and b <= e for s, e in highlight_spans):
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            seg_runs.append((a, b, run))

        # Attach one comment per reaction, anchored to the runs covering its span.
        for s, e, rxn in anchors:
            runs = [run for (a, b, run) in seg_runs if a >= s and b <= e]
            if not runs:
                runs = [seg_runs[0][2]] if seg_runs else None
            if not runs:
                continue
            doc.add_comment(
                runs,
                text=build_comment(rxn),
                author="reaction-extractor",
                initials="RX",
            )

    # ---- Summary section ----
    doc.add_heading("Extraction summary", level=1)
    p = doc.add_paragraph()
    p.add_run(f"Reactions total: {total_rxn}\n")
    p.add_run(f"Anchored to a highlighted evidence sentence: {matched_rxn}\n")
    p.add_run(f"Evidence not found in the pasted text: {len(unmatched)}")
    if unmatched:
        for ci, name in unmatched:
            doc.add_paragraph(f"(json chunk {ci}) {name}", style="List Bullet")

    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_DOCX)

    print(f"Wrote {OUTPUT_DOCX}")
    print(f"Reactions: {total_rxn}  matched: {matched_rxn}  unmatched: {len(unmatched)}")
    for ci, name in unmatched:
        print(f"  UNMATCHED (json chunk {ci}): {name}")


if __name__ == "__main__":
    main()
